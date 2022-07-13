import itertools
import queue
import random
import warnings
import networkx as nx
import numpy as np
import torch
import sys

from ai2thor.platform import CloudRendering, Linux64, OSXIntel64    # needed in order to select with getattr(sys.modules[__name__], ...)
from networkx import find_cliques
from torch import multiprocessing as mp
from typing import List, Optional, Callable, Tuple

import src.core.utils.constants as CONSTANTS
from src.core.ai2thor.environment import AI2ThorEnvironmentWithGraph
from src.core.learning.episodes import JointNavigationEpisode
from src.core.learning.multiagent import MultiAgent
from src.core.services.birds_eye_view import BirdsEyeView
from src.core.services.viewer import Viewer
from src.core.utils.sampler import create_environment, save_talk_reply_data_frame
from src.core.utils.ai2thor import manhattan_dists_between_positions
from src.core.utils.multiagent import TrainingCompleteException


class FurnLiftEpisodeSamplers(object):
    def __init__(
        self,
        scenes: List[str],
        num_agents: int,
        object_type: str,
        env_args=None,
        max_episode_length: int = 500,
        episode_class: Callable = JointNavigationEpisode,
        player_screen_height=224,
        player_screen_width=224,
        save_talk_reply_probs_path: Optional[str] = None,
        min_dist_between_agents_to_pickup: int = 0,
        max_ep_using_expert_actions: int = 10000,
        visible_agents: bool = True,
        max_visible_positions_push: int = 8,
        min_max_visible_positions_to_consider: Tuple[int, int] = (8, 16),
        include_depth_frame: bool = False,
        allow_agents_to_intersect: bool = False,
    ):
        self.visible_agents = visible_agents
        self.max_ep_using_expert_actions = max_ep_using_expert_actions
        self.min_dist_between_agents_to_pickup = min_dist_between_agents_to_pickup
        self.save_talk_reply_probs_path = save_talk_reply_probs_path
        self.player_screen_height = player_screen_height
        self.player_screen_width = player_screen_width
        self.episode_class = episode_class
        self.max_episode_length = max_episode_length
        self.env_args = env_args
        self.num_agents = num_agents
        self.scenes = scenes
        self.object_type = object_type
        self.max_visible_positions_push = max_visible_positions_push
        self.min_max_visible_positions_to_consider = (
            min_max_visible_positions_to_consider
        )
        self.include_depth_frame = include_depth_frame
        self.allow_agents_to_intersect = allow_agents_to_intersect

        self.grid_size = 0.25

        self._current_train_episode = 0
        self._internal_episodes = 0

    @property
    def current_train_episode(self):
        return self._current_train_episode

    @current_train_episode.setter
    def current_train_episode(self, value):
        self._current_train_episode = value

    def _contain_n_positions_at_least_dist_apart(self, positions, n, min_dist):
        for sub_positions in itertools.combinations(positions, n):
            if all(
                x >= min_dist
                for x in sum(
                    manhattan_dists_between_positions(sub_positions, self.grid_size),
                    [],
                )
            ):
                return True

        return False

    def __call__(
        self, 
        agent: MultiAgent,
        agent_location_seed=None,
        env: Optional[AI2ThorEnvironmentWithGraph] = None,
        episode_init_queue: Optional[mp.Queue] = None,
    ) -> None:
        self._internal_episodes += 1

        if env is None:
            if agent.environment is not None:
                env = agent.environment
            else:
                env = create_environment(
                    num_agents=self.num_agents,
                    env_args=self.env_args,
                    visible_agents=self.visible_agents,
                    render_depth_image=self.include_depth_frame,
                    platform=getattr(sys.modules[__name__], self.env_args.platform),
                    allow_agents_to_intersect=self.allow_agents_to_intersect,
                )
                env.start(
                    "FloorPlan201_physics",
                    move_mag=self.grid_size,
                    quality="Very Low",
                    player_screen_height=self.player_screen_height,
                    player_screen_width=self.player_screen_width,
                )

                # event = env.step({
                #     'action': "GetMapViewCameraProperties"
                # })
                # env.step({
                #     'action': "AddThirdPartyCamera",
                #     'agentId': 1,
                #     **event.metadata["actionReturn"]
                # })
                # event = env.step(
                #     {
                #         "action": "Done",
                #         "agentId": 0,
                #     }
                # )
                #
                # birds_eye_view = BirdsEyeView()
                # birds_eye_view.update(event.events[0])
        if (
            self.max_ep_using_expert_actions != 0
            and self.current_train_episode <= self.max_ep_using_expert_actions
        ):
            agent.take_expert_action_prob = 0.9 * (
                1.0 - self.current_train_episode / self.max_ep_using_expert_actions
            )
        else:
            agent.take_expert_action_prob = 0

        if episode_init_queue is not None:
            try:
                self.episode_init_data = episode_init_queue.get(timeout=1)
            except queue.Empty:
                raise TrainingCompleteException("No more data in episode init queue.")

            scene = self.episode_init_data["scene"]
            env.reset(scene)
            assert self.object_type == "Television"

            env.step(
                {
                    "action": "DisableAllObjectsOfType",
                    "objectId": self.object_type,
                    "agentId": 0,
                }
            )

            for agent_id, agent_location in enumerate(
                self.episode_init_data["agent_locations"]
            ):
                env.teleport_agent_to(
                    **agent_location,
                    agent_id=agent_id,
                    only_initially_reachable=False,
                    force_action=True,
                )
                assert env.last_event.metadata["lastActionSuccess"]

            object_location = self.episode_init_data["object_location"]
            env.step(
                {
                    "action": "CreateObjectAtLocation",
                    "objectType": self.object_type,
                    **object_location,
                    "position": {
                        "x": object_location["x"],
                        "y": object_location["y"],
                        "z": object_location["z"],
                    },
                    "rotation": {"x": 0, "y": object_location["rotation"], "z": 0},
                    "agentId": 0,
                    "forceAction": True,
                }
            )
            assert env.last_event.metadata["lastActionSuccess"]

            env.refresh_initially_reachable()

            objects_of_type = env.all_objects_with_properties(
                {"objectType": self.object_type}, agent_id=0
            )
            if len(objects_of_type) != 1:
                print("len(objects_of_type): {}".format(len(objects_of_type)))
                raise (Exception("len(objects_of_type) != 1"))

            object = objects_of_type[0]
            object_id = object["objectId"]
            # obj_rot = int(object["rotation"]["y"])
            #
            # object_points_set = set(
            #     (
            #         round(object["position"]["x"] + t[0], 2),
            #         round(object["position"]["z"] + t[1], 2),
            #     )
            #     for t in CONSTANTS.TELEVISION_ROTATION_TO_OCCUPATIONS[obj_rot]
            # )

            # Find agent metadata from where the target is visible
            env.step(
                {
                    "action": "GetPositionsObjectVisibleFrom",
                    "objectId": object_id,
                    "agentId": 0,
                }
            )
            possible_targets = env.last_event.metadata["actionVector3sReturn"]
            distances = []
            for i, r in enumerate(env.last_event.metadata["actionFloatsReturn"]):
                possible_targets[i]["rotation"] = int(r)
                possible_targets[i]["horizon"] = 0
                distances.append(
                    env.position_dist(object["position"], possible_targets[i])
                )

            # So here the targets are sorted by distance in ascending order and it returns (x, z, rotation, horizon) using get_key and these values are rounded off
            possible_targets = [
                env.get_key(x[1])
                for x in sorted(
                    list(zip(distances, possible_targets)), key=lambda x: x[0]
                )
            ]

            if self.min_dist_between_agents_to_pickup != 0:
                possible_targets_array = np.array([t[:2] for t in possible_targets])
                manhat_dist_mat = np.abs(
                    (
                        (
                            possible_targets_array.reshape((-1, 1, 2))
                            - possible_targets_array.reshape((1, -1, 2))
                        )
                        / self.grid_size
                    ).round()
                ).sum(2)
                sufficiently_distant = (
                    manhat_dist_mat >= self.min_dist_between_agents_to_pickup
                )

                g = nx.Graph(sufficiently_distant)
                good_cliques = []
                for i, clique in enumerate(find_cliques(g)):
                    if i > 1000 or len(good_cliques) > 40:
                        break
                    if len(clique) == env.num_agents:
                        good_cliques.append(clique)
                    elif len(clique) > env.num_agents:
                        good_cliques.extend(
                            itertools.combinations(clique, env.num_agents)
                        )

                good_cliques = [
                    [possible_targets[i] for i in gc] for gc in good_cliques
                ]
            else:
                assert False, "Old code."

            if len(possible_targets) < env.num_agents:
                raise Exception(f"Using data from episode queue (scene {scene} and replication {self.episode_init_data['replication']}) but there seem to be no good final positions for the agents?")
            
            scene_successfully_setup = True
        else:

            scene = random.choice(self.scenes)
            env.reset(scene)
            scene_successfully_setup = False
            failure_reasons = []
            for _ in range(10):

                env.step(
                    {
                        "action": "DisableAllObjectsOfType",
                        "objectId": self.object_type,
                        "agentId": 0,
                    }
                )

                # CHANGE: instead get Television and disable by using action "DisableObject"
                objects_of_type = env.all_objects_with_properties(
                    {"objectType": self.object_type}, agent_id=0
                )
                if len(objects_of_type) != 1:
                    print("len(objects_of_type): {}".format(len(objects_of_type)))
                    raise (Exception("len(objects_of_type) != 1"))
                env.step(
                    {
                        "action": "DisableObject",
                        "objectId": objects_of_type[0]["objectId"],
                        "agentId": 0,
                    }
                )

                for agent_id in range(self.num_agents):
                    env.randomize_agent_location(
                        agent_id=agent_id,
                        seed=agent_location_seed[agent_id]
                        if agent_location_seed
                        else None,
                        partial_position={"horizon": 0},
                        only_initially_reachable=True,
                    )

                env.step(
                    {
                        "action": "RandomlyCreateAndPlaceObjectOnFloor",
                        "objectType": self.object_type,
                        "agentId": 0,
                    }
                )

                if env.last_event.metadata["lastAction"] != "RandomlyCreateAndPlaceObjectOnFloor" or not env.last_event.metadata["lastActionSuccess"]:
                    failure_reasons.append("Could not randomize location of {}.".format(self.object_type, scene))
                    continue

                env.refresh_initially_reachable()

                objects_of_type = env.all_objects_with_properties(
                    {"objectType": self.object_type}, agent_id=0
                )
                if len(objects_of_type) != 1:
                    print("len(objects_of_type): {}".format(len(objects_of_type)))
                    raise (Exception("len(objects_of_type) != 1"))

                object = objects_of_type[0]
                object_id = object["objectId"]
                #
                # obj_rot = int(object["rotation"]["y"])    # CHANGE: needs round, to return 90° instead of 89°!
                # object_points_set = set(
                #     (
                #         round(object["position"]["x"] + t[0], 2),
                #         round(object["position"]["z"] + t[1], 2),
                #     )
                #     for t in CONSTANTS.TELEVISION_ROTATION_TO_OCCUPATIONS[obj_rot]
                # )

                for agent_id in range(self.num_agents):
                    env.randomize_agent_location(
                        agent_id=agent_id,
                        seed=agent_location_seed[agent_id]
                        if agent_location_seed
                        else None,
                        partial_position={"horizon": 0},
                        only_initially_reachable=True,
                    )

                env.step(
                    {
                        "action": "GetReachablePositions",
                        "agentId": 0,
                    }
                )
                # Find agent metadata from where the target is visible
                env.step(
                    {
                        "action": "GetPositionsObjectVisibleFrom",
                        "objectId": object_id,
                        "agentId": 0,
                    }
                )
                possible_targets = env.last_event.metadata["actionVector3sReturn"]
                distances = []
                for i, r in enumerate(env.last_event.metadata["actionFloatsReturn"]):
                    possible_targets[i]["rotation"] = int(r)
                    possible_targets[i]["horizon"] = 0
                    distances.append(
                        env.position_dist(object["position"], possible_targets[i])
                    )

                possible_targets = [
                    env.get_key(x[1])
                    for x in sorted(
                        list(zip(distances, possible_targets)), key=lambda x: x[0]
                    )
                ]

                if self.min_dist_between_agents_to_pickup != 0:
                    # In this, it takes only the x and z value
                    possible_targets_array = np.array([t[:2] for t in possible_targets])
                    manhat_dist_mat = np.abs(
                        (
                            (
                                possible_targets_array.reshape((-1, 1, 2))
                                - possible_targets_array.reshape((1, -1, 2))
                            )
                            / self.grid_size
                        ).round()
                    ).sum(2)
                    sufficiently_distant = (
                        manhat_dist_mat >= self.min_dist_between_agents_to_pickup
                    )

                    g = nx.Graph(sufficiently_distant)
                    good_cliques = []
                    for i, clique in enumerate(find_cliques(g)):
                        if i > 1000 or len(good_cliques) > 40:
                            break
                        if len(clique) == env.num_agents:
                            good_cliques.append(clique)
                        elif len(clique) > env.num_agents:
                            good_cliques.extend(
                                itertools.combinations(clique, env.num_agents)
                            )
                    
                    if len(good_cliques) == 0:
                        # print(possible_targets_array)
                        # print('=========================')
                        # print(manhat_dist_mat)
                        failure_reasons.append(
                            "Failed to find a tuple of {} targets all {} steps apart in scene {} having possible targets {}".format(
                                env.num_agents, self.min_dist_between_agents_to_pickup, env.scene_name, manhat_dist_mat
                            )
                        )
                        continue

                    good_cliques = [
                        [possible_targets[i] for i in gc] for gc in good_cliques
                    ]
                else:
                    assert False, "Old code."
                    (
                        min_considered,
                        max_considered,
                    ) = self.min_max_visible_positions_to_consider
                    offset0 = min(
                        max(len(possible_targets) - min_considered, 0),
                        self.max_visible_positions_push,
                    )
                    # offset1 = min(
                    #     max(len(possible_targets) - max_considered, 0),
                    #     self.max_visible_positions_push,
                    # )
                    min_slice = slice(offset0, offset0 + min_considered)
                    # max_slice = slice(offset1, offset1 + max_considered)

                    possible_targets = possible_targets[min_slice]

                    good_cliques = list(
                        itertools.combinations(possible_targets, self.num_agents)
                    )

                if len(possible_targets) < env.num_agents:
                    failure_reasons.append(
                        "The number of positions from which the object was visible was less than the number of agents."
                    )
                    continue
                scene_successfully_setup = True
                break

        if not scene_successfully_setup:
            warnings.warn(
                (
                    "Failed to randomly initialize objects and agents in scene {} 10 times"
                    + "for the following reasons:"
                    + "\n\t* ".join(failure_reasons)
                    + "\nTrying a new scene."
                ).format(env.scene_name)
            )
            yield from self(agent, env=env)
            return

        # Task data includes the target object id and one agent state/metadata to navigate to.
        good_cliques = {tuple(sorted(gc)) for gc in good_cliques}
        task_data = {"goal_obj_id": object_id, "target_key_groups": good_cliques}

        agent.episode = self.episode_class(
            env,
            task_data,
            max_steps=self.max_episode_length,
            num_agents=self.num_agents,
            min_dist_between_agents_to_pickup=self.min_dist_between_agents_to_pickup,
            #object_points_set=object_points_set,
            include_depth_frame=self.include_depth_frame,
        )

        yield True

        if self.save_talk_reply_probs_path is not None:
            if torch.cuda.is_available():
                save_talk_reply_data_frame(
                    agent,
                    self.save_talk_reply_probs_path,
                    None,
                    use_hash=True,  # self._internal_episodes,
                )
            else:
                save_talk_reply_data_frame(
                    agent, self.save_talk_reply_probs_path, self._internal_episodes
                )
