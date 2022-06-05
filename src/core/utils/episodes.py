import frozendict
import numpy as np
from enum import Enum
from typing import Sequence, Tuple, Callable

import src.core.utils.constants as CONSTANTS
from src.core.ai2thor.environment import AI2ThorEnvironment
from src.core.utils.misc import all_equal


class CoordType(Enum):
    # DO NOT CHANGE THIS WITHOUT CHANGING coordination_type_tensor to match
    # INDIVIDUAL should be the smallest value and equal 0
    INDIVIDUAL = 0

    ROTATE_LIFTED = 1

    MOVE_LIFTED_CARD = 2
    MOVE_LIFTED_EGO = 3

    MOVE_WITH_LIFTED_CARD = 4
    MOVE_WITH_LIFTED_EGO = 5

    PICKUP = 6


ACTION_TO_COORD_TYPE = frozendict.frozendict(
    {
        "Pass": CoordType.INDIVIDUAL,
        #
        **{
            "Move{}".format(dir): CoordType.INDIVIDUAL
            for dir in CONSTANTS.CARD_DIR_STRS + CONSTANTS.EGO_DIR_STRS
        },
        **{"Rotate{}".format(dir): CoordType.INDIVIDUAL for dir in ["Left", "Right"]},
        #
        **{
            "RotateLiftedObject{}".format(dir): CoordType.ROTATE_LIFTED
            for dir in ["Left", "Right"]
        },
        #
        **{
            "MoveLiftedObject{}".format(dir): CoordType.MOVE_LIFTED_CARD
            for dir in CONSTANTS.CARD_DIR_STRS
        },
        #
        **{
            "MoveAgents{}WithObject".format(dir): CoordType.MOVE_WITH_LIFTED_CARD
            for dir in CONSTANTS.CARD_DIR_STRS
        },
        #
        **{
            "MoveLiftedObject{}".format(dir): CoordType.MOVE_LIFTED_EGO
            for dir in CONSTANTS.EGO_DIR_STRS
        },
        #
        **{
            "MoveAgents{}WithObject".format(dir): CoordType.MOVE_WITH_LIFTED_EGO
            for dir in CONSTANTS.EGO_DIR_STRS
        },
        "Pickup": CoordType.PICKUP,
    }
)

COORDINATION_TYPE_TENSOR_CACHE = {}


def rotate_clockwise(l, n):
    return l[-n:] + l[:-n]


def are_actions_coordinated(env: AI2ThorEnvironment, action_strs: Sequence[str]):
    action_types = [ACTION_TO_COORD_TYPE[a] for a in action_strs]
    if not all_equal(action_types):
        return False

    action_type = action_types[0]

    if action_type == CoordType.INDIVIDUAL:
        return True

    if action_type in [
        CoordType.ROTATE_LIFTED,
        CoordType.MOVE_LIFTED_CARD,
        CoordType.MOVE_WITH_LIFTED_CARD,
        CoordType.PICKUP,
    ]:
        return all_equal(action_strs)
    elif action_type in [CoordType.MOVE_LIFTED_EGO, CoordType.MOVE_WITH_LIFTED_EGO]:
        action_relative_ind = [None] * env.num_agents
        for i, action in enumerate(action_strs):
            for j, dir in enumerate(CONSTANTS.EGO_DIR_STRS):
                if dir in action:
                    action_relative_ind[i] = j
                    break
            if action_relative_ind[i] is None:
                raise RuntimeError("Ego action but no ego dir in action name?")

        agent_rot_inds = [
            round(env.get_agent_location(agent_id)["rotation"] / 90)
            for agent_id in range(env.num_agents)
        ]

        return all_equal(
            [
                int(dir_rel_ind + agent_rot_ind) % 4
                for dir_rel_ind, agent_rot_ind in zip(
                    action_relative_ind, agent_rot_inds
                )
            ]
        )
    else:
        raise NotImplementedError(
            "Cannot determine if {} actions are coordinated.".format(action_strs)
        )


def coordination_type_tensor(
    env,
    action_strings: Tuple[str],
    action_coordination_checker: Callable[[AI2ThorEnvironment, Sequence[str]], bool],
):
    agent_rot_inds = tuple(
        round(env.get_agent_location(i)["rotation"] / 90) % 4
        for i in range(env.num_agents)
    )

    key = (agent_rot_inds, action_strings, action_coordination_checker)
    if key in COORDINATION_TYPE_TENSOR_CACHE:
        return COORDINATION_TYPE_TENSOR_CACHE[key]

    coord_tensor = np.full(
        (len(action_strings),) * env.num_agents, fill_value=-1, dtype=int
    )

    for ind in range(np.product(coord_tensor.shape)):
        multi_ind = np.unravel_index(ind, coord_tensor.shape)
        multi_action = tuple(action_strings[i] for i in multi_ind)

        if action_coordination_checker(env, multi_action):
            coord_tensor[multi_ind] = int(ACTION_TO_COORD_TYPE[multi_action[0]].value)

    COORDINATION_TYPE_TENSOR_CACHE[key] = coord_tensor

    return coord_tensor
