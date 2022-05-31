import hashlib
import math
import numpy as np
import os
import pandas as pd
from typing import Dict, List, Optional

from cordialsync.ai2thor.environment import AI2ThorEnvironmentWithGraph


def create_environment(
    num_agents,
    env_args,
    visible_agents: bool,
    render_depth_image: bool,
    headless: bool = False,
    **environment_args,
) -> AI2ThorEnvironmentWithGraph:
    env = AI2ThorEnvironmentWithGraph(
        docker_enabled=False,
        num_agents=num_agents,
        restrict_to_initially_reachable_points=True,
        visible_agents=visible_agents,
        render_depth_image=render_depth_image,
        headless=headless,
        **environment_args,
    )
    return env

def create_or_append(dict: Dict[str, List], key, value):
    if key not in dict:
        dict[key] = [value]
    else:
        dict[key].append(value)

def save_talk_reply_data_frame(
    agent, save_path, index: Optional[int], use_hash: bool = False, prefix: str = ""
):
    num_agents = agent.environment.num_agents
    eval_results = agent.eval_results

    object_id = agent.episode.object_id
    o = agent.environment.get_object_by_id(object_id, agent_id=0)
    o_pos = {**o["position"], "rotation": o["rotation"]["y"]}

    data = {}
    for i in range(len(eval_results)):
        er = eval_results[i]
        for k in er:
            if "cpu" in dir(er[k]):
                er[k] = er[k].cpu()
        sr = er["step_result"]

        for agent_id in range(num_agents):
            agent_key = "agent_{}_".format(agent_id)

            for probs_key in ["talk_probs", "reply_probs"]:
                probs = er[probs_key][agent_id, :].detach().numpy()
                for j in range(len(probs)):
                    create_or_append(
                        data, agent_key + probs_key + "_" + str(j), probs[j]
                    )

            sr_i = sr[agent_id]
            for key in ["goal_visible", "pickup_action_taken", "action_success"]:
                create_or_append(data, agent_key + key, sr_i[key])

            create_or_append(
                data,
                agent_key + "action",
                agent.episode.available_actions[sr_i["action"]],
            )

            before_loc = sr_i["before_location"]
            for key in before_loc:
                create_or_append(data, agent_key + key, before_loc[key])

            create_or_append(
                data,
                agent_key + "l2_dist_to_object",
                math.sqrt(
                    (o_pos["x"] - before_loc["x"]) ** 2
                    + (o_pos["z"] - before_loc["z"]) ** 2
                ),
            )
            create_or_append(
                data,
                agent_key + "manhat_dist_to_object",
                abs(o_pos["x"] - before_loc["x"]) + abs(o_pos["z"] - before_loc["z"]),
            )

        for key in o_pos:
            create_or_append(data, "object_" + key, o_pos[key])

        mutual_agent_distance = int(
            4
            * (
                abs(data["agent_0_x"][-1] - data["agent_1_x"][-1])
                + abs(data["agent_0_z"][-1] - data["agent_1_z"][-1])
            )
        )
        create_or_append(data, "mutual_agent_distance", mutual_agent_distance)

    df = pd.DataFrame(
        data={**data, "scene_name": [agent.environment.scene_name] * len(eval_results)}
    )
    for df_k in df.keys():
        if df[df_k].dtype in [float, np.float32, np.float64]:
            df[df_k] = np.round(df[df_k], 4)
    if use_hash:
        file_name = (
            agent.environment.scene_name
            + "_"
            + hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()
            + ".tsv"
        )
    else:
        file_name = "{}_{}_talk_reply_data.tsv".format(index, prefix)

    df.to_csv(
        os.path.join(save_path, file_name),
        sep="\t",
        # float_format="%.4f",
    )