from torch import nn
from typing import Optional

from src.core.learning.multiagent import MultiAgent
from src.core.experiments.base import FurnLiftBaseConfig
from src.core.learning.model import Model


# originally furnlift_vision_mixture_cl_config.py
# this config is using the Cordial Loss (cl) and SYNC Policies (abstracted from description on https://github.com/allenai/cordial-sync)
class FurnLiftMinDistMixtureConfig(FurnLiftBaseConfig):
    
    min_dist_between_agents_to_pickup = 8       # Env/episode config
    visible_agents = True
    turn_off_communication = False              # Model config
    agent_class = MultiAgent                    # Agent config
    coordinate_actions = True                   # Mixture (SYNC-Policies)
    _checkpoints_dir = ''                       # Path for checkpoints
    _use_checkpoint = ''                        # Path to specific checkpoint to proceed from

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return Model(
            num_inputs_per_agent=3 + 1 * (cls.include_depth_frame),
            action_groups=cls.episode_class.class_available_action_groups(),
            num_agents=cls.num_agents,
            state_repr_length=cls.state_repr_length,
            talk_embed_length=cls.talk_embed_length,
            agent_num_embed_length=cls.agent_num_embed_length,
            reply_embed_length=cls.reply_embed_length,
            num_talk_symbols=cls.num_talk_symbols,
            num_reply_symbols=cls.num_reply_symbols,
            turn_off_communication=cls.turn_off_communication,
            coordinate_actions=cls.coordinate_actions,
            coordinate_actions_dim=2 if cls.coordinate_actions else None,
            separate_actor_weights=False,
        )

    @classmethod
    def create_agent(cls, **kwargs) -> MultiAgent:
        return cls.agent_class(
            model=kwargs["model"],
            gpu_id=kwargs["gpu_id"],
            include_test_eval_results=cls.include_test_eval_results,
            use_a3c_loss_when_not_expert_forcing=cls.use_a3c_loss_when_not_expert_forcing,
            record_all_in_test=cls.record_all_in_test,
            include_depth_frame=cls.include_depth_frame,
            resize_image_as=cls.screen_size,
            dagger_mode=cls.dagger_mode,
            discourage_failed_coordination=False,
            coordination_use_marginal_entropy=True,
        )

    @property
    def checkpoints_dir(self) -> Optional[str]:
        return self._checkpoints_dir

    @checkpoints_dir.setter
    def checkpoints_dir(self, value):
        self._checkpoints_dir = value

    @property
    def use_checkpoint(self) -> Optional[str]:
        return self._use_checkpoint

    @use_checkpoint.setter
    def use_checkpoint(self, value):
        self._use_checkpoint = value

def get_experiment():
    return FurnLiftMinDistMixtureConfig()
