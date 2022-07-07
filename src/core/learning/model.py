from __future__ import division

from collections import OrderedDict
from typing import Tuple, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.learning.clip_prediction import get_clip_encoding

from src.core.utils.misc import norm_col_init, weights_init, outer_product, outer_sum


def _unfold_communications(speech_per_agent: torch.FloatTensor):
    assert len(speech_per_agent.shape) >= 2
    num_agents = speech_per_agent.shape[0]

    unfolded_commns = [speech_per_agent[1:, :].view(1, -1)]
    for i in range(1, num_agents - 1):
        unfolded_commns.append(
            torch.cat(
                (
                    speech_per_agent[(i + 1) :,].view(1, -1),
                    speech_per_agent[:i,].view(1, -1),
                ),
                dim=1,
            )
        )
    unfolded_commns.append(speech_per_agent[:-1,].view(1, -1))

    return torch.cat(unfolded_commns, dim=0)


# originally BigA3CLSTMNStepComCoordinatedActionsEgoVision
class Model(nn.Module):
    def __init__(
        self,
        num_inputs_per_agent: int,
        action_groups: Tuple[Tuple[str, ...], ...],
        num_agents: int,
        state_repr_length: int,
        talk_embed_length: int,
        agent_num_embed_length: int,
        reply_embed_length: int,
        num_talk_symbols: int,
        num_reply_symbols: int,
        turn_off_communication: bool,
        coordinate_actions: bool,
        coordinate_actions_dim: Optional[int] = None,
        central_critic: bool = False,
        separate_actor_weights: bool = False,
        final_cnn_channels: int = 64,
    ):
        super(Model, self).__init__()
        self.num_outputs = sum(len(x) for x in action_groups)

        self.turn_off_communication = turn_off_communication
        self.central_critic = central_critic
        self.num_inputs_per_agent = num_inputs_per_agent
        self.num_agents = num_agents
        self.num_talk_symbols = num_talk_symbols
        self.num_reply_symbols = num_reply_symbols
        self.separate_actor_weights = separate_actor_weights
        self.coordinate_actions_dim = (
            self.num_outputs
            if coordinate_actions_dim is None
            else coordinate_actions_dim
        )

        self.coordinate_actions = coordinate_actions

        # input to conv is (num_agents, self.num_inputs_per_agent, 84, 84)
        print('Num of agents', num_agents)

        # self.cnn = nn.Sequential(
        #     OrderedDict(
        #         [
        #             # shape = 3x84x84
        #             (
        #                 "conv1",
        #                 nn.Conv2d(
        #                     self.num_inputs_per_agent, 32, 5, stride=1, padding=2
        #                 ),
        #             ),
        #             # shape = 32x84x84
        #             ("maxpool1", nn.MaxPool2d(2, 2)),
        #             ("relu1", nn.ReLU(inplace=True)),
        #             # shape = 32x42x42
        #             ("conv2", nn.Conv2d(32, 32, 5, stride=1, padding=1)),
        #             # shape = 32x40x40
        #             ("maxpool2", nn.MaxPool2d(2, 2)),
        #             # shape = 32x20x20
        #             ("relu2", nn.ReLU(inplace=True)),
        #             # shape = 32x20x20
        #             ("conv3", nn.Conv2d(32, 64, 4, stride=1, padding=1)),
        #             # shape = 64x19x19
        #             ("maxpool3", nn.MaxPool2d(2, 2)),
        #             # shape = 64x9x9
        #             ("relu3", nn.ReLU(inplace=True)),
        #             # shape = 64x9x9
        #             (
        #                 "conv4",
        #                 nn.Conv2d(64, final_cnn_channels, 3, stride=1, padding=1),
        #             ),
        #             # shape = 64x9x9
        #             ("maxpool4", nn.MaxPool2d(2, 2)),
        #             # shape = 64x4x4
        #             ("relu4", nn.ReLU(inplace=True)),
        #             # shape = (4, 4)
        #         ]
        #     )
        # )
        # Vocab embed
        self.talk_embeddings = nn.Embedding(num_talk_symbols, talk_embed_length)
        self.reply_embeddings = nn.Embedding(num_reply_symbols, reply_embed_length)

        self.talk_symbol_classifier = nn.Linear(state_repr_length, num_talk_symbols)
        self.reply_symbol_classifier = nn.Linear(state_repr_length, num_reply_symbols)

        # Agent embed (MLP)
        self.agent_num_embeddings = nn.Parameter(
            torch.rand(self.num_agents, agent_num_embed_length)
        )

        # LSTM
        self.lstm = nn.LSTM(
            final_cnn_channels * 4 * 4 + agent_num_embed_length,
            state_repr_length,
            batch_first=True,
        )

        # Belief update MLP
        state_and_talk_rep_len = state_repr_length + talk_embed_length * (
            num_agents - 1
        )
        self.after_talk_mlp = nn.Sequential(
            nn.Linear(state_and_talk_rep_len, state_and_talk_rep_len),
            nn.ReLU(inplace=True),
            nn.Linear(state_and_talk_rep_len, state_repr_length),
            nn.ReLU(inplace=True),
        )
        state_and_reply_rep_len = state_repr_length + reply_embed_length * (
            num_agents - 1
        )
        self.after_reply_mlp = nn.Sequential(
            nn.Linear(state_and_reply_rep_len, state_and_reply_rep_len),
            nn.ReLU(inplace=True),
            nn.Linear(state_and_reply_rep_len, state_repr_length),
            nn.ReLU(inplace=True),
        )

        if coordinate_actions:
            # Randomization MLP
            self.to_randomization_logits = nn.Sequential(
                nn.Linear(self.num_agents * reply_embed_length, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, self.coordinate_actions_dim),
            )

            # self.marginal_linear_actor = nn.Linear(state_repr_length, self.num_outputs)
            # self.marginal_linear_actor.weight.data = norm_col_init(
            #     self.marginal_linear_actor.weight.data, 0.01
            # )
            # self.marginal_linear_actor.bias.data.fill_(0)

        # Linear actor
        self.actor_linear = None
        if coordinate_actions:
            if separate_actor_weights:
                self.actor_linear_list = nn.ModuleList(
                    [
                        nn.Linear(
                            state_repr_length,
                            self.num_outputs * self.coordinate_actions_dim,
                        )
                        for _ in range(2)
                    ]
                )
            else:
                self.actor_linear = nn.Linear(
                    state_repr_length, self.num_outputs * self.coordinate_actions_dim
                )
        else:
            assert not separate_actor_weights
            self.actor_linear = nn.Linear(state_repr_length, self.num_outputs)

        if self.actor_linear is not None:
            self.actor_linear.weight.data = norm_col_init(
                self.actor_linear.weight.data, 0.01
            )
            self.actor_linear.bias.data.fill_(0)
        else:
            for al in self.actor_linear_list:
                al.weight.data = norm_col_init(al.weight.data, 0.01)
                al.bias.data.fill_(0)

        # Linear critic
        if self.central_critic:
            self.critic_linear = nn.Linear(state_repr_length * self.num_agents, 1)
        else:
            self.critic_linear = nn.Linear(state_repr_length, 1)

        # Setting initial weights
        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain("relu")
        # self.cnn._modules["conv1"].weight.data.mul_(relu_gain)
        # self.cnn._modules["conv2"].weight.data.mul_(relu_gain)
        # self.cnn._modules["conv3"].weight.data.mul_(relu_gain)
        # self.cnn._modules["conv4"].weight.data.mul_(relu_gain)

        self.talk_symbol_classifier.weight.data = norm_col_init(
            self.talk_symbol_classifier.weight.data, 0.01
        )
        self.talk_symbol_classifier.bias.data.fill_(0)
        self.reply_symbol_classifier.weight.data = norm_col_init(
            self.reply_symbol_classifier.weight.data, 0.01
        )
        self.reply_symbol_classifier.bias.data.fill_(0)

        self.train()

    def forward(
        self,
        inputs: torch.FloatTensor,
        hidden: Optional[torch.FloatTensor],
        agent_rotations: Sequence[int],
    ):
        if inputs.shape != (self.num_agents, self.num_inputs_per_agent, 84, 84):
            raise Exception("input to model is not as expected, check!")

        # x = self.cnn(inputs)
        # print(x.shape, '----1')  # 2x64x4x4
        # x.shape == (2, 128, 2, 2)


        x = get_clip_encoding(inputs)
        print(x.shape, '----1')  #
        # x.shape == (2, 64, 4, 4)

        x = x.view(x.size(0), -1)
        print(x.shape, '----2')
        # x.shape = [num_agents, 512]

        x = torch.cat((x, self.agent_num_embeddings), dim=1)
        print(x.shape, '----3')
        # x.shape = [num_agents, 512 + agent_num_embed_length]

        x, hidden = self.lstm(x.unsqueeze(1), hidden)
        print(x.shape, '----4')

        # x.shape = [num_agents, 1, state_repr_length]
        # hidden[0].shape == [1, num_agents, state_repr_length]
        # hidden[1].shape == [1, num_agents, state_repr_length]

        x = x.squeeze(1)
        print(x.shape, '----6')
        # x.shape = [num_agents, state_repr_length]

        talk_logits = self.talk_symbol_classifier(x)
        talk_probs = F.softmax(talk_logits, dim=1)
        # talk_probs.shape = [num_agents, num_talk_symbols]

        talk_outputs = torch.mm(talk_probs, self.talk_embeddings.weight)
        # talk_outputs.shape = [num_agents, talk_embed_length]

        if not self.turn_off_communication:
            talk_heard_per_agent = _unfold_communications(talk_outputs)
        else:
            talk_heard_per_agent = torch.cat(
                [talk_outputs] * (self.num_agents - 1), dim=1
            )

        state_talk_repr = x + self.after_talk_mlp(
            torch.cat((x, talk_heard_per_agent), dim=1)
        )
        # feature_talk_repr.shape = [num_agents, state_repr_length]

        reply_logits = self.reply_symbol_classifier(state_talk_repr)
        reply_probs = F.softmax(reply_logits, dim=1)
        # reply_probs.shape = [num_agents, num_reply_symbols]

        reply_outputs = torch.mm(reply_probs, self.reply_embeddings.weight)
        # reply_outputs.shape = [num_agents, reply_embed_length]

        if not self.turn_off_communication:
            reply_heard_per_agent = _unfold_communications(reply_outputs)
        else:
            reply_heard_per_agent = torch.cat(
                [reply_outputs] * (self.num_agents - 1), dim=1
            )

        state_talk_reply_repr = state_talk_repr + self.after_reply_mlp(
            torch.cat((state_talk_repr, reply_heard_per_agent), dim=1)
        )
        # state_talk_reply_repr.shape = [num_agents, state_repr_length]

        # Strangely we have to unsqueeze(0) instead of unsqueeze(1) here.
        # This seems to be because the LSTM is not expecting its hidden
        # state to have the batch first despite the batch_first=True parameter
        # making it expect its input to have the batch first.
        hidden = (state_talk_reply_repr.unsqueeze(0), hidden[1])
        if self.central_critic:
            value_all = self.critic_linear(
                torch.cat(
                    [
                        state_talk_reply_repr,
                        _unfold_communications(state_talk_reply_repr),
                    ],
                    dim=1,
                )
            )
        else:
            value_all = self.critic_linear(state_talk_reply_repr)

        to_return = {
            "value_all": value_all,
            "hidden_all": hidden,
            "talk_probs": talk_probs,
            "reply_probs": reply_probs,
        }

        if self.coordinate_actions:
            to_return["randomization_logits"] = self.to_randomization_logits(
                reply_heard_per_agent.view(1, -1)
            )
            if not self.separate_actor_weights:
                logits = self.actor_linear(state_talk_reply_repr)
            else:
                logits = torch.cat(
                    [
                        linear(state_talk_reply_repr[i].unsqueeze(0))
                        for i, linear in enumerate(self.actor_linear_list)
                    ],
                    dim=0,
                )

            # logits = self.actor_linear(
            #     state_talk_reply_repr
            # ) + self.marginal_linear_actor(state_talk_reply_repr).unsqueeze(1).repeat(
            #     1, self.coordinate_actions_dim, 1
            # ).view(
            #     self.num_agents, self.num_outputs ** 2
            # )
            to_return["coordinated_logits"] = logits
        else:
            to_return["logit_all"] = self.actor_linear(state_talk_reply_repr)

        return to_return


# def predict_clip_cs(inputs):
#     predict_clip(..)