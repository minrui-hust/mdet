import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from mai.utils import FI
from mai.model import BaseModule
from mai.model.utils.construct_mask import construct_mask


@FI.register
class MMTrans(BaseModule):
    def __init__(self,
                 hparam,
                 agent_enc,
                 agent_dec,
                 lane_net,
                 lane_enc,
                 lane_dec,
                 social_enc,
                 head,
                 ):
        super().__init__()

        self.hparam = hparam
        model_dim = hparam['model_dim']
        pos_dim = hparam['pos_dim']
        dist_dim = hparam['dist_dim']
        lane_enc_dim = hparam['lane_enc_dim']
        agent_dim = hparam['agent_dim']
        K = hparam['num_queries']
        dropout = hparam['dropout']

        self.agent_enc = FI.create(agent_enc)
        self.agent_dec = FI.create(agent_dec)
        self.lane_net = FI.create(lane_net)
        self.lane_enc = FI.create(lane_enc)
        self.lane_dec = FI.create(lane_dec)
        self.social_enc = FI.create(social_enc)
        self.head = FI.create(head)

        self.pos_mlp = FI.create(dict(type='MLP',
                                      in_channels=2,
                                      hidden_channels=pos_dim,
                                      out_channels=pos_dim))

        self.agent_mlp = FI.create(dict(type='MLP',
                                        in_channels=model_dim + pos_dim,
                                        hidden_channels=model_dim,
                                        out_channels=model_dim))

        self.lane_mlp = FI.create(dict(type='MLP',
                                       in_channels=K * model_dim,
                                       hidden_channels=dist_dim,
                                       out_channels=dist_dim))

        self.social_mlp = FI.create(dict(type='MLP',
                                         in_channels=dist_dim + pos_dim,
                                         hidden_channels=model_dim,
                                         out_channels=model_dim))

        self.lane_emb = LinearEmbedding(lane_enc_dim, model_dim)
        self.agent_emb = LinearEmbedding(agent_dim, model_dim)
        self.agent_pos_enc = PositionalEncoding(model_dim, dropout)

        self.query = nn.Embedding(K, model_dim)

        self.initialize()

    def initialize(self):
        for _, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

        self.query.weight.requires_grad == False
        nn.init.orthogonal_(self.query.weight)

    def forward_train(self, batch):
        agent = batch['input']['agent']  # (B, A, 19, 4)
        lane = batch['input']['lane']  # (B, L, 9, 7)
        pos = batch['input']['pos']  # (B, A, 2)
        agent_num = batch['input']['agent_num']  # (B)
        lane_num = batch['input']['lane_num']  # (B)

        # batch_size, max_agent_num, max_lane_num
        B, A, L, K = agent.shape[0], agent.shape[1], lane.shape[1], self.hparam['num_queries']

        agent_mask = construct_mask(agent_num, A, inverse=True)  # (B, A)
        lane_mask = construct_mask(lane_num, L, inverse=True)  # (B, L)

        agent = self.agent_emb(agent)
        agent = self.agent_pos_enc(agent)  # (B, A, 19, model_dim)
        # (B*A, 19, model_dim)
        agent = self.agent_enc(agent.view(-1, *agent.shape[-2:]))

        lane = self.lane_net(lane)  # (B, L, 64)
        lane = self.lane_emb(lane)  # (B, L, model_dim)
        lane = self.lane_enc(lane, mask=lane_mask)  # (B, L, model_dim)

        # (B, A, model_dim)
        pos = self.pos_mlp(pos)

        # (B*A, K, model_dim)
        query_batches = self.query.weight.unsqueeze(
            0).expand(agent.shape[0], -1, -1)

        # (B*A, K, model_dim)
        agent_out = self.agent_dec(query_batches, agent)

        # (B*A, K, pos_dim)
        pos_expand = pos.view(-1, 1, pos.shape[-1]).expand(-1, K, -1)
        lane_in = self.agent_mlp(
            torch.cat([agent_out, pos_expand], dim=-1))  # (B*A, K, model_dim)

        # (B*A, K, model_dim)
        lane_out = self.lane_dec(lane_in, torch.repeat_interleave(
            lane, A, dim=0, output_size=B*A), mask=torch.repeat_interleave(lane_mask, A, dim=0, output_size=B*A))

        # (B, A, model_dim)
        social_in = self.lane_mlp(lane_out.view(B, A, -1))
        social_in = self.social_mlp(torch.cat([social_in, pos], dim=-1))

        # (B, A, model_dim)
        social_out = self.social_enc(social_in, mask=agent_mask)

        # (B, A, K, model_dim*2)
        head_in = torch.cat([social_out.unsqueeze(
            2).expand(-1, -1, K, -1), lane_out.view(B, A, K, -1)], dim=-1)

        head_out = self.head(head_in)

        return head_out


@FI.register
class LaneNet(nn.Module):
    def __init__(self, in_channels, hidden_unit, layer_num):
        super(LaneNet, self).__init__()
        self.layer_list = nn.ModuleList()
        for i in range(layer_num):
            self.layer_list.append(FI.create(dict(type='MLP', in_channels=in_channels,
                                             hidden_channels=hidden_unit, out_channels=hidden_unit)))
            in_channels = hidden_unit*2

    def forward(self, x):
        '''
            Extract lane_feature from vectorized lane representation

        Args:
            lane: [batch size, max_lane_num, 9, 7] (vectorized representation)

        Returns:
            x_max: [batch size, max_lane_num, 64]
        '''
        for layer in self.layer_list:
            # x [bs,max_lane_num,9,dim]
            x = layer(x)
            x_max = torch.max(x, -2)[0]
            x_max_expand = x_max.unsqueeze(2).repeat(1, 1, x.shape[2], 1)
            x = torch.cat([x, x_max_expand], dim=-1)
        return x_max


class LinearEmbedding(nn.Module):
    def __init__(self, inp_size, d_model):
        super(LinearEmbedding, self).__init__()
        # lut => lookup table
        self.lut = nn.Linear(inp_size, d_model, bias=True)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Implement the PE function.
    """

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        x = x + self.pe[:x.shape[-2]]
        return self.dropout(x)
