import copy
import logging
import math

from os.path import join as pjoin
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
import torch
import torch.nn as nn
import numpy as np

def swish(x):
    return x * torch.sigmoid(x)
ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Attention(nn.Module):
    def __init__(self, input_dim, vis, mode=None):
        super(Attention, self).__init__()
        self.vis = vis
        self.mode = mode
        self.num_attention_heads = 4
        self.attention_head_size = int(input_dim / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(input_dim, self.all_head_size)
        self.key = nn.Linear(input_dim, self.all_head_size)
        self.value = nn.Linear(input_dim, self.all_head_size)
        self.out = nn.Linear(input_dim, input_dim)

        self.attn_dropout = nn.Dropout(0.2)
        self.proj_dropout = nn.Dropout(0.2)
        
        self.queryd = nn.Linear(input_dim, self.all_head_size)
        self.keyd = nn.Linear(input_dim, self.all_head_size)
        self.valued = nn.Linear(input_dim, self.all_head_size)
        self.outd = nn.Linear(input_dim, input_dim)

        if self.mode == 'mba':
            self.w11 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.w12 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.w21 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.w22 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.w11.data.fill_(0.5)
            self.w12.data.fill_(0.5)
            self.w21.data.fill_(0.5)
            self.w22.data.fill_(0.5)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1)

    def forward(self, hidden_statesx, hidden_statesy):
        mixed_query_layer = self.query(hidden_statesx)
        mixed_key_layer = self.key(hidden_statesx)
        mixed_value_layer = self.value(hidden_statesx)

        mixed_queryd_layer = self.queryd(hidden_statesy)
        mixed_keyd_layer = self.keyd(hidden_statesy)
        mixed_valued_layer = self.valued(hidden_statesy)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        queryd_layer = self.transpose_for_scores(mixed_queryd_layer)
        keyd_layer = self.transpose_for_scores(mixed_keyd_layer)
        valued_layer = self.transpose_for_scores(mixed_valued_layer)

        ## Self Attention x: Qx, Kx, Vx
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_sx = self.out(context_layer)
        attention_sx = self.proj_dropout(attention_sx)
        
        ## Self Attention y: Qy, Ky, Vy
        attention_scores = torch.matmul(queryd_layer, keyd_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, valued_layer)
        context_layer = context_layer.permute(0, 2, 1).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_sy = self.outd(context_layer)
        attention_sy = self.proj_dropout(attention_sy)
        
        if self.mode == 'mba':
            ## Cross Attention x: Qx, Ky, Vy
            attention_scores = torch.matmul(query_layer, keyd_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            attention_probs = self.softmax(attention_scores)
            attention_probs = self.attn_dropout(attention_probs)

            context_layer = torch.matmul(attention_probs, valued_layer)
            context_layer = context_layer.permute(0, 2, 1).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)
            attention_cx = self.out(context_layer)
            attention_cx = self.proj_dropout(attention_cx)
            
            ## Cross Attention y: Qy, Kx, Vx
            attention_scores = torch.matmul(queryd_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            attention_probs = self.softmax(attention_scores)
            attention_probs = self.attn_dropout(attention_probs)

            context_layer = torch.matmul(attention_probs, value_layer)
            context_layer = context_layer.permute(0, 2, 1).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)
            attention_cy = self.outd(context_layer)
            attention_cy = self.proj_dropout(attention_cy)
    
            # Adaptative MBA
            attention_sx = self.w11 * attention_sx + self.w12 * attention_cx
            attention_sy = self.w21 * attention_sy + self.w22 * attention_cy
        
        return attention_sx, attention_sy

class Mlp(nn.Module):
    def __init__(self, input_dim):
        super(Mlp, self).__init__()
        self.fc1 = Linear(input_dim, input_dim*4)
        self.fc2 = Linear(input_dim*4, input_dim)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(0.2)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    
class Block(nn.Module):
    def __init__(self, input_dim, vis, mode=None):
        super(Block, self).__init__()
        self.attention_norm = LayerNorm(input_dim, eps=1e-6)
        self.attention_normd = LayerNorm(input_dim, eps=1e-6)
        self.ffn_norm = LayerNorm(input_dim, eps=1e-6)
        self.ffn_normd = LayerNorm(input_dim, eps=1e-6)
        self.ffn = Mlp(input_dim)
        self.ffnd = Mlp(input_dim)
        self.attn = Attention(input_dim, vis, mode=mode)

    def forward(self, x, y):
        hx = x
        hy = y
        x = self.attention_norm(x)
        y = self.attention_normd(y)
        x, y = self.attn(x, y)
        x = x + hx
        y = y + hy

        hx = x
        hy = y
        x = self.ffn_norm(x)
        y = self.ffn_normd(y)
        x = self.ffn(x)
        y = self.ffnd(y)
        x = x + hx
        y = y + hy
        return x, y
    
class Fusion(nn.Module):
    def __init__(self, input_dim, num_layer, vis=False):
        super(Fusion, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(input_dim, eps=1e-6)
        self.encoder_normd = LayerNorm(input_dim, eps=1e-6)
        for i in range(num_layer):
            # if i < 3 or i > 8:
            if i < 1 or i > 1:
                layer = Block(input_dim, vis, mode='sa')
            else:
                layer = Block(input_dim, vis, mode='mba')
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_statesx, hidden_statesy):
        for layer_block in self.layer:
            hidden_statesx, hidden_statesy = layer_block(hidden_statesx, hidden_statesy)
        encodedx = self.encoder_norm(hidden_statesx)
        encodedy = self.encoder_normd(hidden_statesy)
        fusion = encodedx + encodedy
        return fusion
    
    
if __name__ == '__main__':
    x = torch.randn(4, 128)
    y = torch.randn(4, 128)
    model = Fusion(input_dim=128, num_layer=7)
    fusion = model(x, y)
    print(fusion.shape)