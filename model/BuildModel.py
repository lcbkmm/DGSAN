import sys
sys.path.append('')
import numpy as np
import torch
from  torch import nn
from model.resnet_build import generate_model
from model.DeepSurv import DeepSurv
from torch_geometric.data import Data, DataLoader, Batch
from model.diffusion import ConditionalModel
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_dense_batch
import torch.nn.functional as F
from math import sqrt
from model.GAT import GATNet

def create_fully_connected_graph(num_nodes, t_emb, n_emb, text_embed):
    edge_index = [[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(t_emb.device)

    t_emb = t_emb.unsqueeze(0)
    n_emb = n_emb.unsqueeze(0)
    text_embed = text_embed.unsqueeze(0)

    x = torch.concat((t_emb,n_emb,text_embed), dim=0).to(t_emb.device)

    return Data(x=x, edge_index=edge_index)


device = torch.device('cuda')

class BuildModel(nn.Module):
    def __init__(self,num_class=10):
        super(BuildModel,self).__init__()
        self.resnet_t0 = generate_model(model_type='resnet', model_depth=50, 
                                        input_W=16, input_H=64, input_D=64, resnet_shortcut='B', 
                                        no_cuda=False, gpu_id=[0], 
                                        pretrain_path='outputs/resnet50_best.pth',
                                        nb_class=128)
        self.resnet_t1 = generate_model(model_type='resnet', model_depth=50, 
                                        input_W=16, input_H=64, input_D=64, resnet_shortcut='B', 
                                        no_cuda=False, gpu_id=[0], 
                                        pretrain_path='outputs/resnet50_best.pth',
                                        nb_class=128)

        self.cross_attn = CrossAttention(heads=4,d_model=64,tumor_dim=128,node_dim=128)

        # self.mlp_blood = DeepSurv(33) # 临床特征数量
        # self.mlp_t_radiomics = DeepSurv(32)
        # self.mlp_n_radiomics = DeepSurv(19)
        
        self.mlp_clinical = DeepSurv(6) # 临床特征数量

        self.GAT = GATNet(256, 64) # 利用图注意力网络得HGA过程

        self.fc = nn.Sequential(
            nn.Linear(192,1024),
            nn.SELU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024,256),
            nn.SELU(),
            nn.Dropout(p=0.2),
            nn.Linear(256,num_class)
        )

    def forward(self, input1=None, input2=None, text=None, concat_type='train'):
        # t_img = self.resnet_t(t_img)
        # n0 = self.resnet_n0(n_patch0)
        # n1 = self.resnet_n1(n_patch1)
        # n2 = self.resnet_n2(n_patch2)
        # node_embed = torch.concat((n0,n1,n2),dim=1).to(t_img.device)
        
        t0 = self.resnet_t0(input1)
        t1 = self.resnet_t1(input2)
        
        t_um,t_m,n_um,n_m,t_s,n_s = self.cross_attn(t0, t1)

        if concat_type == 'train':
            t_emb = torch.concat((t_m,t_s),dim=-1)
            n_emb = torch.concat((n_m,n_s),dim=-1)
        else:
            t_emb = torch.concat((t_um, t_s), dim=-1)
            n_emb = torch.concat((n_um, n_s), dim=-1)

        # blo_embed = self.mlp_blood(blood)
        # t_rad_embed = self.mlp_t_radiomics(t_radiomics)
        # n_rad_embed = self.mlp_n_radiomics(n_radiomics)
        text_embed = self.mlp_clinical(text)
        after_GAT = []

        for idx in range(t_emb.size(0)):
            data = create_fully_connected_graph(3, t_emb[idx,:], n_emb[idx,:], text_embed[idx,:])
            out = self.GAT(data).flatten().unsqueeze(0)
            after_GAT.append(out)

        after_GAT = torch.concat(after_GAT, dim=0).to(t_emb.device)

        out_logits = self.fc(after_GAT)

        return t_um, t_m, n_um, n_m, after_GAT, out_logits.softmax(dim=-1)

# test
if __name__ == '__main__':
        guide_model = BuildModel(num_class=2).to(device)
        input1 = torch.randn(4, 1, 16, 64, 64).to(device)
        input2 = torch.randn(4, 1, 16, 64, 64).to(device)
        table = torch.randn(4, 1, 6)
        table = torch.squeeze(table).to(device)
        t_um, t_m, n_um, n_m, feats, yhat = guide_model(input1=input1, input2=input2, text=table, concat_type='train')