import sys
sys.path.append('./')
import numpy as np
import torch
from torch.nn import init
from torch import nn
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_dense_batch
import torch.nn.functional as F
from math import sqrt
from model.GAT import GATNet
from model.HCMGF import Fusion
from model.Encoder import Encoder_Small

def create_fully_connected_graph1(num_nodes, t0, t1, text_embed):
    edge_index = [[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(text_embed.device)

    t0_emb = t0.unsqueeze(0)
    t1_emb = t1.unsqueeze(0)
    text_embed = text_embed.unsqueeze(0)

    x = torch.concat((t0_emb,t1_emb,text_embed), dim=0).to(text_embed.device)

    return Data(x=x, edge_index=edge_index)

def create_fully_connected_graph2(num_nodes, t0_l, t0_g, t1_l, t1_g):
    edge_index = [[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(t0_l.device)

    t0_l_emb = t0_l.unsqueeze(0)
    t0_g_emb = t0_g.unsqueeze(0)
    t1_l_emb = t1_l.unsqueeze(0)
    t1_g_emb = t1_g.unsqueeze(0)

    x = torch.concat((t0_l_emb,t0_g_emb,t1_l_emb,t1_g_emb), dim=0).to(t0_l.device)

    return Data(x=x, edge_index=edge_index)

device = torch.device('cuda')

class DeepSurv(nn.Module):
    def __init__(self,in_dim):
        super(DeepSurv, self).__init__()
        self.deepsurv = nn.Sequential(
            nn.Linear(in_dim,32),
            nn.BatchNorm1d(32),
            nn.ELU(alpha=0.5),
            nn.Dropout(p=0.22),
            nn.Linear(32,64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(p=0.22),
            nn.Linear(64,128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(p=0.22)
        )

        for m in self.modules():
            if isinstance(m,nn.Linear):
                init.kaiming_uniform_(m.weight,mode='fan_out',nonlinearity='relu')

    def forward(self,clinical_features):
        out3 = self.deepsurv(clinical_features)
        return out3

class BuildModel(nn.Module):
    def __init__(self, num_class=2, pretrain_path=None):
        super(BuildModel,self).__init__()
        self.Encoder1 = Encoder_Small(num_classes=128, pretrain_path=pretrain_path)
        self.Encoder2 = Encoder_Small(num_classes=128, pretrain_path=pretrain_path)
        self.mlp_clinical = DeepSurv(6) # 临床特征数量
        self.GAT1 = GATNet(in_channels=128, out_channels=64) # 利用图注意力网络得HGA过程
        self.GAT2 = GATNet(in_channels=128, out_channels=48)
        self.fusion = Fusion(input_dim=192, num_layer=3)
        self.fc = nn.Sequential(
            nn.Linear(192,1024),
            nn.SELU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024,128),
            nn.SELU(),
            nn.Dropout(p=0.2),
            nn.Linear(128,num_class)
        )

    def forward(self, input1=None, input2=None, text=None, concat_type='train'):
        t0,t0_g,t0_l = self.Encoder1(input1)
        t1,t1_g,t1_l = self.Encoder2(input2)
        
        text_embed = self.mlp_clinical(text)
        after_GAT_multimodal = []
        after_GAT_global_local = []

        # 创建每个模态的图
        # 多模态
        for idx in range(t0.size(0)):
            data = create_fully_connected_graph1(3, t0[idx,:],  t1[idx,:],  text_embed[idx,:])
            out = self.GAT1(data).flatten().unsqueeze(0)
            after_GAT_multimodal.append(out)
        
        # 全局+局部
        for idx in range(t0.size(0)):
            data = create_fully_connected_graph2(4,  t0_l[idx,:], t0_g[idx,:], t1_l[idx,:], t1_g[idx,:])
            out = self.GAT2(data).flatten().unsqueeze(0)
            after_GAT_global_local.append(out)

        after_GAT_multimodal = torch.cat(after_GAT_multimodal, dim=0).to(t0.device)
        after_GAT_global_local = torch.cat(after_GAT_global_local, dim=0).to(t0.device)
        
        # 权重合并特征
        combined_features = self.fusion(after_GAT_multimodal, after_GAT_global_local)
        out_logits = self.fc(combined_features)

        # return after_GAT, out_logits.softmax(dim=-1)
        return out_logits.softmax(dim=-1)

# test
if __name__ == '__main__':
    guide_model = BuildModel(num_class=2, pretrain_path='/home/wangchangmiao/yuxiao/Lung_nodule/MMFusion-main/outputs/Encoder_Small_myHFF.pth').to(device)
    input1 = torch.randn(4, 1, 16, 64, 64).to(device)
    input2 = torch.randn(4, 1, 16, 64, 64).to(device)
    table = torch.randn(4, 1, 6)
    table = torch.squeeze(table).to(device)
    yhat = guide_model(input1=input1, input2=input2, text=table, concat_type='train')
    print(yhat.shape)