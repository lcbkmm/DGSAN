import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class AGCA3D(nn.Module):
    def __init__(self, in_channel, ratio):
        super(AGCA3D, self).__init__()
        hide_channel = in_channel // ratio
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv1 = nn.Conv3d(in_channel, hide_channel, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.A0 = torch.eye(hide_channel)
        # A2 is initialized to 1e-6
        self.A2 = nn.Parameter(torch.FloatTensor(torch.zeros((hide_channel, hide_channel))), requires_grad=True)
        init.constant_(self.A2, 1e-6)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv3d(hide_channel, in_channel, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv1(y)
        B, C, _, _, _ = y.size()
        y = y.flatten(2).transpose(1, 2)
        A1 = self.softmax(self.conv2(y))
        A1 = A1.expand(B, C, C)
        A = (self.A0 * A1) + self.A2
        y = torch.matmul(y, A)
        y = self.relu(self.conv3(y))
        y = y.transpose(1, 2).view(-1, C, 1, 1, 1)
        y = self.sigmoid(self.conv4(y))

        return x * y
    
class Conv3D(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True, group=1):
        super(Conv3D, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv3d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias, groups=group)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm3d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    

def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, depth, height, width]
            mean = x.mean([2, 3, 4], keepdim=True)
            var = (x - mean).pow(2).mean([2, 3, 4], keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x

# Hierachical Feature Fusion Block
class HFF_block3D(nn.Module):
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out, drop_rate=0.):
        super(HFF_block3D, self).__init__()
        self.W_l = Conv3D(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_g = Conv3D(ch_2, ch_int, 1, bn=True, relu=False)
        
        self.Avg = nn.AvgPool3d(2, stride=2)
        self.Updim = Conv3D(ch_int // 2, ch_int, 1, bn=True, relu=True)
        self.norm1 = LayerNorm(ch_int * 3, eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(ch_int * 2, eps=1e-6, data_format="channels_first")
        # 定义卷积层
        self.conv_layer = nn.Conv3d(in_channels=2*ch_int, out_channels=2, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=1)
        self.W3 = Conv3D(ch_int * 3, ch_int, 1, bn=True, relu=False)
        self.W = Conv3D(ch_int * 2, ch_int, 1, bn=True, relu=False)

        self.gelu = nn.GELU()

        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, l, g, f):

        f_local = self.W_l(l)   # local feature from Local Feature Block
        f_global = self.W_g(g)   # global feature from Global Feature Block
        if f is not None:
            f_f = self.Updim(f)
            f_f = self.Avg(f_f)
            shortcut = f_f
            X_f = torch.cat([f_f, f_local, f_global], 1)
            X_lg = torch.cat([f_local, f_global], 1)
            X_lg = self.conv_layer(X_lg)
            X_lg = self.softmax(X_lg)
            
            X_f = self.norm1(X_f)
            X_f = self.W3(X_f)
            X_f = self.gelu(X_f)

            wights = torch.split(X_lg, 1, dim=1)
            W_local = wights[0]
            W_global = wights[1]
            fuse = W_local*f_local + W_global*f_global + X_f
            fuse = shortcut + self.drop_path(fuse)
            
        else:
            shortcut = 0
            X_lg = torch.cat([f_local, f_global], 1)
            X_lg = self.conv_layer(X_lg)
            X_lg = self.softmax(X_lg)
            
            wights = torch.split(X_lg, 1, dim=1)
            W_local = wights[0]
            W_global = wights[1]
            fuse = W_local*f_local + W_global*f_global
            
            # X_lg = self.norm2(X_lg)
            # X_lg = self.W(X_lg)
            # X_f = self.gelu(X_lg)
            

        return fuse
    
if __name__ == '__main__':
    x_c_1 = torch.randn(4, 96, 4, 16, 16)
    x_s_1 = torch.randn(4, 96, 4, 16, 16)
    feature = torch.randn(4, 48, 8, 32, 32)
    fu1 = HFF_block3D(ch_1=96, ch_2=96, r_2=8, ch_int=96, ch_out=96, drop_rate=0.2)
    x_f_1 = fu1(x_c_1, x_s_1, None)  # [4, 96, 4, 16, 16]
    print(x_f_1.shape)