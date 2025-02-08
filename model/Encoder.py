import sys
sys.path.append('')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional
from model.AFF_block import HFF_block3D

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

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)

class main_model(nn.Module):
    def __init__(self, num_classes, patch_size=(2, 2, 2), in_chans=1, embed_dim=48, depths=(2, 2, 2, 2),
                 num_heads=(3, 6, 12, 24), window_size=(1, 5, 5), qkv_bias=True, drop_rate=0,
                 attn_drop_rate=0, drop_path_rate=0., norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, HFF_dp=0.,
                 conv_depths=(2, 2, 2, 2), conv_dims=(48, 96, 192, 384), conv_drop_path_rate=0.,
                 conv_head_init_scale: float = 1., pretrain_path=None):
        super().__init__()

        if pretrain_path is not None:
            self.load_pretrained_model(pretrain_path)
            print('Pretrained model loaded successfully')
        ###### Local Branch Setting #######

        self.downsample_layers = nn.ModuleList()   # stem + 3 stage downsample

        stem = nn.Sequential(nn.Conv3d(in_chans, conv_dims[0], kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                             LayerNorm(conv_dims[0], eps=1e-6, data_format="channels_first"))
        
        self.downsample_layers.append(stem)

        # stage2-4 downsample
        for i in range(3):
            downsample_layer = nn.Sequential(LayerNorm(conv_dims[i], eps=1e-6, data_format="channels_first"),
                                             nn.Conv3d(conv_dims[i], conv_dims[i+1], kernel_size=(2, 2, 2), stride=(2, 2, 2)))

            self.downsample_layers.append(downsample_layer)
        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple blocks
        dp_rates = [x.item() for x in torch.linspace(0, conv_drop_path_rate, sum(conv_depths))]
        cur = 0

        # Build stacks of blocks in each stage
        for i in range(4):
            stage = nn.Sequential(
                *[Local_block3D(dim=conv_dims[i], drop_rate=dp_rates[cur + j])
                  for j in range(conv_depths[i])]
            )
            self.stages.append((stage))
            cur += conv_depths[i]

        # self.conv_norm = nn.LayerNorm(conv_dims[-1], eps=1e-6)   # final norm layer
        # self.conv_head = nn.Linear(conv_dims[-1], num_classes)
        # self.conv_head.weight.data.mul_(conv_head_init_scale)
        # self.conv_head.bias.data.mul_(conv_head_init_scale)
        # self.classifier_l = nn.Sequential(
        #                 Conv3D(384, 384, 1, bn=True, relu=False),
        #                 nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten(),
        #                 nn.Linear(in_features=384, out_features=128, bias=True))
        # self.classifier_g = nn.Sequential(
        #         Conv3D(384, 384, 1, bn=True, relu=False),
        #         nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten(),
        #         nn.Linear(in_features=384, out_features=128, bias=True))
        self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten(),
                nn.Linear(in_features=384, out_features=128, bias=True))
        ###### Global Branch Setting ######

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm

        # The channels of stage4 output feature matrix
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        i_layer = 0
        self.layers1 = BasicLayer3D(dim=int(embed_dim * 2 ** i_layer),
                                    depth=depths[i_layer],
                                    num_heads=num_heads[i_layer],
                                    window_size=window_size,
                                    qkv_bias=qkv_bias,
                                    drop=drop_rate,
                                    attn_drop=attn_drop_rate,
                                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                    norm_layer=norm_layer,
                                    downsample=PatchMerging3D if (i_layer > 0) else None,
                                    use_checkpoint=use_checkpoint)

        i_layer = 1
        self.layers2 = BasicLayer3D(dim=int(embed_dim * 2 ** i_layer),
                                    depth=depths[i_layer],
                                    num_heads=num_heads[i_layer],
                                    window_size=window_size,
                                    qkv_bias=qkv_bias,
                                    drop=drop_rate,
                                    attn_drop=attn_drop_rate,
                                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                    norm_layer=norm_layer,
                                    downsample=PatchMerging3D if (i_layer > 0) else None,
                                    use_checkpoint=use_checkpoint)

        i_layer = 2
        self.layers3 = BasicLayer3D(dim=int(embed_dim * 2 ** i_layer),
                                    depth=depths[i_layer],
                                    num_heads=num_heads[i_layer],
                                    window_size=window_size,
                                    qkv_bias=qkv_bias,
                                    drop=drop_rate,
                                    attn_drop=attn_drop_rate,
                                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                    norm_layer=norm_layer,
                                    downsample=PatchMerging3D if (i_layer > 0) else None,
                                    use_checkpoint=use_checkpoint)

        i_layer = 3
        self.layers4 = BasicLayer3D(dim=int(embed_dim * 2 ** i_layer),
                                    depth=depths[i_layer],
                                    num_heads=num_heads[i_layer],
                                    window_size=window_size,
                                    qkv_bias=qkv_bias,
                                    drop=drop_rate,
                                    attn_drop=attn_drop_rate,
                                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                    norm_layer=norm_layer,
                                    downsample=PatchMerging3D if (i_layer > 0) else None,
                                    use_checkpoint=use_checkpoint)

        self.norm = norm_layer(self.num_features)

        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.avgpool = nn.AdaptiveAvgPool3d(1)

        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

        ###### Hierachical Feature Fusion Block Setting #######
        self.fu1 = HFF_block3D(ch_1=48, ch_2=48, r_2=8, ch_int=48, ch_out=48, drop_rate=HFF_dp)
        self.fu2 = HFF_block3D(ch_1=96, ch_2=96, r_2=8, ch_int=96, ch_out=96, drop_rate=HFF_dp)
        self.fu3 = HFF_block3D(ch_1=192, ch_2=192, r_2=8, ch_int=192, ch_out=192, drop_rate=HFF_dp)
        self.fu4 = HFF_block3D(ch_1=384, ch_2=384, r_2=8, ch_int=384, ch_out=384, drop_rate=HFF_dp)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.2)
            nn.init.constant_(m.bias, 0)
            
    def load_pretrained_model(self, pretrain_path):
        # load model
        state_dict = torch.load(pretrain_path)
        self.load_state_dict(state_dict, strict=False)
        
    def forward(self, imgs):
        ######  Global Branch ######
        x_s, D, H, W = self.patch_embed(imgs)
        #print(x_s.shape,D,H,W)
        x_s = self.pos_drop(x_s)
        #print(x_s.shape)
        x_s_1, D, H, W = self.layers1(x_s, D, H, W)
        #print(x_s_1.shape,D,H,W)
        x_s_2, D, H, W = self.layers2(x_s_1, D, H, W)
        #print(x_s_2.shape,D,H,W)
        x_s_3, D, H, W = self.layers3(x_s_2, D, H, W)
        #print(x_s_3.shape,D,H,W)
        x_s_4, D, H, W = self.layers4(x_s_3, D, H, W)
        #print(x_s_4.shape,D,H,W)
    

        # [B,L,C] ---> [B,C,D,H,W]
        D, H, W = 8, 32, 32
        x_s_1 = torch.transpose(x_s_1, 1, 2)
        x_s_1 = x_s_1.view(x_s_1.shape[0], x_s_1.shape[1], D, H, W)
        #print(x_s_1.shape)
        x_s_2 = torch.transpose(x_s_2, 1, 2)
        x_s_2 = x_s_2.view(x_s_2.shape[0], x_s_2.shape[1], D // 2, H // 2, W // 2)
        #print(x_s_2.shape)
        x_s_3 = torch.transpose(x_s_3, 1, 2)
        x_s_3 = x_s_3.view(x_s_3.shape[0], x_s_3.shape[1], D // 4, H // 4, W // 4)
        #print(x_s_3.shape)
        x_s_4 = torch.transpose(x_s_4, 1, 2)
        x_s_4 = x_s_4.view(x_s_4.shape[0], x_s_4.shape[1], D // 8, H // 8, W // 8)
       # print(x_s_4.shape)

        ######  Local Branch ######
        x_c = self.downsample_layers[0](imgs)
        #print(x_c.shape)
        x_c_1 = self.stages[0](x_c)
        #print(x_c_1.shape)
        x_c = self.downsample_layers[1](x_c_1)
        x_c_2 = self.stages[1](x_c)
        #print(x_c_2.shape)
        x_c = self.downsample_layers[2](x_c_2)
        x_c_3 = self.stages[2](x_c)
        #print(x_c_3.shape)
        x_c = self.downsample_layers[3](x_c_3)
        x_c_4 = self.stages[3](x_c)
        #print(x_c_4.shape)

        ###### Hierachical Feature Fusion Path ######
        x_f_1 = self.fu1(x_c_1, x_s_1, None)
        #print(x_f_1.shape)
        x_f_2 = self.fu2(x_c_2, x_s_2, x_f_1)
        #print(x_f_2.shape)
        x_f_3 = self.fu3(x_c_3, x_s_3, x_f_2)
        #print(x_f_3.shape)
        x_f_4 = self.fu4(x_c_4, x_s_4, x_f_3) # [4, 384, 1, 4, 4]
        #print(x_f_4.shape)
        x_fu = self.classifier(x_f_4) 
        
        ### Globle Feature ###
        x_g = self.classifier(x_s_4)
        ### Local Feature ###
        x_l = self.classifier(x_c_4)
        return x_fu, x_g, x_l

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


class Local_block3D(nn.Module):
    r""" Local Feature Block for 3D data. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, D, H, W)
    (2) DwConv -> Permute to (N, D, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_rate (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_rate=0.):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv = nn.Linear(dim, dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1)  # [N, C, D, H, W] -> [N, D, H, W, C]
        x = self.norm(x)
        x = self.pwconv(x)
        x = self.act(x)
        x = x.permute(0, 4, 1, 2, 3)  # [N, D, H, W, C] -> [N, C, D, H, W]
        x = shortcut + self.drop_path(x)
        return x
    

# # # Hierachical Feature Fusion Block
# class HFF_block3D(nn.Module):
#     def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out, drop_rate=0.):
#         super(HFF_block3D, self).__init__()
#         self.maxpool = nn.AdaptiveMaxPool3d(1)
#         self.avgpool = nn.AdaptiveAvgPool3d(1)
#         self.se = nn.Sequential(
#             nn.Conv3d(ch_2, ch_2 // r_2, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv3d(ch_2 // r_2, ch_2, 1, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()
#         self.spatial = Conv3D(2, 1, 7, bn=True, relu=False, bias=False)
#         self.W_l = Conv3D(ch_1, ch_int, 1, bn=True, relu=False)
#         self.W_g = Conv3D(ch_2, ch_int, 1, bn=True, relu=False)
#         self.Avg = nn.AvgPool3d(2, stride=2)
#         self.Updim = Conv3D(ch_int // 2, ch_int, 1, bn=True, relu=True)
#         self.norm1 = LayerNorm(ch_int * 3, eps=1e-6, data_format="channels_first")
#         self.norm2 = LayerNorm(ch_int * 2, eps=1e-6, data_format="channels_first")
#         self.norm3 = LayerNorm(ch_1 + ch_2 + ch_int, eps=1e-6, data_format="channels_first")
#         self.W3 = Conv3D(ch_int * 3, ch_int, 1, bn=True, relu=False)
#         self.W = Conv3D(ch_int * 2, ch_int, 1, bn=True, relu=False)

#         self.gelu = nn.GELU()

#         self.residual = IRMLP3D(ch_1 + ch_2 + ch_int, ch_out)
#         self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

#     def forward(self, l, g, f):

#         W_local = self.W_l(l)   # local feature from Local Feature Block
#         W_global = self.W_g(g)   # global feature from Global Feature Block
#         if f is not None:
#             W_f = self.Updim(f)
#             W_f = self.Avg(W_f)
#             shortcut = W_f
#             X_f = torch.cat([W_f, W_local, W_global], 1)
#             X_f = self.norm1(X_f)
#             X_f = self.W3(X_f)
#             X_f = self.gelu(X_f)
#         else:
#             shortcut = 0
#             X_f = torch.cat([W_local, W_global], 1)
#             X_f = self.norm2(X_f)
#             X_f = self.W(X_f)
#             X_f = self.gelu(X_f)

#         # spatial attention for ConvNeXt branch
#         l_jump = l
#         max_result, _ = torch.max(l, dim=1, keepdim=True)
#         avg_result = torch.mean(l, dim=1, keepdim=True)
#         result = torch.cat([max_result, avg_result], 1)
#         l = self.spatial(result)
#         l = self.sigmoid(l) * l_jump

#         # channel attention for transformer branch
#         g_jump = g
#         max_result = self.maxpool(g)
#         avg_result = self.avgpool(g)
#         max_out = self.se(max_result)
#         avg_out = self.se(avg_result)
#         g = self.sigmoid(max_out + avg_out) * g_jump

#         fuse = torch.cat([g, l, X_f], 1)
#         fuse = self.norm3(fuse)
#         fuse = self.residual(fuse)
#         fuse = shortcut + self.drop_path(fuse)
#         return fuse
    
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

#### Inverted Residual MLP
class IRMLP3D(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(IRMLP3D, self).__init__()
        self.conv1 = Conv3D(inp_dim, inp_dim, 3, relu=False, bias=False, group=inp_dim)
        self.conv2 = Conv3D(inp_dim, inp_dim * 4, 1, relu=False, bias=False)
        self.conv3 = Conv3D(inp_dim * 4, out_dim, 1, relu=False, bias=False, bn=True)
        self.gelu = nn.GELU()
        self.bn1 = nn.BatchNorm3d(inp_dim)

    def forward(self, x):

        residual = x
        out = self.conv1(x)
        out = self.gelu(out)
        out += residual

        out = self.bn1(out)
        out = self.conv2(out)
        out = self.gelu(out)
        out = self.conv3(out)

        return out


####### Shift Window MSA #############
class WindowAttention3D(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias for 3D data.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The depth, height, and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [Md, Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))  # [2*Md-1 * 2*Mh-1 * 2*Mw-1, nH]

        # get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_d, coords_h, coords_w], indexing="ij"))  # [3, Md, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [3, Md*Mh*Mw]
        # [3, Md*Mh*Mw, 1] - [3, 1, Md*Mh*Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [3, Md*Mh*Mw, Md*Mh*Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Md*Mh*Mw, Md*Mh*Mw, 3]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)  # [Md*Mh*Mw, Md*Mh*Mw]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, Md*Mh*Mw, C)
            mask: (0/-inf) mask with shape of (num_windows, Md*Mh*Mw, Md*Mh*Mw) or None
        """
        B_, N, C = x.shape
        # qkv(): -> [batch_size*num_windows, Md*Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Md*Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Md*Mh*Mw, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size*num_windows, num_heads, Md*Mh*Mw, embed_dim_per_head]
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Md*Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Md*Mh*Mw, Md*Mh*Mw]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias_table.view: [Md*Mh*Mw*Md*Mh*Mw,nH] -> [Md*Mh*Mw,Md*Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2], self.window_size[0] * self.window_size[1] * self.window_size[2], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Md*Mh*Mw, Md*Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: [nW, Md*Mh*Mw, Md*Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Md*Mh*Mw, Md*Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Md*Mh*Mw, Md*Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size*num_windows, num_heads, Md*Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Md*Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Md*Mh*Mw, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True, group=1):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x    

### Global Feature Block
class Global_block3D(nn.Module):
    r""" Global Feature Block from modified Swin Transformer Block for 3D data.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for SW-MSA.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=(1, 7, 7), shift_size=(0, 0, 0),
                 qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        assert all(0 <= s < w for s, w in zip(self.shift_size, self.window_size)), "shift_size must be in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = act_layer()

    def forward(self, x, attn_mask):
        D, H, W = self.D, self.H, self.W
        B, L, C = x.shape
        assert L == D * H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, D, H, W, C)

        pad_l = pad_t = pad_f = 0
        pad_r = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2]
        pad_b = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
        pad_d = (self.window_size[0] - D % self.window_size[0]) % self.window_size[0]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_f, pad_d))
        _, Dp, Hp, Wp, _ = x.shape

        # cyclic shift
        if any(self.shift_size):
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition3D(shifted_x, self.window_size)  # [nW*B, Md, Mh, Mw, C]
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)  # [nW*B, Md*Mh*Mw, C]

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Md*Mh*Mw, C]

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)  # [nW*B, Md, Mh, Mw, C]
        shifted_x = window_reverse3D(attn_windows, self.window_size, Dp, Hp, Wp)  # [B, D', H', W', C]

        # reverse cyclic shift
        if any(self.shift_size):
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0 or pad_d > 0:
            x = x[:, :D, :H, :W, :].contiguous()

        x = x.view(B, D * H * W, C)
        x = self.fc1(x)
        x = self.act(x)
        x = shortcut + self.drop_path(x)

        return x
    
class BasicLayer3D(nn.Module):
    """
    Downsampling and Global Feature Block for one stage for 3D data.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, num_heads, window_size=(1, 7, 7),
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2)

        # build blocks
        self.blocks = nn.ModuleList([
            Global_block3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def create_mask(self, x, D, H, W):
        # calculate attention mask for SW-MSA
        Dp = int(np.ceil(D / self.window_size[0])) * self.window_size[0]
        Hp = int(np.ceil(H / self.window_size[1])) * self.window_size[1]
        Wp = int(np.ceil(W / self.window_size[2])) * self.window_size[2]

        img_mask = torch.zeros((1, Dp, Hp, Wp, 1), device=x.device)  # [1, Dp, Hp, Wp, 1]
        d_slices = (slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None))
        h_slices = (slice(0, -self.window_size[1]),
                    slice(-self.window_size[1], -self.shift_size[1]),
                    slice(-self.shift_size[1], None))
        w_slices = (slice(0, -self.window_size[2]),
                    slice(-self.window_size[2], -self.shift_size[2]),
                    slice(-self.shift_size[2], None))
        cnt = 0
        for d in d_slices:
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1

        mask_windows = window_partition3D(img_mask, self.window_size)  # [nW, Md, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2])  # [nW, Md*Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Md*Mh*Mw] - [nW, Md*Mh*Mw, 1]
        # [nW, Md*Mh*Mw, Md*Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, D, H, W):

        if self.downsample is not None:
            x = self.downsample(x, D, H, W)  # patch merging stage2 in [B, D*H*W, C] out [B, D/2*H/2*W/2, 2*C]
            D, H, W = (D + 1) // 2, (H + 1) // 2, (W + 1) // 2

        attn_mask = self.create_mask(x, D, H, W)  # [nW, Md*Mh*Mw, Md*Mh*Mw]
        for blk in self.blocks:  # global block
            blk.D, blk.H, blk.W = D, H, W
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)

        return x, D, H, W


def window_partition3D(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size (Md, Mh, Mw)

    Returns:
        windows: (num_windows*B, Md, Mh, Mw, C)
    """
    B, D, H, W, C = x.shape
    Md, Mh, Mw = window_size
    x = x.view(B, D // Md, Md, H // Mh, Mh, W // Mw, Mw, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, Md, Mh, Mw, C)
    return windows



def window_reverse3D(windows, window_size, D, H, W):
    """
    Args:
        windows: (num_windows*B, Md, Mh, Mw, C)
        window_size (tuple[int]): Window size (Md, Mh, Mw)
        D (int): Depth of volume
        H (int): Height of volume
        W (int): Width of volume

    Returns:
        x: (B, D, H, W, C)
    """
    B = int(windows.shape[0] / (D * H * W / window_size[0] / window_size[1] / window_size[2]))
    Md, Mh, Mw = window_size
    x = windows.view(B, D // Md, H // Mh, W // Mw, Md, Mh, Mw, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x

class PatchEmbed3D(nn.Module):
    """
    3D Image to Patch Embedding
    """
    def __init__(self, patch_size=(1, 4, 4), in_c=1, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size  # 3D patch size
        self.in_chans = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)  # Conv3d for 3D data
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, D, H, W = x.shape  # D: depth, H: height, W: width

        # padding
        pad_input = (D % self.patch_size[0] != 0) or (H % self.patch_size[1] != 0) or (W % self.patch_size[2] != 0)
        if pad_input:
            # Padding to make dimensions divisible by patch_size
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2],
                          0, self.patch_size[1] - H % self.patch_size[1],
                          0, self.patch_size[0] - D % self.patch_size[0]))

        # downsample patch_size times
        x = self.proj(x)
        _, _, D, H, W = x.shape  # Get new dimensions after convolution

        # flatten: [B, C, D, H, W] -> [B, C, D*H*W]
        x = x.flatten(2)  # Flatten the depth, height, and width dimensions
        # transpose: [B, C, D*H*W] -> [B, D*H*W, C]
        x = x.transpose(1, 2)

        x = self.norm(x)
        return x, D, H, W



class PatchMerging3D(nn.Module):
    r""" Patch Merging Layer for 3D data.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        dim = dim // 2
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x, D, H, W):
        """
        x: B, D*H*W, C
        """
        B, L, C = x.shape
        assert L == D * H * W, "input feature has wrong size"

        x = x.view(B, D, H, W, C)

        # padding
        pad_input = (D % 2 == 1) or (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2, 0, D % 2))

        x0 = x[:, 0::2, 0::2, 0::2, :]  # [B, D/2, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, 0::2, :]  # [B, D/2, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, 0::2, :]  # [B, D/2, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, 0::2, :]  # [B, D/2, H/2, W/2, C]
        x4 = x[:, 0::2, 0::2, 1::2, :]  # [B, D/2, H/2, W/2, C]
        x5 = x[:, 1::2, 0::2, 1::2, :]  # [B, D/2, H/2, W/2, C]
        x6 = x[:, 0::2, 1::2, 1::2, :]  # [B, D/2, H/2, W/2, C]
        x7 = x[:, 1::2, 1::2, 1::2, :]  # [B, D/2, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # [B, D/2, H/2, W/2, 8*C]
        x = x.view(B, -1, 8 * C)  # [B, D/2*H/2*W/2, 8*C]

        x = self.norm(x)
        x = self.reduction(x)  # [B, D/2*H/2*W/2, 2*C]

        return x

def Encoder_Tiny(num_classes: int, pretrain_path=None):
    model = main_model(depths=(2, 2, 2, 2),
                     conv_depths=(2, 2, 2, 2),
                     num_classes=num_classes)
    if pretrain_path !=None:
        print('loading pretrained model {}'.format(pretrain_path))
        state_dict = torch.load(pretrain_path)
        model.load_state_dict(state_dict)
        print("-------- pre-train model load successfully --------")
    return model


def Encoder_Small(num_classes: int, pretrain_path=None):
    model = main_model(depths=(2, 2, 6, 2),
                     conv_depths=(2, 2, 6, 2),
                     num_classes=num_classes)
    if pretrain_path !=None:
        print('loading pretrained model {}'.format(pretrain_path))
        state_dict = torch.load(pretrain_path)
        model.load_state_dict(state_dict)
        print("-------- pre-train model load successfully --------")
    return model

def Encoder_Base(num_classes: int, pretrain_path=None):
    model = main_model(depths=(2, 2, 18, 2),
                     conv_depths=(2, 2, 18, 2),
                     num_classes=num_classes)
    if pretrain_path !=None:
        print('loading pretrained model {}'.format(pretrain_path))
        state_dict = torch.load(pretrain_path)
        model.load_state_dict(state_dict)
        print("-------- pre-train model load successfully --------")
    return model

if __name__ == '__main__':
    imgs=torch.randn(4, 1, 16, 64, 64)
    model = Encoder_Base(num_classes=128)
    output, Globle, Local= model(imgs)
    print(output.shape, Globle.shape, Local.shape)