import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.init as init
import torch.nn.functional as F

from einops import rearrange, repeat
from torch import einsum, softmax

from models.models_others import SoftAttn
from einops.layers.torch import Rearrange

# -------------Initialization----------------------------------------
from helpers import make_patches

patch_size = 32


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                # variance_scaling_initializer(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, dilation=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, dilation=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, dilation=3, padding=3)
        self.conv4 = nn.Conv2d(in_channels // 2, out_channels, kernel_size=3, padding=1)
        self.CA = ChannelAttention(in_channels // 2)

    def forward(self, x):
        # x1 = F.leaky_relu(self.conv1(x))
        # x2 = F.leaky_relu(self.conv2(x1))
        # x3 = F.leaky_relu(self.conv3(x1))
        # x4 = torch.cat([x3, x2], dim=1)
        # x4 = x1 + x4
        # x = self.conv4(x4)
        x1 = F.leaky_relu(self.conv1(x))
        # x1 = x1 + x1 * Highpass3
        x2 = F.leaky_relu(self.conv2(x1))
        # x2 = x2 * Highpass3
        x3 = F.leaky_relu(self.conv3(x1))
        # x3 = x3 * Highpass3
        x4 = torch.cat([x3, x2], dim=1)
        # x4 = Highpass3 * x4
        x5 = self.CA(x4)
        x6 = x1 + x5
        out = self.conv4(x6)

        return out


class TrEncoderStg1(nn.Module):
    def __init__(self, num_channel, num_feature):
        super(TrEncoderStg1, self).__init__()
        ####################

        self.T_l1 = Transformer_l(dim=num_feature, window_size=800)
        # self.T_l2 = Transformer_l(dim=num_feature * 2, window_size=500)

        self.Embedding = nn.Sequential(
            # nn.Linear(num_channel+1,num_feature),  # changed 3  to 1
            # nn.Conv1d(num_channel + 1, num_feature, kernel_size=3, stride=1, padding=1),
            # nn.ReLU()  # 可以加上激活函数，如 ReLU
            nn.Conv2d(num_channel + 1, num_feature, kernel_size=3, padding=1),
            # SeparableConv2d(num_channel + 1, num_feature, kernel_size=3, padding=1),
            nn.PReLU()  # 可以加上激活函数，如 ReLU
        )
        self.lin1 = nn.Linear(num_feature, num_feature * 2)
        # self.lin2 = nn.Linear(num_feature * 2, num_feature)

    def forward(self, x):
        sz = x.size(2)
        E = self.Embedding(x)
        # E1 = self.conv_div1(E)
        # E2 = self.conv_div2(E)
        # E_cat = torch.cat([E1, E2], dim=1)

        E_re = rearrange(E, 'B c H W -> B (H W) c', H=sz)
        att1 = self.T_l1(E_re)
        # Code = self.T_l2(E_re)
        Highpass0 = rearrange(att1, 'B (H W) C -> B C H W', H=sz)
        Highpass0 = E * Highpass0
        Highpass1 = Highpass0 + E
        lin = self.lin1(att1)

        return Highpass1, lin


class TrEncoderStg2(nn.Module):
    def __init__(self, num_feature):
        super(TrEncoderStg2, self).__init__()
        ####################

        # self.T_l1 = Transformer_l(dim=num_feature, window_size=500)
        self.T_l2 = Transformer_l(dim=num_feature * 2, window_size=500)
        self.encode_conv = nn.Sequential(
            # nn.Linear(num_channel+1,num_feature),  # changed 3  to 1
            # nn.Conv1d(num_channel + 1, num_feature, kernel_size=3, stride=1, padding=1),
            # nn.ReLU()  # 可以加上激活函数，如 ReLU
            nn.Conv2d(num_feature, num_feature * 2, kernel_size=3, padding=1),
            # SeparableConv2d(num_feature, num_feature * 2, kernel_size=3, padding=1),
            nn.PReLU()  # 可以加上激活函数，如 ReLU
        )

        # self.lin1 = nn.Linear(num_feature, num_feature * 2)
        # self.lin2 = nn.Linear(num_feature * 2, num_feature)

    def forward(self, x, lin):
        sz = x.size(2)

        # if flag == 1:   # when in original scale.
        #     x = x + up
        conv2 = self.encode_conv(x)
        E_re2 = rearrange(conv2, 'B c H W -> B (H W) c', H=sz)
        E_re2 = E_re2 + lin
        att2 = self.T_l2(E_re2)
        # Code2 = self.T_g(E_re2)
        Highpass2 = rearrange(att2, 'B (H W) C -> B C H W', H=sz)
        conv3 = conv2 * Highpass2
        conv3 = conv2 + conv3
        return conv3


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y


# --------------------------Main------------------------------- #

class MainNet(nn.Module):

    def __init__(self):
        super(MainNet, self).__init__()
        num_channel = 8
        ################ decoder ################
        self.tr_enc_stg1 = TrEncoderStg1(num_channel,
                                         24)  ###__init__(self, num_channel, num_feature), out_channel = 2* num_feature
        self.tr_enc_stg2 = TrEncoderStg2(24)

        self.tr_enc_stg1_d = TrEncoderStg1(num_channel,
                                           24)  ###__init__(self, num_channel, num_feature), out_channel = 2* num_feature
        self.tr_enc_stg2_d = TrEncoderStg2(24)

        self.decoder = Decoder(48, 24)  ##__init__(self, in_channels, out_channels)
        self.conv_out = nn.Conv2d(24, num_channel, kernel_size=3, padding=1)
        self.up_conv1 = UpsampleBLock(48, 4)


    def forward(self, ms_up, ms_org, pan):
        # data1 = torch.cat([ms_up, pan], dim=1)
        ################LR-HSI###################

        UP_LRHSI = ms_up
        sz = UP_LRHSI.size(2)

        pan_d = F.interpolate(pan, scale_factor=(1 / 4, 1 / 4), mode='bilinear')
        ms_d = F.interpolate(ms_org, scale_factor=(1 / 4, 1 / 4), mode='bilinear')
        ms_d_up = F.interpolate(ms_d, scale_factor=(4, 4), mode='bilinear')

        Input = torch.cat((UP_LRHSI, pan), 1)

        #### first stage of original scale
        out_en_st1, lin = self.tr_enc_stg1(Input)
        # torch.save(out_en_st1, 'visualize\\wv3_35_st1.pt')  ###　change here!! ############
        #### first stage of down scale
        Input_d = torch.cat((pan_d, ms_d_up), 1)
        out_en_st1_d, lin_d = self.tr_enc_stg1_d(Input_d)
        # torch.save(out_en_st1_d, 'visualize\\wv3_35_st1_d.pt')  ###　change here!! ############

        #### cross attetnion
        out1_d_up = F.interpolate(out_en_st1_d, scale_factor=(4, 4), mode='bilinear')
        out_en_st1 = out_en_st1 + out1_d_up
        # torch.save(out_en_st1, 'visualize\\wv3_35_st1_crs.pt')  ###　change here!! ############
        out_en_st1_d2 = F.interpolate(out_en_st1, scale_factor=(1 / 4, 1 / 4), mode='bilinear')
        out_en_st1_d = out_en_st1_d + out_en_st1_d2
        # torch.save(out_en_st1_d, 'visualize\\wv3_35_st1_d_crs.pt')  ###　change here!! ############

        #### second stage of down scale
        out_en_st2_d = self.tr_enc_stg2_d(out_en_st1_d, lin_d)
        out_2_up = F.interpolate(out_en_st2_d, scale_factor=(4, 4), mode='bilinear')
        # torch.save(out_2_up, 'visualize\\wv3_35_st2_d.pt')  ###　change here!! ############
        #### second stage of org scale
        out_en_st2 = self.tr_enc_stg2(out_en_st1, lin)
        # torch.save(out_en_st2, 'visualize\\wv3_35_st2.pt')  ###　change here!! ############
        out_mix = out_en_st2 + out_2_up  ## change here !!
        # torch.save(out_mix, 'visualize\\wv3_35_st2_mix.pt')  ###　change here!! ############
        # out_mix = out_en_st2

        out_de = self.decoder(out_mix)
        out_de = self.conv_out(out_de)

        # out_de_d = self.decoder(out_en_d)
        # out_de_d = self.conv_out(out_de_d)

        output = out_de + UP_LRHSI
        return output


# -----------------Transformer-----------------

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class LocalAttention1(nn.Module):
    def __init__(self, dim, heads, dim_head, window_size=512, max_range=1024, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.window_size = window_size
        self.max_range = max_range

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # Compute the mask
        mask = torch.ones(n, n).bool().to(x.device)
        for i in range(n):
            mask[i, max(0, i - self.max_range):i - self.window_size] = False
            mask[i, i + self.window_size + 1:min(n, i + self.max_range + 1)] = False

        # Compute the attention weights
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = rearrange(mask, 'i j -> () () i j')
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


def get_EF(input_size, dim, method="learnable", head_dim=None, bias=True):
    """
    Retuns the E or F matrix, initialized via xavier initialization.
    This is the recommended way to do it according to the authors of the paper.
    Includes a method for convolution, as well as a method for no additional params.
    """
    assert method == "learnable" or method == "convolution" or method == "no_params", "The method flag needs to be either 'learnable', 'convolution', or 'no_params'!"
    if method == "convolution":
        conv = nn.Conv1d(head_dim, head_dim, kernel_size=int(input_size / dim), stride=int(input_size / dim))
        return conv
    if method == "no_params":
        mat = torch.zeros((input_size, dim))
        torch.nn.init.normal_(mat, mean=0.0, std=1 / dim)
        return mat
    lin = nn.Linear(input_size, dim, bias)
    torch.nn.init.xavier_normal_(lin.weight)
    return lin


class LinformerAttention(nn.Module):
    def __init__(self, dim=48, heads=3, dim_head=16, dropout=0., k=240):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.k = k if k is not None else dim_head

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_head = nn.Linear(dim, inner_dim, bias=False)

        # reshape the output of self.to_qkv
        self.reshape = Rearrange('b n (head dim_head) -> b head n dim_head', head=self.heads)
        self.E = None
        self.F = None

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        b, n, c = x.shape

        # apply self.to_qkv and reshape its output
        qkv = self.reshape(self.to_qkv(x)).chunk(3, dim=-1)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # adjust the shape of E and F if necessary
        if self.E is None or self.F is None or self.E.shape[-1] < n or self.F.shape[-1] < n:
            self.E = nn.Parameter(torch.randn(self.heads, self.k, n)).cuda()
            self.F = nn.Parameter(torch.randn(self.heads, self.k, n)).cuda()

        # k = torch.einsum('b n d, h k d -> b h n k', k, self.E)
        # k = torch.einsum('b h n d, h k d -> b h n k', k, self.E)
        # v = torch.einsum('b h n d, h d k -> b h n k', v, self.F)

        # dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # apply linear transformations
        k = torch.einsum('b h n d, h k n -> b h k d', k, self.E)
        v = torch.einsum('b h n d, h k n -> b h k d', v, self.F)

        # compute attention scores
        dots = torch.einsum('b h n d, b h k d -> b h n k', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, float('-inf'))

        attn = dots.softmax(dim=-1)
        # out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = torch.einsum('b h n k, b h k d -> b h n d', attn, v)
        out = rearrange(out, 'b head n dim_head -> b n (head dim_head)')

        out = self.to_out(out)
        return out.view(b, n, -1)


class LinformerAttention_wind1(nn.Module):  ## earliest and good
    def __init__(self, dim=48, heads=3, dim_head=16, dropout=0., k=None, window_size=800):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.w = 30
        self.window_size = window_size

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_head = nn.Linear(dim, inner_dim, bias=False)

        # reshape the output of self.to_qkv
        self.reshape = Rearrange('b n (head dim_head) -> b head n dim_head', head=self.heads)
        self.E = nn.Parameter(torch.randn(heads, dim_head, self.w))
        self.F = nn.Parameter(torch.randn(heads, dim_head, self.w))
        # self.E = nn.Parameter(torch.randn(heads, dim_head, self.w))
        # self.F = nn.Parameter(torch.randn(heads, dim_head, self.w))
        self.lin1 = nn.Linear(self.w, dim_head)
        self.lin2 = nn.Linear(self.w, dim_head)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        b, n, c = x.shape

        # compute the number of windows
        num_windows = (n + self.window_size - 1) // self.window_size

        # prepare output tensor
        out = torch.empty(b, n, self.heads * self.dim_head, device=x.device)

        # iterate over windows
        for i in range(num_windows):
            # get start and end indices of the current window
            start_idx = i * self.window_size
            end_idx = min(start_idx + self.window_size, n)

            # extract the current window
            x_window = x[:, start_idx:end_idx, :]

            # apply self.to_qkv and reshape its output
            qkv = self.reshape(self.to_qkv(x_window)).chunk(3, dim=-1)
            q, k, v = qkv[0], qkv[1], qkv[2]
            ######################## change to h d d form  #################
            # apply linear transformations
            # e = self.E[:, i, :]

            # 创建两个线性层
            # linear1 = nn.Linear(self.w, self.dim_head).cuda()
            # linear2 = nn.Linear(self.dim_head, self.dim_head).cuda()
            # 将输入张量的最后一个维度从 n 变换为 d
            # e = self.li(self.E)
            # 将张量沿着最后两个维度进行矩阵乘法运算
            # e = linear2(temp.transpose(1, 2)).transpose(1, 2)

            # e = torch.matmul(self.E, self.E.transpose(-2, -1))
            # 计算张量的乘积
            # e = self.E
            # e = e.matmul(e.transpose(1, 2))
            # f = self.F
            # f = f.matmul(f.transpose(1, 2))
            e = self.lin1(self.E)
            f = self.lin2(self.F)
            # f = torch.matmul(self.F, self.F.transpose(-2, -1))
            k = torch.einsum('b h n d, h d d  -> b h n d', k, e)
            v = torch.einsum('b h n d, h d d  -> b h n d', v, f)

            # compute attention scores
            dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

            if mask is not None:
                # pad the mask to match the shape of the attention scores
                mask_window = mask[:, start_idx:end_idx]
                mask_window = F.pad(mask_window.flatten(1), (1, 0), value=True)
                assert mask_window.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
                mask_window = rearrange(mask_window, 'b i -> b () i ()') * rearrange(mask_window, 'b j -> b () () j')
                dots.masked_fill_(~mask_window, float('-inf'))

            # apply softmax and compute weighted sum
            attn = dots.softmax(dim=-1)
            out_window = torch.einsum('b h i j, b h j d -> b h i d', attn, v)

            # reshape and store the output of the current window
            out_window = rearrange(out_window, 'b head n dim_head -> b n (head dim_head)')
            out[:, start_idx:end_idx, :] = out_window

        # apply final linear transformation and return the output
        out = self.to_out(out)
        return out


class LinformerAttention_wind2(nn.Module):
    def __init__(self, dim, window_size, k=None, heads=4, dim_head=16, dropout=0.):
        super().__init__()
        self.inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.k = k if k is not None else dim_head

        self.window_size = window_size

        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.to_head = nn.Linear(dim, self.inner_dim, bias=False)

        # reshape the output of self.to_qkv
        self.reshape = Rearrange('b n (head dim_head) -> b head n dim_head', head=self.heads)
        self.E = None
        self.F = None

        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        b, n, c = x.shape

        # compute the number of windows
        # self.window_size = max(1, (self.k // self.heads) * 4)
        num_windows = (n + self.window_size - 1) // self.window_size

        # prepare output tensor
        # out = torch.empty(b, num_windows, self.window_size, self.inner_dim, device=x.device)
        out = torch.empty(b, n, self.inner_dim, device=x.device)

        # compute q, k vectors for all windows
        qkv = self.reshape(self.to_qkv(x)).chunk(3, dim=-1)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # k = torch.einsum('b h n d, d k -> b h n k', k, self.to_head.weight.unsqueeze(0).unsqueeze(0))
        # q = torch.einsum('b h n d, d k -> b h n k', q, self.to_head.weight)

        # adjust the shape of E and F if necessary
        if self.E is None or self.F is None or self.E.shape[-1] < n or self.F.shape[-1] < n:
            self.E = nn.Parameter(torch.randn(self.heads, self.k, n)).cuda()
            self.F = nn.Parameter(torch.randn(self.heads, self.k, n)).cuda()

        # iterate over windows
        for i in range(num_windows):
            # get start and end indices of the current window

            start_idx = i * self.window_size
            end_idx = min(start_idx + self.window_size, n)

            # extract the current window
            x_window = x[:, start_idx:end_idx, :]

            # extract k and v vectors for the current window
            q_window = q[:, :, start_idx:end_idx, :]
            k_window = k[:, :, start_idx:end_idx, :]
            v_window = v[:, :, start_idx:end_idx, :]

            # apply linear transformations
            k_window = torch.einsum('b h n k, h m n -> b h m k', k_window, self.E[:, :, start_idx:end_idx])
            v_window = torch.einsum('b h n k, h m n -> b h m k', v_window, self.F[:, :, start_idx:end_idx])

            # compute attention scores
            dots = torch.einsum('b h n k, b h m k -> b h n m', q_window, k_window) * self.scale

            if mask is not None:
                # pad the mask to match the shape of the attention scores
                mask_window = mask[:, start_idx:end_idx]
                mask_window = F.pad(mask_window.flatten(1), (1, 0), value=True)
                assert mask_window.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
                mask_window = rearrange(mask_window, 'b i -> b () i ()') * rearrange(mask_window, 'b j -> b () () j')
                dots.masked_fill_(~mask_window, float('-inf'))

            # apply softmax and compute weighted sum
            attn = dots.softmax(dim=-1)
            out_window = torch.einsum('b h n m, b h m k -> b h n k', attn, v_window)
            out_window = rearrange(out_window, 'b head n dim_head -> b n (head dim_head)')
            # store the output of the current window
            # out[:, i, :end_idx - start_idx, :] = out_window
            out[:, start_idx:end_idx, :] = out_window

        # flatten and concatenate the output tensor
        # out = rearrange(out, 'b w n d -> b (w n) d')

        # apply final linear transformation and return the output
        out = self.to_out(out)
        return out


class BlockAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, window_size, block_size):
        super().__init__()

        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)

        self.window_size = window_size
        self.block_size = block_size

    def forward(self, x, mask=0):
        batch_size, seq_len, input_dim = x.size()

        # 计算查询、键和值
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # 分块计算
        out_list = []
        for i in range(0, seq_len, self.block_size):
            start_idx = i
            end_idx = min(seq_len, i + self.block_size)

            # 计算注意力分数
            q_block = q[:, start_idx:end_idx, :]
            k_block = k[:, max(0, start_idx - self.window_size + 1):end_idx + self.window_size - 1, :]
            attn = torch.bmm(q_block, k_block.transpose(1, 2)) / (input_dim ** 0.5)

            # 计算加权和
            v_block = v[:, max(0, start_idx - self.window_size + 1):end_idx + self.window_size - 1, :]
            out = torch.bmm(attn, v_block)
            out_list.append(out)

        # 拼接所有块的加权和
        out = torch.cat(out_list, dim=1)

        return out


class Transformer_g(nn.Module):  ### heads maybe 3, dim_head may be 16
    def __init__(self, dim, depth=1, heads=2, dim_head=16, mlp_dim=48, sp_sz=64 * 64, num_channels=48, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        # self.pos_embedding = nn.Parameter(torch.randn(1, sp_sz, num_channels))
        # self.attention = nn.MultiheadAttention(dim, heads, dropout)
        # self.spectral_norm = nn.utils.spectral_norm(nn.Linear(dim, dim))
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                # Residual(PreNorm(dim, nn.MultiheadAttention(dim, num_heads=heads, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))]))

    def forward(self, x, mask=None):
        # pos = self.pos_embedding
        # x += pos
        # x = self.spectral_norm(x)
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class Transformer_l(nn.Module):  ### heads maybe 3, dim_head may be 16
    def __init__(self, dim=48, depth=1, heads=3, dim_head=24, mlp_dim=48, window_size=800, sp_sz=64 * 64,
                 num_channels=48, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        # self.pos_embedding = nn.Parameter(torch.randn(1, sp_sz, num_channels))
        # self.attention = nn.MultiheadAttention(dim, heads, dropout)
        # self.encoder_layers = nn.TransformerEncoderLayer(d_model=dim, nhead=heads)
        # self.encoder = nn.TransformerEncoder(self.encoder_layers, num_layers=depth)
        # self.spectral_norm = nn.utils.spectral_norm(nn.Linear(dim, dim))
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # Residual(PreNorm(dim, self.encoder(self.encoder_layers, num_layers=depth))),
                # Residual(PreNorm(dim, BlockAttention(dim, dim, 128, 1024))),
                # Residual(PreNorm(dim, Attention_wind3(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(
                    PreNorm(dim, LinformerAttention_wind1(dim, heads=heads, dim_head=dim_head, window_size=window_size,
                                                          dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))]))

    def forward(self, x, mask=None):
        # pos = self.pos_embedding
        # x += pos
        # x = self.spectral_norm(x)
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class Transformer_l2(nn.Module):
    def __init__(self, dim, depth=1, heads=3, dim_head=16, mlp_dim=48, sp_sz=64 * 64, num_channels=48, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.pos_embedding = nn.Parameter(torch.randn(1, sp_sz, num_channels))
        self.attention = nn.MultiheadAttention(dim, heads)
        self.spectral_norm = nn.utils.spectral_norm(nn.Linear(dim, dim))
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, self.attention)),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))]))

    def forward(self, x, mask=None):
        if x.dim() != 3:
            x = x.view(-1, x.size(-1), 1).transpose(0, 2)
        pos = self.pos_embedding
        x += pos
        x = self.spectral_norm(x)
        query, key, value = x, x, x
        for attn, ff in self.layers:
            x, _ = attn(x, x, x)
            x = ff(x)
        return x


class Transformer_D(nn.Module):
    def __init__(self, dim, depth=2, heads=3, dim_head=16, mlp_dim=48, sp_sz=64 * 64, num_channels=48, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.pos_embedding = nn.Parameter(torch.randn(1, sp_sz, num_channels))
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        # pos = self.pos_embedding
        # x += pos
        for attn1, attn2, ff in self.layers:
            x = attn1(x, mask=mask)
            x = attn2(x, mask=mask)
            x = ff(x)
        return x
