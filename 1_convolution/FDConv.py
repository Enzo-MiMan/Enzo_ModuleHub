"""Standalone adaptation of the official FDConv implementation.

Source: https://github.com/Linwei-Chen/FDConv
Revision: bf7a33d4f39ac7ea8172158f5628107fcaaba469
File: FDConv_detection/mmdet_custom/FDConv.py

保留官方 FDW、全局/局部 KSM、空间 FBM、初始化和计算顺序。
移除 MMCV 注册和未使用依赖；外层仅兼容 channels= 调用和可选残差。
默认不使用残差，也不添加输出投影，行为与官方卷积层一致。
reduction 是官方的通道比例（默认 0.0625），不是旧版本的除数。
官方全局分支使用 BatchNorm：训练时 batch size 必须大于 1。
旧版本的 state_dict 与此版本不兼容。
"""
# MIT License
# 
# Copyright (c) 2025 陈林蔚
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """

    def __init__(self, scale_value=1.0, bias_value=0.0, scale_learnable=True, bias_learnable=True, mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1), requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1), requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias

class KernelSpatialModulation_Global(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16, temp=1.0, kernel_temp=None, kernel_att_init='dyconv_as_extra', att_multi=2.0, ksm_only_kernel_att=False, att_grid=1, stride=1, spatial_freq_decompose=False, act_type='sigmoid'):
        super(KernelSpatialModulation_Global, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.act_type = act_type
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = temp
        self.kernel_temp = kernel_temp
        self.ksm_only_kernel_att = ksm_only_kernel_att
        self.kernel_att_init = kernel_att_init
        self.att_multi = att_multi
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.att_grid = att_grid
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = StarReLU()
        self.spatial_freq_decompose = spatial_freq_decompose
        if ksm_only_kernel_att:
            self.func_channel = self.skip
        else:
            if spatial_freq_decompose:
                self.channel_fc = nn.Conv2d(attention_channel, in_planes * 2 if self.kernel_size > 1 else in_planes, 1, bias=True)
            else:
                self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
            self.func_channel = self.get_channel_attention
        if in_planes == groups and in_planes == out_planes or self.ksm_only_kernel_att:
            self.func_filter = self.skip
        else:
            if spatial_freq_decompose:
                self.filter_fc = nn.Conv2d(attention_channel, out_planes * 2, 1, stride=stride, bias=True)
            else:
                self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, stride=stride, bias=True)
            self.func_filter = self.get_filter_attention
        if kernel_size == 1 or self.ksm_only_kernel_att:
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention
        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if hasattr(self, 'channel_spatial'):
            nn.init.normal_(self.channel_spatial.conv.weight, std=1e-06)
        if hasattr(self, 'filter_spatial'):
            nn.init.normal_(self.filter_spatial.conv.weight, std=1e-06)
        if hasattr(self, 'spatial_fc') and isinstance(self.spatial_fc, nn.Conv2d):
            nn.init.normal_(self.spatial_fc.weight, std=1e-06)
            if self.kernel_att_init == 'dyconv_as_extra':
                pass
            else:
                pass
        if hasattr(self, 'func_filter') and isinstance(self.func_filter, nn.Conv2d):
            nn.init.normal_(self.func_filter.weight, std=1e-06)
            if self.kernel_att_init == 'dyconv_as_extra':
                pass
            else:
                pass
        if hasattr(self, 'kernel_fc') and isinstance(self.kernel_fc, nn.Conv2d):
            nn.init.normal_(self.kernel_fc.weight, std=1e-06)
            if self.kernel_att_init == 'dyconv_as_extra':
                pass
            else:
                pass
        if hasattr(self, 'channel_fc') and isinstance(self.channel_fc, nn.Conv2d):
            nn.init.normal_(self.channel_fc.weight, std=1e-06)
            if self.kernel_att_init == 'dyconv_as_extra':
                pass
            else:
                pass

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        if self.act_type == 'sigmoid':
            channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), 1, 1, -1, x.size(-2), x.size(-1)) / self.temperature) * self.att_multi
        elif self.act_type == 'tanh':
            channel_attention = 1 + torch.tanh_(self.channel_fc(x).view(x.size(0), 1, 1, -1, x.size(-2), x.size(-1)) / self.temperature)
        else:
            raise NotImplementedError
        return channel_attention

    def get_filter_attention(self, x):
        if self.act_type == 'sigmoid':
            filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), 1, -1, 1, x.size(-2), x.size(-1)) / self.temperature) * self.att_multi
        elif self.act_type == 'tanh':
            filter_attention = 1 + torch.tanh_(self.filter_fc(x).view(x.size(0), 1, -1, 1, x.size(-2), x.size(-1)) / self.temperature)
        else:
            raise NotImplementedError
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        if self.act_type == 'sigmoid':
            spatial_attention = torch.sigmoid(spatial_attention / self.temperature) * self.att_multi
        elif self.act_type == 'tanh':
            spatial_attention = 1 + torch.tanh_(spatial_attention / self.temperature)
        else:
            raise NotImplementedError
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        if self.act_type == 'softmax':
            kernel_attention = F.softmax(kernel_attention / self.kernel_temp, dim=1)
        elif self.act_type == 'sigmoid':
            kernel_attention = torch.sigmoid(kernel_attention / self.kernel_temp) * 2 / kernel_attention.size(1)
        elif self.act_type == 'tanh':
            kernel_attention = (1 + torch.tanh(kernel_attention / self.kernel_temp)) / kernel_attention.size(1)
        else:
            raise NotImplementedError
        return kernel_attention

    def forward(self, x, use_checkpoint=False):
        if use_checkpoint:
            return checkpoint(self._forward, x)
        else:
            return self._forward(x)

    def _forward(self, x):
        # 路径一 / 全局 KSM：输入已由主流程池化；先用 1×1 卷积、BN、StarReLU 提取摘要信息。
        avg_x = self.relu(self.bn(self.fc(x)))
        # 分成四个注意力头；频率组系数用于组装频谱，其余系数用于调节卷积权重。
        return self.func_channel(avg_x), self.func_filter(avg_x), self.func_spatial(avg_x), self.func_kernel(avg_x)

class KernelSpatialModulation_Local(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel=None, kernel_num=1, out_n=1, k_size=3, use_global=False):
        super(KernelSpatialModulation_Local, self).__init__()
        self.kn = kernel_num
        self.out_n = out_n
        self.channel = channel
        if channel is not None:
            k_size = round(math.log2(channel) / 2 + 0.5) // 2 * 2 + 1
        self.conv = nn.Conv1d(1, kernel_num * out_n, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        nn.init.constant_(self.conv.weight, 1e-06)
        self.use_global = use_global
        if self.use_global:
            self.complex_weight = nn.Parameter(torch.randn(1, self.channel // 2 + 1, 2, dtype=torch.float32) * 1e-06)
        self.norm = nn.LayerNorm(self.channel)

    def forward(self, x, x_std=None):
        # 路径一 / 局部 KSM：通道摘要 (B,C,1,1) → (B,1,C)，将输入通道排成一维序列。
        x = x.squeeze(-1).transpose(-1, -2)
        (b, _, c) = x.shape
        # 可选通道序列频域处理，默认关闭；这里的 FFT 沿通道轴，不是图像高宽。
        if self.use_global:
            x_rfft = torch.fft.rfft(x.float(), dim=-1)
            x_real = x_rfft.real * self.complex_weight[..., 0][None]
            x_imag = x_rfft.imag * self.complex_weight[..., 1][None]
            x = x + torch.fft.irfft(torch.view_as_complex(torch.stack([x_real, x_imag], dim=-1)), dim=-1)
        # 沿通道序列做 LayerNorm，稳定数值尺度。
        x = self.norm(x)
        # Conv1d 读取相邻通道信息，为各输出通道和核位置预测细粒度分数。
        att_logit = self.conv(x)
        # 整理分数轴：(B,kn*out_n,C) → (B,kn,out_n,C) → (B,kn,C,out_n)。
        att_logit = att_logit.reshape(x.size(0), self.kn, self.out_n, c)
        att_logit = att_logit.permute(0, 1, 3, 2)
        # 返回原始分数；主流程随后重排并激活，得到与每个卷积权重对应的调节系数。
        return att_logit

class FrequencyBandModulation(nn.Module):

    def __init__(self, in_channels, k_list=[2], lowfreq_att=False, fs_feat='feat', act='sigmoid', spatial='conv', spatial_group=1, spatial_kernel=3, init='zero', max_size=(64, 64), **kwargs):
        super().__init__()
        self.k_list = k_list
        self.lowfreq_att = lowfreq_att
        self.in_channels = in_channels
        self.fs_feat = fs_feat
        self.act = act
        if spatial_group > 64:
            spatial_group = in_channels
        self.spatial_group = spatial_group
        if spatial == 'conv':
            self.freq_weight_conv_list = nn.ModuleList()
            _n = len(k_list)
            if lowfreq_att:
                _n += 1
            for i in range(_n):
                freq_weight_conv = nn.Conv2d(in_channels=in_channels, out_channels=self.spatial_group, stride=1, kernel_size=spatial_kernel, groups=self.spatial_group, padding=spatial_kernel // 2, bias=True)
                if init == 'zero':
                    nn.init.normal_(freq_weight_conv.weight, std=1e-06)
                    if freq_weight_conv.bias is not None:
                        freq_weight_conv.bias.data.zero_()
                self.freq_weight_conv_list.append(freq_weight_conv)
        else:
            raise NotImplementedError
        self.register_buffer('cached_masks', self._precompute_masks(max_size, k_list), persistent=False)

    def _precompute_masks(self, max_size, k_list):
        """
        在初始化时预先计算一组最大尺寸的掩码。
        """
        (max_h, max_w) = max_size
        (_, freq_indices) = get_fft2freq(d1=max_h, d2=max_w, use_rfft=True)
        freq_indices = freq_indices.abs().max(dim=-1, keepdims=False)[0]
        masks = []
        for freq in k_list:
            mask = freq_indices < 0.5 / freq + 1e-08
            masks.append(mask)
        return torch.stack(masks, dim=0).unsqueeze(1)

    def sp_act(self, freq_weight):
        if self.act == 'sigmoid':
            return freq_weight.sigmoid() * 2
        elif self.act == 'tanh':
            return 1 + freq_weight.tanh()
        elif self.act == 'softmax':
            return freq_weight.softmax(dim=1) * freq_weight.shape[1]
        else:
            raise NotImplementedError

    def forward(self, x, att_feat=None):
        # FBM 步骤 1：保留完整输入用于预测空间调节图，不做全局池化
        if att_feat is None:
            att_feat = x
        x_list = []
        x = x.to(torch.float32)
        pre_x = x.clone()
        (b, _, h, w) = x.shape

        # FBM 步骤 2：对输入特征的高宽做实数 FFT，得到输入频谱
        x_fft = torch.fft.rfft2(x, norm='ortho')
        (freq_h, freq_w) = (h, w // 2 + 1)
        # 将缓存低通掩码调整到当前频谱尺寸，供逐级低通使用。
        current_masks = F.interpolate(self.cached_masks.float(), size=(freq_h, freq_w), mode='nearest')

        # FBM 步骤 3：默认 k_list=[2,4,8]，逐级提取更低频的成分。
        for (idx, freq) in enumerate(self.k_list):
            mask = current_masks[idx]
            # 频谱乘低通掩码，再逆 FFT，得到这一层低通特征。
            low_part = torch.fft.irfft2(x_fft * mask, s=(h, w), norm='ortho')
            # 上一层剩余部分减去本层低通结果，得到一个频带；三次循环得到三个频带。
            high_part = pre_x - low_part
            pre_x = low_part

            # FBM 步骤 4：从完整输入预测当前频带的逐位置调节图。
            freq_weight = self.freq_weight_conv_list[idx](att_feat)
            freq_weight = self.sp_act(freq_weight)
            # 各频带逐位置乘调节系数；默认空间分组为 1，图在通道之间共享。
            tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * high_part.reshape(b, self.spatial_group, -1, h, w)
            x_list.append(tmp.reshape(b, -1, h, w))

        # FBM 步骤 5：最低频可选调节，默认 lowfreq_att=False，直接保留。
        if self.lowfreq_att:
            freq_weight = self.freq_weight_conv_list[len(self.k_list)](att_feat)
            freq_weight = self.sp_act(freq_weight)
            # 各频带逐位置乘调节系数；默认空间分组为 1，图在通道之间共享。
            tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * pre_x.reshape(b, self.spatial_group, -1, h, w)
            x_list.append(tmp.reshape(b, -1, h, w))
        else:
            x_list.append(pre_x)

        # FBM 步骤 6：把调节后的频带和最低频部分相加，得到 X′，交给最终卷积
        return sum(x_list)

def get_fft2freq(d1, d2, use_rfft=False):
    freq_h = torch.fft.fftfreq(d1)
    if use_rfft:
        freq_w = torch.fft.rfftfreq(d2)
    else:
        freq_w = torch.fft.fftfreq(d2)
    freq_hw = torch.stack(torch.meshgrid(freq_h, freq_w, indexing='ij'), dim=-1)
    dist = torch.norm(freq_hw, dim=-1)
    (sorted_dist, indices) = torch.sort(dist.view(-1))
    if use_rfft:
        d2 = d2 // 2 + 1
    sorted_coords = torch.stack([indices // d2, indices % d2], dim=-1)
    return (sorted_coords.permute(1, 0), freq_hw)

class _FDConvCore(nn.Conv2d):

    def __init__(self, *args, reduction=0.0625, kernel_num=4, use_fdconv_if_c_gt=16, use_fdconv_if_k_in=[1, 3], use_fdconv_if_stride_in=[1], use_fbm_if_k_in=[3], use_fbm_for_stride=False, kernel_temp=1.0, temp=None, att_multi=2.0, param_ratio=1, param_reduction=1.0, ksm_only_kernel_att=False, att_grid=1, use_ksm_local=True, ksm_local_act='sigmoid', ksm_global_act='sigmoid', spatial_freq_decompose=False, convert_param=True, linear_mode=False, fbm_cfg={'k_list': [2, 4, 8], 'lowfreq_att': False, 'fs_feat': 'feat', 'act': 'sigmoid', 'spatial': 'conv', 'spatial_group': 1, 'spatial_kernel': 3, 'init': 'zero', 'global_selection': False}, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_fdconv_if_c_gt = use_fdconv_if_c_gt
        self.use_fdconv_if_k_in = use_fdconv_if_k_in
        self.use_fdconv_if_stride_in = use_fdconv_if_stride_in
        self.kernel_num = kernel_num
        self.param_ratio = param_ratio
        self.param_reduction = param_reduction
        self.use_ksm_local = use_ksm_local
        self.att_multi = att_multi
        self.spatial_freq_decompose = spatial_freq_decompose
        self.use_fbm_if_k_in = use_fbm_if_k_in
        self.ksm_local_act = ksm_local_act
        self.ksm_global_act = ksm_global_act
        assert self.ksm_local_act in ['sigmoid', 'tanh']
        assert self.ksm_global_act in ['softmax', 'sigmoid', 'tanh']
        if self.kernel_num is None:
            self.kernel_num = self.out_channels // 2
            kernel_temp = math.sqrt(self.kernel_num * self.param_ratio)
        if temp is None:
            temp = kernel_temp
        if min(self.in_channels, self.out_channels) <= self.use_fdconv_if_c_gt or self.kernel_size[0] not in self.use_fdconv_if_k_in:
            return
        self.alpha = min(self.out_channels, self.in_channels) // 2 * self.kernel_num * self.param_ratio / param_reduction
        self.KSM_Global = KernelSpatialModulation_Global(self.in_channels, self.out_channels, self.kernel_size[0], groups=self.groups, temp=temp, kernel_temp=kernel_temp, reduction=reduction, kernel_num=self.kernel_num * self.param_ratio, kernel_att_init=None, att_multi=att_multi, ksm_only_kernel_att=ksm_only_kernel_att, act_type=self.ksm_global_act, att_grid=att_grid, stride=self.stride, spatial_freq_decompose=spatial_freq_decompose)
        if self.kernel_size[0] in use_fbm_if_k_in or (use_fbm_for_stride and self.stride[0] > 1):
            self.FBM = FrequencyBandModulation(self.in_channels, **fbm_cfg)
        if self.use_ksm_local:
            self.KSM_Local = KernelSpatialModulation_Local(channel=self.in_channels, kernel_num=1, out_n=int(self.out_channels * self.kernel_size[0] * self.kernel_size[1]))
        self.linear_mode = linear_mode
        self.convert2dftweight(convert_param)

    def convert2dftweight(self, convert_param):
        (d1, d2, k1, k2) = (self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        # 路径一 / 初始化准备：按真实频率半径排序索引，供后续等量分组；不是在每个小核上单独分组。
        (freq_indices, _) = get_fft2freq(d1 * k1, d2 * k2, use_rfft=True)
        # 将 (C_out,C_in,K_h,K_w) 重排成 (C_out*K_h,C_in*K_w) 大平面。
        weight = self.weight.permute(0, 2, 1, 3).reshape(d1 * k1, d2 * k2)
        # 把普通卷积权重转为复数频谱，为可学习频域参数提供初始化。
        weight_rfft = torch.fft.rfft2(weight, dim=(0, 1))
        if self.param_reduction < 1:
            num_to_keep = int(freq_indices.size(1) * self.param_reduction)
            freq_indices = freq_indices[:, :num_to_keep]
            weight_rfft = torch.stack([weight_rfft.real, weight_rfft.imag], dim=-1)
            weight_rfft = weight_rfft[freq_indices[0, :], freq_indices[1, :]]
            weight_rfft = weight_rfft.reshape(-1, 2)[None,].repeat(self.param_ratio, 1, 1) / (min(self.out_channels, self.in_channels) // 2)
        else:
            weight_rfft = torch.stack([weight_rfft.real, weight_rfft.imag], dim=-1)[None,].repeat(self.param_ratio, 1, 1, 1) / (min(self.out_channels, self.in_channels) // 2)
        if convert_param:
            # 保存为可训练参数；forward 根据输入调节这些参数的各组贡献。
            self.dft_weight = nn.Parameter(weight_rfft, requires_grad=True)
            del self.weight
        elif self.linear_mode:
            assert self.kernel_size[0] == 1 and self.kernel_size[1] == 1
            self.weight = torch.nn.Parameter(self.weight.squeeze(), requires_grad=True)
        indices = []
        for i in range(self.param_ratio):
            # 按 kernel_num 把排序后的频率索引等量分组；默认示例为 4 组。
            indices.append(freq_indices.reshape(2, self.kernel_num, -1))
        self.register_buffer('indices', torch.stack(indices, dim=0), persistent=False)

    def get_FDW(self):
        (d1, d2, k1, k2) = (self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        weight = self.weight.reshape(d1, d2, k1, k2).permute(0, 2, 1, 3).reshape(d1 * k1, d2 * k2)
        weight_rfft = torch.fft.rfft2(weight, dim=(0, 1)).contiguous()
        weight_rfft = torch.stack([weight_rfft.real, weight_rfft.imag], dim=-1)[None,].repeat(self.param_ratio, 1, 1, 1) / (min(self.out_channels, self.in_channels) // 2)
        return weight_rfft

    def forward(self, x):
        # 步骤 0：判断是否启用 FDConv；条件不满足时直接返回普通卷积结果。
        if min(self.in_channels, self.out_channels) <= self.use_fdconv_if_c_gt or self.kernel_size[0] not in self.use_fdconv_if_k_in:
            return super().forward(x)

        # ======================== 路径一 ： 准备卷积核 ========================
        # 1：读取输入摘要。全局平均池化：(B,C,H,W) → (B,C,1,1)
        global_x = F.adaptive_avg_pool2d(x, 1)

        # 2：全局 KSM ：输出四类系数：输入通道、输出通道、核位置、频率组
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.KSM_Global(global_x)

        # 3：局部 KSM ：生成细粒度分数，每个输入通道、输出通道和核位置都有对应值
        if self.use_ksm_local:
            hr_att_logit = self.KSM_Local(global_x) # b, kn, cin, cout * ratio, k1*k2,
            # 把输出通道×核位置拆开；再交换输入/输出通道轴，使形状与卷积权重对应。
            hr_att_logit = hr_att_logit.reshape(x.size(0), 1, self.in_channels, self.out_channels, self.kernel_size[0], self.kernel_size[1])
            # hr_att_logit = hr_att_logit + self.hr_cin_bias[None, None, :, None, None, None] + self.hr_cout_bias[None, None, None, :, None, None] + self.hr_spatial_bias[None, None, None, None, :, :]
            hr_att_logit = hr_att_logit.permute(0, 1, 3, 2, 4, 5)
            # 将原始分数转换成调节系数；默认 sigmoid×2，既能减弱也能增强。
            if self.ksm_local_act == 'sigmoid':
                hr_att = hr_att_logit.sigmoid() * self.att_multi
            elif self.ksm_local_act == 'tanh':
                hr_att = 1 + hr_att_logit.tanh()
            else:
                raise NotImplementedError
        else:   # 局部分支关闭时，乘以 1，不改变权重。
            hr_att = 1

        b = x.size(0)
        batch_size, in_planes, height, width = x.size()

        # 4：为每个样本准备自己的频谱容器；最后一维 2 存放实部与虚部。
        DFT_map = torch.zeros((b, self.out_channels * self.kernel_size[0], self.in_channels * self.kernel_size[1] // 2 + 1, 2), device=x.device)
        kernel_attention = kernel_attention.reshape(b, self.param_ratio, self.kernel_num, -1)

        # 5：取出可学习频域参数；未转换参数时，从空间权重计算频谱
        if hasattr(self, 'dft_weight'):
            dft_weight = self.dft_weight
        else:
            dft_weight = self.get_FDW()

        # 6：按组加权，组装频谱 ： 按频率组索引取参数，乘当前样本的频率组系数，再填入该样本的频谱。
        for i in range(self.param_ratio):
            indices = self.indices[i]
            if self.param_reduction < 1:
                w = dft_weight[i].reshape(self.kernel_num, -1, 2)[None]
                DFT_map[:, indices[0, :, :], indices[1, :, :]] += torch.stack([w[..., 0] * kernel_attention[:, i], w[..., 1] * kernel_attention[:, i]], dim=-1)
            else:
                w = dft_weight[i][indices[0, :, :], indices[1, :, :]][None] * self.alpha
                DFT_map[:, indices[0, :, :], indices[1, :, :]] += torch.stack([w[..., 0] * kernel_attention[:, i], w[..., 1] * kernel_attention[:, i]], dim=-1)
                pass

        # 7：逆 FFT 得到空间权重平面，再重排成各样本的卷积核。此时尚未乘全部 KSM 系数。
        adaptive_weights = torch.fft.irfft2(torch.view_as_complex(DFT_map), s=(self.out_channels * self.kernel_size[0], self.in_channels * self.kernel_size[1]), dim=(1, 2)).reshape(batch_size, 1, self.out_channels, self.kernel_size[0], self.in_channels, self.kernel_size[1])
        adaptive_weights = adaptive_weights.permute(0, 1, 2, 4, 3, 5)

        # ======================== 路径二 ： 处理输入特征 ========================
        # FBM 拆输入频带、按空间位置调节并合并，得到调整后的输入 X′
        if hasattr(self, 'FBM'):
            x = self.FBM(x)

        # ======================== 汇合，完成卷积计算 ========================
        # 以下两种计算方式二选一，是把通道注意力乘到权重或输入/输出的等价实现。
        if self.out_channels * self.in_channels * self.kernel_size[0] * self.kernel_size[1] < (in_planes + self.out_channels) * height * width:
            # 路径一完成：结合全局与局部 KSM，得到最终动态卷积核 W。
            aggregate_weight = spatial_attention * channel_attention * filter_attention * adaptive_weights * hr_att
            aggregate_weight = torch.sum(aggregate_weight, dim=1)
            aggregate_weight = aggregate_weight.view([-1, self.in_channels // self.groups, self.kernel_size[0], self.kernel_size[1]])
            # 汇合 / 步骤 10：把 batch 暂时并入通道维，方便一次完成每个样本各用自己的核的分组卷积。
            x = x.reshape(1, -1, height, width)
            # 使用路径一的核 W，对路径二处理后的输入 X′ 做卷积；groups 隔开各个样本。
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups * batch_size)
            if isinstance(filter_attention, float):
                # 步骤 11：恢复 (B,C_out,H_out,W_out)；另一计算分支同时施加输出通道注意力。
                output = output.view(batch_size, self.out_channels, output.size(-2), output.size(-1))
            else:
                # 步骤 11：恢复 (B,C_out,H_out,W_out)；另一计算分支同时施加输出通道注意力。
                output = output.view(batch_size, self.out_channels, output.size(-2), output.size(-1))
        else:
            # 另一种计算方式：先合并核位置与局部 KSM；输入/输出通道系数分别作用于特征。
            aggregate_weight = spatial_attention * adaptive_weights * hr_att
            aggregate_weight = torch.sum(aggregate_weight, dim=1)
            if not isinstance(channel_attention, float):
                # 输入通道注意力作用于调整后的输入，等价于将其乘在卷积核对应的输入通道上。
                x = x * channel_attention.view(b, -1, 1, 1)
            aggregate_weight = aggregate_weight.view([-1, self.in_channels // self.groups, self.kernel_size[0], self.kernel_size[1]])
            # 汇合 / 步骤 10：把 batch 暂时并入通道维，方便一次完成每个样本各用自己的核的分组卷积。
            x = x.reshape(1, -1, height, width)
            # 使用路径一的核 W，对路径二处理后的输入 X′ 做卷积；groups 隔开各个样本。
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups * batch_size)
            if isinstance(filter_attention, float):
                # 步骤 11：恢复 (B,C_out,H_out,W_out)；另一计算分支同时施加输出通道注意力。
                output = output.view(batch_size, self.out_channels, output.size(-2), output.size(-1))
            else:
                # 步骤 11：恢复 (B,C_out,H_out,W_out)；另一计算分支同时施加输出通道注意力。
                output = output.view(batch_size, self.out_channels, output.size(-2), output.size(-1)) * filter_attention.view(b, -1, 1, 1)
        # 步骤 12：加卷积偏置，返回输出特征 Y。默认流程没有额外输出投影或残差。
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        return output

class FDConv(_FDConvCore):
    """官方核心的独立接口。支持 channels=C 或 in_channels/out_channels。

    不指定 padding 时，奇数核默认保持尺寸（stride=1）。
    use_residual=True 仅为可选外层封装，不属于官方 FDConv。
    """

    def __init__(self, in_channels=None, out_channels=None, kernel_size=3,
                 stride=1, padding=None, dilation=1, groups=1, bias=True,
                 *, channels=None, use_residual=False, reduction=0.0625,
                 kernel_num=4, **kwargs):
        if channels is not None:
            if in_channels is not None or out_channels is not None:
                raise ValueError("channels cannot be combined with in_channels/out_channels")
            in_channels = out_channels = channels
        if in_channels is None:
            raise ValueError("Specify channels or in_channels")
        if out_channels is None:
            out_channels = in_channels
        if not 0 < reduction <= 1:
            raise ValueError("reduction is an official channel ratio in (0, 1], e.g. 0.0625")
        k = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        if len(k) != 2 or min(k) <= 0:
            raise ValueError("kernel_size must contain two positive values")
        if padding is None:
            dil = (dilation, dilation) if isinstance(dilation, int) else dilation
            padding = tuple(d * (size - 1) // 2 for size, d in zip(k, dil))
        active = min(in_channels, out_channels) > kwargs.get('use_fdconv_if_c_gt', 16) and k[0] in kwargs.get('use_fdconv_if_k_in', [1, 3])
        if active:
            # 官方动态路径未正确支持这些组合；提前报错，避免隐蔽 reshape 错误。
            if groups != 1 or k[0] != k[1]:
                raise ValueError("The official dynamic path requires groups=1 and square kernels")
            if kwargs.get('padding_mode', 'zeros') != 'zeros':
                raise ValueError("The dynamic path supports zero padding only")
            if kwargs.get('spatial_freq_decompose', False):
                raise ValueError("spatial_freq_decompose is unsupported by the upstream forward")
            n = out_channels // 2 if kernel_num is None else kernel_num
            ratio = kwargs.get('param_ratio', 1)
            pr = kwargs.get('param_reduction', 1.0)
            if not isinstance(n, int) or n < 1 or not isinstance(ratio, int) or ratio < 1 or not 0 < pr <= 1:
                raise ValueError("Invalid kernel_num, param_ratio or param_reduction")
            bins = int(out_channels * k[0] * (in_channels * k[1] // 2 + 1) * pr)
            if bins == 0 or bins % n:
                raise ValueError("Official FDW requires the retained FFT coefficient count to be divisible by kernel_num")
        st = (stride, stride) if isinstance(stride, int) else tuple(stride)
        if use_residual and (in_channels != out_channels or st != (1, 1)):
            raise ValueError("Residual requires equal channels and stride=1")
        super().__init__(in_channels, out_channels, kernel_size, stride=stride,
                         padding=padding, dilation=dilation, groups=groups,
                         bias=bias, reduction=reduction, kernel_num=kernel_num,
                         **kwargs)
        self.use_residual = use_residual

    def forward(self, x):
        # 进入上面的核心流程：路径一配核 + 路径二调输入 → 动态卷积 → 输出 Y。
        out = super().forward(x)
        # 可选外层残差，默认关闭；这一步不属于示意图中的官方核心计算。
        if self.use_residual:
            if out.shape != x.shape:
                raise ValueError("Residual requires matching input/output shapes")
            out = out + x
        return out


if __name__ == '__main__':
    x = torch.randn(2, 32, 64, 64)
    module = FDConv(channels=32, kernel_size=3, kernel_num=4)
    y = module(x)

    print('input shape :', x.shape)
    print('output shape:', y.shape)
