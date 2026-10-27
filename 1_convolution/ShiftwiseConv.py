# MIT License
# Copyright (c) Meta Platforms, Inc. and affiliates.
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

"""ShiftwiseConv V2 basic operator, with a portable PyTorch shift-add backend.

Reference: lidc54/shift-wiseConv, commit 2202e6b5c94df2606fecd8202831bb58966d14bc
  backbones/SW_v2_unirep.py: ShifthWiseConv2dImplicit
  shiftadd/ops/ops_py/add_shift.py: AddShift_mp_module
Original repository license: MIT; Copyright (c) Meta Platforms, Inc. and affiliates.

Preserves the official channel selection, multi-path convolutions, multi-rep
mappings, three branch sums, BN, residual, and rep/ghost output order. CUDA
atomic-add is replaced with differentiable PyTorch gathers and crops. Floating
point accumulation order can differ. RNGs are local to avoid resetting the
application's random state. Mapping tensors are registered as persistent buffers.
"""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class AddShift_mp_module(nn.Module):
    """PyTorch equivalent of the official unweighted multi-rep CUDA operator."""

    def __init__(self, big_kernel, small_kernel, c_out, c_in, group_in):
        super().__init__()
        self.c_in, self.c_out, self.group_in = c_in, c_out, group_in
        self.nk = math.ceil(big_kernel / small_kernel)
        self.kernels = (small_kernel, big_kernel)
        padding = small_kernel - 1
        self.extra_pad = padding - small_kernel // 2
        self.real_pad = [big_kernel // 2 - i * small_kernel - padding
                         for i in range(self.nk)]
        if c_in != c_out * self.nk or group_in < 1:
            raise ValueError("Expected c_in=c_out*ceil(big_kernel/small_kernel), N_rep>=1")

        # Same draw order and CPU seed as the official shuffle_idx_2_gen_pads.
        generator = torch.Generator(device="cpu").manual_seed(123)
        horizon = [torch.cat([torch.randperm(self.nk, generator=generator)
                              + i * self.nk for i in range(c_out)]).int()
                   for _ in range(group_in)]
        vertical = [torch.cat([torch.randperm(self.nk, generator=generator)
                               + i * self.nk for i in range(c_out)]).int()
                    for _ in range(group_in)]
        identity = [(torch.randint(0, self.nk, (c_out,), generator=generator)
                     + torch.arange(c_out) * self.nk).int()
                    for _ in range(group_in)]
        offsets = torch.tensor(self.real_pad * c_out, dtype=torch.int32)
        pad_h = torch.stack([offsets[index.argsort()] for index in horizon])
        pad_v = torch.stack([offsets[index.argsort()] for index in vertical])
        self.register_buffer("pad_hv", torch.cat([pad_h, pad_v]).T.contiguous())
        self.register_buffer("idx_identit", torch.stack(identity).T.contiguous())
        self.register_buffer("idx_out", torch.arange(c_out, dtype=torch.int32)
                             .repeat_interleave(self.nk))

    @staticmethod
    def _translate_crop(x, dy, dx, height, width):
        """Scatter input (y,x) to output (y+dy,x+dx), clipping to output size."""
        x = F.pad(x, (max(dx, 0), max(-dx, 0), max(dy, 0), max(-dy, 0)))
        top, left = max(-dy, 0), max(-dx, 0)
        return x[:, :, top:top + height, left:left + width]

    def forward(self, x, b, hout, wout):
        expected = (b, self.c_in, hout + 2 * self.extra_pad,
                    wout + 2 * self.extra_pad)
        if tuple(x.shape) != expected:
            raise ValueError(f"Expected expanded shape {expected}, got {tuple(x.shape)}")
        # Recover the per-slot input channel from the official padding tables.
        # Each group of nk expanded channels belongs to ONE original channel.
        tables = self.pad_hv.reshape(self.c_out, self.nk, 2 * self.group_in)
        order = tables.argsort(dim=1, descending=True)
        base = torch.arange(self.c_out, device=x.device) * self.nk
        horizontal = x.new_zeros((b, self.c_out, hout, wout))
        vertical = torch.zeros_like(horizontal)
        small = torch.zeros_like(horizontal)
        for rep in range(self.group_in):
            for slot, offset in enumerate(self.real_pad):
                h = x.index_select(1, base + order[:, slot, rep])
                v = x.index_select(1, base + order[:, slot, rep + self.group_in])
                horizontal = horizontal + self._translate_crop(
                    h, -self.extra_pad, offset, hout, wout)
                vertical = vertical + self._translate_crop(
                    v, offset, -self.extra_pad, hout, wout)
            local = x.index_select(1, self.idx_identit[:, rep].long())
            small = small + self._translate_crop(
                local, -self.extra_pad, -self.extra_pad, hout, wout)
        return horizontal, vertical, small


class ShiftwiseConv(nn.Module):
    """Official V2 operator structure; stride=1 and equal input/output channels.

    As in the official operator, output channels are ordered [rep, ghost],
    rather than restored to their original input indices. No GELU or 1x1 fuse
    belongs inside this basic operator.
    """

    def __init__(self, channels, big_kernel=51, small_kernel=3,
                 ghost_ratio=0.23, n_paths=2, n_rep=4, bn=True):
        super().__init__()
        if (not 0 <= ghost_ratio < 1 or n_paths < 1 or n_rep < 1
                or small_kernel < 1 or small_kernel % 2 != 1
                or big_kernel < small_kernel or big_kernel % 2 != 1):
            raise ValueError("Invalid kernel sizes, ghost_ratio, n_paths, or n_rep")
        self.kernels = (small_kernel, big_kernel)
        self.stride = 1
        self.nk = math.ceil(big_kernel / small_kernel)
        padding = small_kernel - 1
        self.pad = (padding, [big_kernel // 2 - i * small_kernel - padding
                              for i in range(self.nk)])
        repN = int(channels * (1 - ghost_ratio)) // 2 * 2
        if repN < 2:
            raise ValueError("V2 requires at least two rep channels; increase channels or lower ghost_ratio")
        ghostN = channels - repN
        ghost = sorted(np.random.RandomState(123).choice(
            channels, ghostN, replace=False).tolist())
        rep = sorted(set(range(channels)) - set(ghost))
        self.register_buffer("ghost", torch.tensor(ghost, dtype=torch.int32))
        self.register_buffer("rep", torch.tensor(rep, dtype=torch.int32))
        self.channels = channels
        self.rep_channels, self.ghost_channels = repN, ghostN
        self.big_kernel, self.small_kernel = big_kernel, small_kernel
        self.n_paths, self.n_rep = n_paths, n_rep
        out_n = repN * self.nk
        self.LoRA = None
        self.LoRAs = nn.ModuleList([
            nn.Conv2d(repN, out_n, small_kernel, stride=1,
                      padding=padding, groups=repN, bias=False)
            for _ in range(n_paths)])
        self.loras = AddShift_mp_module(big_kernel, small_kernel, repN, out_n, n_rep)
        self.use_bn = bn
        self.bn_loras = None
        # Official backbones use use_sync_bn=True.
        self.bn_lora1 = nn.SyncBatchNorm(repN) if bn else None
        self.bn_lora2 = nn.SyncBatchNorm(repN) if bn else None
        self.bn_small = nn.SyncBatchNorm(repN) if bn else None

    def forward(self, inputs):
        if inputs.ndim != 4 or inputs.shape[1] != self.channels:
            raise ValueError(f"Expected (B, {self.channels}, H, W), got {tuple(inputs.shape)}")
        b, _, h, w = inputs.shape
        ghost_inputs = inputs.index_select(1, self.ghost)
        rep_inputs = inputs.index_select(1, self.rep)
        if self.LoRA is None:
            out = self.LoRAs[0](rep_inputs)
            for conv in self.LoRAs[1:]:
                out = out + conv(rep_inputs)
        else:
            out = self.LoRA(rep_inputs)
        horizontal, vertical, small = self.loras(out, b, h, w)
        if self.use_bn:
            horizontal = self.bn_lora1(horizontal)
            vertical = self.bn_lora2(vertical)
            small = self.bn_small(small)
        rep_out = horizontal + vertical + small + rep_inputs
        return torch.cat([rep_out, ghost_inputs], dim=1)

    @torch.no_grad()
    def merge_branches(self):
        """Merge the parallel small convolutions exactly as in the official code."""
        if self.LoRA is not None:
            return
        first = self.LoRAs[0]
        merged = nn.Conv2d(first.in_channels, first.out_channels, first.kernel_size,
                           stride=first.stride, padding=first.padding,
                           dilation=first.dilation, groups=first.groups, bias=False)
        merged = merged.to(device=first.weight.device, dtype=first.weight.dtype)
        merged.weight.copy_(sum(conv.weight for conv in self.LoRAs))
        merged.train(self.training)
        self.LoRA = merged
        del self.LoRAs


if __name__ == "__main__":
    x = torch.randn(2, 64, 32, 32)
    module = ShiftwiseConv(channels=64)
    y = module(x)
    print("input shape :", x.shape)
    print("output shape:", y.shape)
