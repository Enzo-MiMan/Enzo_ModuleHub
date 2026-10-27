import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize_grid(y_map, x_map, height, width):
    y_map = y_map.clamp(0, height - 1)
    x_map = x_map.clamp(0, width - 1)
    y_norm = 2.0 * y_map / max(height - 1, 1) - 1.0
    x_norm = 2.0 * x_map / max(width - 1, 1) - 1.0
    return torch.stack([x_norm, y_norm], dim=-1)


def _snake_coordinates(offset, morph, extend_scope):
    b, _, h, w = offset.shape
    k = offset.shape[1] // 2
    center = k // 2
    device = offset.device
    dtype = offset.dtype

    y_offset, x_offset = torch.split(offset, k, dim=1)
    y_center = torch.arange(h, device=device, dtype=dtype).view(1, 1, h, 1).expand(b, k, h, w)
    x_center = torch.arange(w, device=device, dtype=dtype).view(1, 1, 1, w).expand(b, k, h, w)

    if morph == 0:
        x_spread = torch.linspace(-center, center, k, device=device, dtype=dtype).view(1, k, 1, 1)
        y_base = y_center
        x_base = x_center + x_spread

        y_delta = y_offset.clone()
        y_delta[:, center] = 0
        for idx in range(1, center + 1):
            y_delta[:, center + idx] = y_delta[:, center + idx - 1] + y_offset[:, center + idx]
            y_delta[:, center - idx] = y_delta[:, center - idx + 1] + y_offset[:, center - idx]

        y_map = y_base + y_delta * extend_scope
        x_map = x_base
        y_map = y_map.permute(0, 2, 1, 3).reshape(b, h * k, w)
        x_map = x_map.permute(0, 2, 1, 3).reshape(b, h * k, w)
    else:
        y_spread = torch.linspace(-center, center, k, device=device, dtype=dtype).view(1, k, 1, 1)
        y_base = y_center + y_spread
        x_base = x_center

        x_delta = x_offset.clone()
        x_delta[:, center] = 0
        for idx in range(1, center + 1):
            x_delta[:, center + idx] = x_delta[:, center + idx - 1] + x_offset[:, center + idx]
            x_delta[:, center - idx] = x_delta[:, center - idx + 1] + x_offset[:, center - idx]

        y_map = y_base
        x_map = x_base + x_delta * extend_scope
        y_map = y_map.permute(0, 2, 3, 1).reshape(b, h, w * k)
        x_map = x_map.permute(0, 2, 3, 1).reshape(b, h, w * k)

    return y_map, x_map


class DSConvBranch(nn.Module):
    def __init__(self, channels, kernel_size=7, morph=0, extend_scope=1.0):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size should be odd."
        assert morph in (0, 1), "morph=0 for x-axis branch, morph=1 for y-axis branch."

        self.kernel_size = kernel_size
        self.morph = morph
        self.extend_scope = extend_scope
        self.offset_conv = nn.Conv2d(channels, 2 * kernel_size, kernel_size=3, padding=1)
        self.offset_norm = nn.GroupNorm(kernel_size, 2 * kernel_size)
        self.offset_act = nn.Tanh()

        if morph == 0:
            self.snake_conv = nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1), stride=(kernel_size, 1))
        else:
            self.snake_conv = nn.Conv2d(channels, channels, kernel_size=(1, kernel_size), stride=(1, kernel_size))

        self.out_norm = nn.GroupNorm(1, channels)
        self.out_act = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (B, C, H, W)
        b, c, h, w = x.shape

        # 预测 2K 个偏移通道: (B, 2K, H, W)，并压缩到 [-1, 1]
        offset = self.offset_act(self.offset_norm(self.offset_conv(x)))

        # 根据 morph 构造蛇形连续坐标；中间采样图会变为 (B, C, H*K, W) 或 (B, C, H, W*K)
        y_map, x_map = _snake_coordinates(offset, self.morph, self.extend_scope)
        grid = _normalize_grid(y_map, x_map, h, w)
        sampled = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

        # 用方向卷积把拉长的 K 个采样点聚合回原空间尺寸: (B, C, H, W)
        out = self.snake_conv(sampled)
        return self.out_act(self.out_norm(out))


class DynamicSnakeConv(nn.Module):
    def __init__(self, channels, kernel_size=7, extend_scope=1.0):
        super().__init__()
        self.channels = channels
        self.conv_x = DSConvBranch(channels, kernel_size, morph=0, extend_scope=extend_scope)
        self.conv_y = DSConvBranch(channels, kernel_size, morph=1, extend_scope=extend_scope)
        self.base_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, channels),
            nn.ReLU(inplace=True),
        )

        self.conv_fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1),
            nn.GroupNorm(1, channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_fuse(torch.cat([self.base_conv(x), self.conv_x(x), self.conv_y(x)], dim=1))


if __name__ == "__main__":
    input_tensor = torch.randn(2, 32, 64, 64)
    module = DynamicSnakeConv(channels=32, kernel_size=7)
    output = module(input_tensor)

    print("input shape :", input_tensor.shape)
    print("output shape:", output.shape)