import torch
import torch.nn as nn


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, groups=1):
        if padding is None:
            # 根据卷积核形状自动补齐 padding，保证 stride=1 时空间尺寸不变
            if isinstance(kernel_size, tuple):
                padding = tuple(k // 2 for k in kernel_size)
            else:
                padding = kernel_size // 2

        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


class CKConv(nn.Module):
    """Chinese Knot Convolution 的独立 PyTorch 版本。

    输入:  x, shape 为 (B, C, H, W)
    输出: out, shape 为 (B, C, H, W)
    说明: 默认 stride=1，不改变空间尺寸，适合作为即插即用的 spatial mixing 模块。
    """

    def __init__(self, channels, kernels=(3, 5, 7), stride=1, fuse_groups=16):
        super().__init__()
        if any(k not in (3, 5, 7, 9) for k in kernels):
            raise ValueError("kernels 必须从 3、5、7、9 中选择")
        if channels % 2 != 0:
            raise ValueError("channels 需要为偶数，因为每个分支会先压到 C/2")
        if channels % fuse_groups != 0 or (len(kernels) * channels) % fuse_groups != 0:
            raise ValueError("channels 与 len(kernels) * channels 需要能被 fuse_groups 整除")
        if stride != 1:
            raise ValueError("当前 CKConv 仅支持 stride=1")

        self.channels = channels
        self.kernels = tuple(kernels)
        self.stride = stride
        mid_channels = channels // 2

        # 每个尺度 k 都有一组 body 分支和 Chinese-knot 条带分支
        self.branches = nn.ModuleDict()
        for k in self.kernels:
            # body 分支: 3x3 group 卷积，聚合局部空间信息
            # (B, C, H, W) -> (B, C/2, H, W)
            self.branches[f"k{k}_body"] = ConvBNAct(
                channels, mid_channels, kernel_size=(3, 3), stride=1, padding=(1, 1), groups=mid_channels
            )

            # head_h 分支: 1xk 横向条带卷积，扩大水平方向感受野
            # (B, C, H, W) -> (B, C/2, H, W)
            self.branches[f"k{k}_head_h"] = ConvBNAct(
                channels, mid_channels, kernel_size=(1, k), stride=stride, padding=(0, (k - 1) // 2), groups=mid_channels
            )

            # head_v 分支: kx1 纵向条带卷积，扩大垂直方向感受野
            # (B, C/2, H, W) -> (B, C/2, H, W)
            self.branches[f"k{k}_head_v"] = ConvBNAct(
                mid_channels, mid_channels, kernel_size=(k, 1), stride=1, padding=((k - 1) // 2, 0), groups=mid_channels
            )

            # 1x1 grouped convolution 将当前尺度输出恢复到 C 个通道
            # (B, C/2, H, W) -> (B, C, H, W)
            self.branches[f"k{k}_conv2"] = nn.Conv2d(mid_channels, channels, kernel_size=1, groups=mid_channels, bias=True)

        # 多尺度输出拼接后做 1x1 grouped fusion
        # (B, len(kernels)*C, H, W) -> (B, C, H, W)
        self.conv_fuse = nn.Conv2d(len(self.kernels) * channels, channels, kernel_size=1, groups=fuse_groups, bias=True)

    def forward(self, x):
        # x: (B, C, H, W)
        outputs = []

        for k in self.kernels:
            # Chinese-knot 条带路径: 先横向 1xk，再纵向 kx1
            # strip: (B, C/2, H, W)
            strip = self.branches[f"k{k}_head_h"](x)
            strip = self.branches[f"k{k}_head_v"](strip)

            # body 路径: 3x3 局部卷积分支
            # body: (B, C/2, H, W)
            body = self.branches[f"k{k}_body"](x)

            # 同一尺度内融合局部纹理与方向条带上下文
            # out: (B, C/2, H, W) -> (B, C, H, W)
            out = body + strip
            out = self.branches[f"k{k}_conv2"](out)
            outputs.append(out)

        # 多尺度拼接: [(B, C, H, W)] * K -> (B, K*C, H, W)
        multi_scale = torch.cat(outputs, dim=1)

        # 1x1 分组融合，恢复输入通道数，输出空间尺寸不变
        out = self.conv_fuse(multi_scale)
        return out


if __name__ == "__main__":
    # 构造图像数据
    input_tensor = torch.randn(2, 32, 64, 64)

    module = CKConv(channels=32, kernels=(3, 5, 7))
    output_tensor = module(input_tensor)

    print("input shape :", input_tensor.shape)
    print("output shape:", output_tensor.shape)