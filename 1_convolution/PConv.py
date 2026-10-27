import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    """YOLO-style Conv + BN + SiLU block."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        # x: (B, C, H, W)
        return self.act(self.bn(self.conv(x)))


class PConv(nn.Module):
    """Pinwheel-shaped Convolution.

    输入形状:  (B, C_in, H, W)
    输出形状:  stride=1 且 C_out=C_in 时为 (B, C_in, H, W)，可替换同通道 3x3 Conv。
    """

    def __init__(self, in_channels, out_channels=None, kernel_size=3, stride=1):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        assert out_channels % 4 == 0, "out_channels should be divisible by 4."

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        # 四个方向的非对称 padding: left/right/top/bottom 顺序由 ZeroPad2d 接收
        pads = [
            (kernel_size, 0, 1, 0),  # 水平方向分支 1
            (0, kernel_size, 0, 1),  # 水平方向分支 2
            (0, 1, kernel_size, 0),  # 垂直方向分支 1
            (1, 0, 0, kernel_size),  # 垂直方向分支 2
        ]
        self.pad = nn.ModuleList([nn.ZeroPad2d(padding=p) for p in pads])

        branch_channels = out_channels // 4
        # cw 使用 1 x k 条形卷积，负责水平叶片；ch 使用 k x 1 条形卷积，负责垂直叶片。
        self.conv_w = ConvBNAct(in_channels, branch_channels, kernel_size=(1, kernel_size), stride=stride)
        self.conv_h = ConvBNAct(in_channels, branch_channels, kernel_size=(kernel_size, 1), stride=stride)

        # 四个分支拼接后通道数为 out_channels；2 x 2 卷积用于融合方向信息并恢复预期空间尺寸。
        self.fuse = ConvBNAct(out_channels, out_channels, kernel_size=2, stride=1)

    def forward(self, x):
        # x: (B, C_in, H, W)
        x_w0 = self.conv_w(self.pad[0](x))
        x_w1 = self.conv_w(self.pad[1](x))
        x_h0 = self.conv_h(self.pad[2](x))
        x_h1 = self.conv_h(self.pad[3](x))

        # 四个方向分支在通道维拼接:
        # 4 * (B, C_out/4, H+1, W+1) -> (B, C_out, H+1, W+1)  when stride=1
        x = torch.cat([x_w0, x_w1, x_h0, x_h1], dim=1)

        # 2 x 2 融合后恢复为 (B, C_out, H, W)  when stride=1
        return self.fuse(x)


if __name__ == "__main__":
    # 构造一组常见图像特征输入: (batch, channels, height, width)
    input_tensor = torch.randn(2, 64, 128, 128)

    # C_out=C_in 且 stride=1 时，PConv 可以直接替换同通道 3x3 Conv。
    module = PConv(in_channels=64, out_channels=64, kernel_size=3, stride=1)
    output = module(input_tensor)

    # 检查输入输出尺寸是否一致
    print("input shape :", input_tensor.shape)
    print("output shape:", output.shape)