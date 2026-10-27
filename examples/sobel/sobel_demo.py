from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# 读取井字图，转换成 (B, C, H, W)，像素范围为 [0, 1]。
folder = Path(__file__).resolve().parent
image = np.array(Image.open(folder / "jing.png").convert("L"), dtype=np.float32) / 255.0
x = torch.from_numpy(image)[None, None]

sobel_x = torch.tensor([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
], dtype=torch.float32).view(1, 1, 3, 3)

sobel_y = torch.tensor([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1],
], dtype=torch.float32).view(1, 1, 3, 3)

# 保留正负号的梯度，与 DEGConv 中的计算一致。
dx = F.conv2d(x, sobel_x, padding=1)
dy = F.conv2d(x, sobel_y, padding=1)
print("input:", tuple(x.shape))
print("dx:", tuple(dx.shape), "dy:", tuple(dy.shape))

# 仅显示时取绝对值；裁掉最外圈，避免零填充产生的图像边框响应。
fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
for ax, data, title in zip(
    axes, [x, dx.abs() / 4, dy.abs() / 4],
    ["Input", "Sobel x: vertical edges", "Sobel y: horizontal edges"],
):
    ax.imshow(data[0, 0, 1:-1, 1:-1].numpy(), cmap="gray", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")

fig.savefig(folder / "sobel_result.png", dpi=160)
plt.show()
