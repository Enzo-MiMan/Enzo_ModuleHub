import torch
x = torch.randn(2, 64, 32, 32)

f = torch.fft.rfft2(x)
x_back = torch.fft.irfft2(f, s=x.shape[-2:])   # 需要指定 目标空间尺寸

print(x_back.shape)  # torch.Size([2, 64, 32, 32])
print(x_back.dtype)  # torch.float32风

# 浮点运算有微小误差，通常接近而非逐位相等
print(torch.allclose(x, x_back, atol=1e-5))