# MobileMambaBlock 独立提取版

从当前 MobileMamba 工作区提取完整 Block 及其依赖，输入输出均为 `[B, C, H, W]`，无需原项目的 `model` 包、注册器或配置系统。

## 快速使用

```bash
python -m pip install -r requirements.txt
python example.py --device cpu
```

在当前目录直接导入：

```python
import torch
from mobilemamba_block import MobileMambaBlock

block = MobileMambaBlock(type='s', ed=64)
x = torch.randn(2, 64, 32, 32)
y = block(x)  # [2, 64, 32, 32]
```

也可把整个文件夹复制到工程，在父目录使用 `from MobileMambaBlock import MobileMambaBlock`。请一起复制 `lib_mamba`，不能只复制入口文件。

## CUDA 安装

在 Linux + NVIDIA GPU、CUDA 版 PyTorch、匹配的 CUDA Toolkit/nvcc 与 C++ 编译器环境中：

```bash
python -m pip install triton
cd lib_mamba/kernels/selective_scan
# 根据目标显卡填写；不设置时由 PyTorch 根据可见 GPU 选择。
# export TORCH_CUDA_ARCH_LIST="8.6"
python -m pip install . --no-build-isolation
cd ../../..
python example.py --device cuda
```

默认编译 `selective_scan_cuda_oflex`，这是 Block 默认 `v052d` 路径在 CUDA 上选择的后端。`setup.py` 的 `MODES` 可另外加入 `core`、`ndstate`、`nrow`；这些源码全部保留。安装额外后端不会自动改变 SS2D 中原有的后端选择逻辑。Triton 加速 cross scan/merge；缺少 Triton 时使用 PyTorch 实现。缺少扫描 CUDA 扩展时源码提供慢速 PyTorch 回退，不代表 CUDA 扩展安装成功。macOS 无法验证 NVIDIA CUDA。

## 文件内容

- `mobilemamba_block.py`：完整 Block、MobileMambaModule、MBWTConv2d、小波变换、局部分支、FFN 和残差。
- `lib_mamba/vmambanew.py`：SS2D 及其上游辅助类；删除后续完整骨干网络与 FLOPs 依赖。
- `lib_mamba/csm_triton*.py`：扫描和合并、Triton kernel、PyTorch 实现与原有检查工具。
- `lib_mamba/csms6s.py`：Selective Scan 的 autograd 包装与参考实现。
- `lib_mamba/kernels/selective_scan/`：完整原始 CUDA/C++/头文件、构建脚本、说明和原始测试。
- `SOURCE_MANIFEST.json`：原工作区文件哈希及基准提交。源码包含用户已有本地修改，并非只读取 Git HEAD。

## 接口与原实现行为

保留原始签名：

```python
MobileMambaBlock(type, ed, global_ratio=0.25, local_ratio=0.25,
                 kernels=5, drop_path=0., has_skip=True,
                 ssm_ratio=1, forward_type='v052d')
```

- `type='s'`；原实现其他值不会创建 mixer。
- `ed` 为输入通道数，输出通道和空间大小不变。
- global 分支通道由比例计算后就近取 16 的倍数；local 为局部深度卷积分支，其余通道直连。请保证取整后的通道划分非负；建议从 `ed=64` 或更大开始。
- 默认 `v052d` 是双向扫描，`d_state=1`、`k_group=2`、小波 `db1`、一级分解。小波分支支持奇数尺寸的填充和裁剪。
- 原始 `has_skip=True` 在内部多次残差外还执行 `shortcut + drop_path(x)`；零 BN 初始化时，默认输出约为 `2*x`，这里按原样保留。
- 原 SS2D 还保留实验性接口；本模块以 MobileMamba 默认 `v052d` 路径为使用和验证目标。

## 移植改动与验证

1. 入口改用相对/本地导入，移除模型注册器、分类头、下采样与完整骨干。
2. `lib_mamba/__init__.py` 改为空包入口，避免导入另一套完整 VSSM。
3. 给两份扫描文件中 Triton 定义增加 `if WITH_TRITON` 保护，修复未安装 Triton 时导入失败的问题。
4. 构建脚本移除固定 GPU 架构列表（尤其 `sm_120`），由 PyTorch 和 `TORCH_CUDA_ARCH_LIST` 选择目标；CUDA/C++ kernel 内容保持原样。
5. `example.py` 会检查偶数与奇数尺寸的前向、反向、输出形状、有限值与扫描参数梯度。仅在测试内把 BN 权重设为 1，以避免原始零初始化掩盖扫描梯度。

具体本机验证结果见 `VALIDATION.md`。原始源码遵循随附 LICENSE；CUDA 文件保留其各自版权与许可声明。
