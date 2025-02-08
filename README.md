# Better Intel AI Docker Image

基于 [Intel ARC/BMG GPU](https://github.com/intel/compute-runtime) 的深度学习环境镜像，搭载全新升级的 Intel GPU 驱动和 oneAPI，预装 Miniconda 以简化 Python 环境管理，并集成了 Intel Extension for PyTorch (IPEX LLM) 及其他依赖，方便构建面向大模型推理的容器化应用。

> **特别说明**  
> 本项目内容参考自文章  
> [《Intel B580 GPU 大模型容器推理实践：构建更好的模型 Docker 容器环境（二）》](https://soulteary.com/2025/02/08/intel-b580-gpu-llm-practice-building-a-better-model-docker-container-environment.html)，  
> 原文作者为 [苏洋](https://soulteary.com/)，  
> 使用 [署名 4.0 国际 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/deed.zh) 协议授权。  
> 转载请注明来源。

## 简介

本项目旨在为 Intel ARC (如 A 系列、B 系列等) 以及后续 Intel GPU 提供一套**更好、更易维护**的容器环境，适配大模型推理及未来的多卡推理场景。相较官方镜像（如 `intel/oneapi`、`intel/deep-learning-essentials`），本项目：

1. **升级**到更新版本的 GPU 驱动及相关组件（如 Compute Runtime、Level Zero Loader 等），支持更多新硬件特性。  
2. **集成** Intel 维护的 PyTorch XPU 扩展（含 Triton XPU）与 IPEX LLM 最新版，便于用户快速构建大模型推理环境。  
3. **内置** Miniconda，灵活管理 Python 依赖，可在同一镜像内并行适配多种模型及工具链。  
4. **支持**从国内源构建镜像，可选加速下载。

> **注意**  
> - 本容器环境 **不** 针对生产环境做性能终极优化，仅提供快速搭建和探索大模型推理的基础设施。  
> - 若需商用部署，仍建议深入评估官方驱动兼容性、性能表现、持续维护投入等。

## 功能特性

- **操作系统基础**: Ubuntu 24.04  
- **GPU 相关依赖**:
  - Intel GPU Compute Runtime（针对 ARC/BMG 进行适配与更新）  
  - Intel Graphics Compiler (IGC)  
  - oneAPI Level Zero Loader  
- **Python/Conda 环境**:
  - Miniconda (默认 Python 3.11)  
  - 国内/海外软件源可选  
- **大模型相关依赖**:
  - ipex-llm[xpu_arc] (Intel Extension for PyTorch / Large Language Models)  
  - PyTorch Triton XPU  
  - HuggingFace Transformers / Accelerate  
  - 额外常见运行包（libpng、libjpeg、libaio 等）

## 目录

1. [使用场景](#使用场景)
2. [构建镜像](#构建镜像)
3. [运行容器](#运行容器)
4. [安装/更新 Python 依赖](#安装更新-python-依赖)
5. [示例：LLM 推理](#示例llm-推理)
6. [已知问题与局限](#已知问题与局限)
7. [交流与贡献](#交流与贡献)
8. [参考链接](#参考链接)
9. [License](#license)

---

## 使用场景

- **Intel GPU 驱动探索**：快速实验新版本驱动/编译器，提高 ARC/BMG 卡在 AI 场景下的可行性。
- **大模型推理测试**：如 [DeepSeek R1 Distill Qwen](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) 等开源模型的推理演示。
- **多模态或大型模型应用开发**：在较新版本的 Transformers、Accelerate、IPEX 等环境下测试可行性。
- **Conda Python 环境**：简化多 Python 包依赖冲突，在同一个镜像内进行模型适配与运行调试。

## 构建镜像

请先克隆/下载本项目，进入包含 `Dockerfile` 的目录。默认会使用海外源构建，如需使用国内源，可加上 `--build-arg USE_CHINA_MIRROR=true` 参数。

```bash
git clone https://github.com/soulteary/better-intel-ai-docker-image.git
cd better-intel-ai-docker-image

# 使用国内源
docker build --build-arg USE_CHINA_MIRROR=true -t better-intel-env:latest .

# 使用海外源
docker build -t better-intel-env:latest .

构建过程会下载并安装多个大型依赖，耐心等待即可。构建完成后可通过 docker images 命令查看镜像是否生成成功。
```

## 运行容器

由于需要使用宿主机 Intel GPU，请确保宿主机已经安装并启用了必要的内核支持。运行命令示例：

```bash
docker run --rm -it \
  --privileged \
  --net=host \
  --device=/dev/dri \
  --memory="16G" \
  --shm-size="16g" \
  -v $(pwd)/your-llm-models:/llm/models \
  better-intel-env:latest
```

- `--device=/dev/dri`：将宿主机的显卡设备映射到容器内，便于容器访问 GPU。
- `--memory & --shm-size`：一些大模型应用对内存/共享内存需求高，建议适当调大。
- `-v $(pwd)/your-llm-models:/llm/models`：模型文件目录映射到容器内 `/llm/models`，实际路径可根据需要修改。


进入容器后，可执行 `clinfo` 命令，或 `ls /dev/dri` 等命令验证 GPU 是否成功挂载。


**提示：进入容器后，如需初始化 oneAPI 变量，可执行**

```bash
. /opt/intel/oneapi/2025.0/oneapi-vars.sh --force
```

## 安装/更新 Python 依赖

容器使用了 Miniconda 管理 Python 包，已默认安装下列关键依赖：

- ipex-llm[xpu_arc]
- pytorch-triton-xpu
- transformers
- accelerate

若需要安装其他依赖包，可在容器内使用 conda 或 pip 命令：

```bash
# 使用 conda
conda install numpy
# 或使用 pip
pip install fastapi uvicorn
```

如需更新依赖，可直接 `pip install -U` 目标依赖。当需要在使用国内环境下更新包时，可手动替换为国内镜像源。

## 示例：LLM 推理

```bash
cd /llm
python app.py
```

具体代码示例参见原文。

若推理过程对 transformers 版本有特殊需求，可通过 `pip install transformers==指定版本` 的方式进行切换。


## 测试命令

你可以简单执行以下命令快速验证环境：

```bash
python -c "import torch; print(torch.__version__)"
python -c "import intel_extension_for_pytorch as ipex; print(ipex.__version__)"
clinfo | grep 'Device'
```

以上命令若正常输出版本号，说明基础环境就绪；`clinfo` 检测到 Intel GPU 即说明已挂载成功。


## 已知问题与局限

- GPU 性能或支持：Intel ARC/BMG 系列显卡在某些框架或场景下支持度不如 NVIDIA/AMD 成熟，仍可能出现兼容性或性能问题。
- 文档滞后：Intel 相关驱动及软件栈更新较频繁，官方文档常常不及时。本镜像也将持续改进，但可能无法覆盖所有特殊场景。
- 容器体积较大：由于内含多项大型依赖（oneAPI、conda、各类驱动库等），镜像体积在数 GB 至十余 GB 之间。
- 多卡与微调：目前仅做基本推理验证，多卡训练或微调需要额外适配。相关依赖（如 deepspeed、vllm 等）可能需要手动编译/适配。


## 参考链接

- [Intel GPU Compute Runtime 项目](https://github.com/intel/compute-runtime)
- [Intel Graphics Compiler (IGC)](https://github.com/intel/intel-graphics-compiler)
- [oneAPI Level Zero](https://github.com/oneapi-src/level-zero)
- [Intel Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch)
- [IPEX LLM 仓库](https://github.com/intel/ipex-llm)
