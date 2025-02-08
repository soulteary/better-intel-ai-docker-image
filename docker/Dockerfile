FROM intel/deep-learning-essentials:2025.0.2-0-devel-ubuntu24.04
LABEL maintainer="soulteary <soulteary@gmail.com>"
LABEL description="Dockerfile for Intel GPU based deep learning development environment."

ARG USE_CHINA_MIRROR=false
ENV USE_CHINA_MIRROR=$USE_CHINA_MIRROR

# Set up the mirror for Ubuntu
RUN if [ "$USE_CHINA_MIRROR" = "true" ]; then \
        sed -i 's/\(archive\|security\).ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/ubuntu.sources; \
    fi
# Install the necessary packages
RUN apt-get update && apt upgrade -y && \
    apt-get install -y --no-install-recommends clinfo && \
    rm -rf /var/lib/apt/lists/*

# Install the necessary Python installer
# https://soulteary.com/2022/12/09/use-docker-to-quickly-get-started-with-the-chinese-stable-diffusion-model-taiyi.html
ENV PATH="/root/miniconda3/bin:${PATH}"
ENV CONDA_URL=https://repo.anaconda.com/miniconda/
ENV CONDA_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/
RUN if [ "$USE_CHINA_MIRROR" = "true" ]; then export CONDA_URL=$CONDA_MIRROR; fi; \
    curl -LO $CONDA_URL/Miniconda3-latest-Linux-x86_64.sh && \
    mkdir /root/.conda && \
    bash Miniconda3-latest-Linux-x86_64.sh -b && \
    rm -f Miniconda3-latest-Linux-x86_64.sh

# https://soulteary.com/2025/01/17/guide-to-setting-up-ubuntu-24-04-basic-development-environment.html
RUN if [ "$USE_CHINA_MIRROR" = "true" ]; then \
        conda config --set show_channel_urls true && \
        conda config --remove-key channels && \
        conda config --add channels defaults && \
        conda config --add default_channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2 && \
        conda config --add default_channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r && \
        conda config --add default_channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main && \
        ( echo "custom_channels:" >> ~/.condarc ) && \
        ( echo "  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud" >> ~/.condarc ) && \
        ( echo "  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud" >> ~/.condarc ); \
    fi

# Remove breaks install packages
RUN apt-get remove -y libze-dev libze-intel-gpu1

# Intel GPU OpenCL Driver and Intel GPU Compute Runtime
# https://github.com/intel/compute-runtime/releases/tag/24.52.32224.5
WORKDIR /tmp/neo
RUN curl -LO https://github.com/intel/intel-graphics-compiler/releases/download/v2.5.6/intel-igc-core-2_2.5.6+18417_amd64.deb && \
    curl -LO https://github.com/intel/intel-graphics-compiler/releases/download/v2.5.6/intel-igc-opencl-2_2.5.6+18417_amd64.deb && \
    curl -LO https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-level-zero-gpu-dbgsym_1.6.32224.5_amd64.ddeb && \
    curl -LO https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-level-zero-gpu_1.6.32224.5_amd64.deb && \
    curl -LO https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-opencl-icd-dbgsym_24.52.32224.5_amd64.ddeb && \
    curl -LO https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-opencl-icd_24.52.32224.5_amd64.deb && \
    curl -LO https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/libigdgmm12_22.5.5_amd64.deb && \
    curl -LO https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/ww52.sum && \
    sha256sum -c ww52.sum && \
    dpkg -i *.deb && \
    rm -rf *

# oneAPI Level Zero Loader
# https://github.com/oneapi-src/level-zero/releases/tag/v1.20.2
WORKDIR /tmp/level-zero
RUN curl -LO https://github.com/oneapi-src/level-zero/releases/download/v1.20.2/level-zero_1.20.2+u24.04_amd64.deb && \
    curl -LO https://github.com/oneapi-src/level-zero/releases/download/v1.20.2/level-zero-devel_1.20.2+u24.04_amd64.deb && \
    dpkg -i *.deb && \
    rm -rf *.deb

# Intel GPU PyTorch Environment
RUN conda install python=3.11 -y
ENV PIP_ROOT_USER_ACTION=ignore
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=true
ENV IPEX_LLM_URL=https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
ENV IPEX_LLM_MIRROR=https://pytorch-extension.intel.com/release-whl/stable/xpu/cn/
ENV PYTORCH_TRITON_XPU_URL=https://download.pytorch.org/whl/nightly/xpu
ENV PYTORCH_TRITON_XPU_MIRROR=https://mirror.sjtu.edu.cn/pytorch-wheels/nightly/xpu
RUN if [ "$USE_CHINA_MIRROR" = "true" ]; then export IPEX_LLM_URL=$IPEX_LLM_MIRROR; fi; \
    if [ "$PYTORCH_TRITON_XPU_URL" = "true" ]; then export PYTORCH_TRITON_XPU_URL=$PYTORCH_TRITON_XPU_MIRROR; fi; \
    pip install --pre --upgrade ipex-llm[xpu_arc]==2.2.0b20250205 --extra-index-url $IPEX_LLM_URL && \
    pip install --pre pytorch-triton-xpu==3.2.0+gite98b6fcb --index-url $PYTORCH_TRITON_XPU_URL && \
    rm -Rf /root/.cache/pip

# Fix conda libc++ errors
RUN rm /root/miniconda3/lib/libstdc++.so.6 && \
    rm /root/miniconda3/lib/libstdc++.so.6.0.29 && \
    rm /root/miniconda3/lib/libstdc++.so && \
    ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.33 /root/miniconda3/lib/libstdc++.so.6 && \
    ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.33 /root/miniconda3/lib/libstdc++.so.6.0.29 && \
    ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.33 /root/miniconda3/lib/libstdc++.so

# Fix Intel IPEX LLM errors and warnings
RUN apt-get update && apt-get install -y --no-install-recommends libpng16-16 libjpeg-turbo8 libaio-dev && \
    rm -rf /var/lib/apt/lists/* && \    
    pip install -U transformers && \
    pip install 'accelerate>=0.26.0'
