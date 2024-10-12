# 基于官方的 CUDA 镜像构建环境
FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

ARG GIT_USER=anonymous
ARG GIT_EMAIL=anonymous@gmail.com

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

RUN yes | unminimize

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    unzip \
    ca-certificates \
    sudo \
    tzdata \
    openssh-server \
    openssh-client \
    nano \
    htop \
    net-tools \
    openjdk-11-jdk \
    nodejs \
    npm \
    screen \
    python3 \
    python3-pip \
    python3-virtualenv \
    uuid-dev \
    libz-dev \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# 设置时区为中国
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && echo "Asia/Shanghai" > /etc/timezone

COPY ./container_key/id_rsa.pub /tmp/id_rsa.pub
COPY ./container_key/id_rsa /tmp/id_rsa
COPY ./dev_key/id_ed25519.pub /tmp/id_ed25519.pub
RUN mkdir -p /root/.ssh && \
    cat /tmp/id_rsa.pub >> /root/.ssh/authorized_keys && \
    cat /tmp/id_ed25519.pub >> /root/.ssh/authorized_keys && \
    chmod 700 /root/.ssh && \
    chmod 600 /root/.ssh/authorized_keys && \
    cp /tmp/id_rsa /root/.ssh/id_rsa && \
    chmod 600 /root/.ssh/id_rsa && \
    cp /tmp/id_rsa.pub /root/.ssh/id_rsa.pub && \
    chmod 644 /root/.ssh/id_rsa.pub && \
    rm /tmp/id_rsa.pub /tmp/id_rsa /tmp/id_ed25519.pub

# 安装 Miniconda (Python 环境)
RUN curl -o ~/miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p ~/miniconda3 \
    && rm ~/miniconda.sh \
    && ~/miniconda3/bin/conda init bash

# 创建并激活 conda 环境
RUN ~/miniconda3/bin/conda create -n mononn python=3.8 -y

# 设置默认的 conda 环境
SHELL ["/root/miniconda3/bin/conda", "run", "-n", "mononn", "/bin/bash", "-c"]

RUN pip install transformers==4.20.0 opt_einsum && \
    conda install numpy wheel packaging requests && \
    pip install keras_preprocessing --no-deps

RUN conda config --set auto_activate_base false

RUN echo 'export PATH="$PATH:~/miniconda3/bin"' >> ~/.bashrc \
    && echo 'export PATH="$PATH:/usr/local/cuda/bin"' >> ~/.bashrc \
    && echo 'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"' >> ~/.bashrc \
    && echo 'export CUDA_HOME="/usr/local/cuda"' >> ~/.bashrc \
    && echo 'conda activate mononn' >> ~/.bashrc

RUN git lfs install \
    && git config --global user.email $GIT_EMAIL \
    && git config --global user.name $GIT_USER

RUN mkdir -p /data/models \
    && ssh-keyscan hf.co >> ~/.ssh/known_hosts \
    && git clone git@hf.co:google-bert/bert-base-uncased /data/models/bert-base-uncased --depth 1 \
    && git clone git@hf.co:google-bert/bert-large-cased /data/models/bert-large-cased --depth 1

RUN cd /tmp && \
    wget https://github.com/bazelbuild/bazel/releases/download/5.0.0/bazel-5.0.0-installer-linux-x86_64.sh \
    && chmod +x bazel-5.0.0-installer-linux-x86_64.sh \
    && ./bazel-5.0.0-installer-linux-x86_64.sh \
    && rm bazel-5.0.0-installer-linux-x86_64.sh \
    && echo 'export PATH="$PATH:/usr/local/bin"' >> ~/.bashrc \
    && echo 'source /usr/local/lib/bazel/bin/bazel-complete.bash' >> ~/.bashrc

COPY . /mononn
WORKDIR /mononn

RUN cd tensorflow_mononn \
    && cp ../.tf_configure.bazelrc.backup .tf_configure.bazelrc

# RUN bazel build //tensorflow/tools/pip_package:build_pip_package --nocheck_visibility \
#     && ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg \
#     && pip install /tmp/tensorflow_pkg/tensorflow-2.9.2-cp38-cp38-linux_x86_64.whl --force-reinstall

# 设置容器启动时的默认命令
CMD ["/bin/bash", "-c", "sudo service ssh start && /bin/bash"]