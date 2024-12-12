FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=US/Pacific

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-get update && apt-get install -y --no-install-recommends wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb

# Install dependencies and command-line tools.
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    g++ \
    wget \
    bzip2 \
    git \
    vim \
    tmux \
    htop \
    git \
    zip \
    unzip \
    ca-certificates \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    patchelf \
    libglu1-mesa \
    libxext6 \
    libxtst6 \
    libxrender1 \
    libxi6 \
    libjpeg-dev \
    libpng-dev \
    libopenblas-dev \
    libopencv-dev \
    libyaml-dev \
    libavformat-dev \
    libavcodec-dev \
    libswscale-dev \
    libavutil-dev \
    libavfilter-dev \
    libavdevice-dev \
    libswresample-dev \
    less \
    groff \
    mpich \
    ninja-build \
    python-dev \
    python3-pip

# Then clean up apt-get cache.
RUN apt-get clean && rm -rf /var/lib/apt/lists/*


# Set CUDA_ROOT
RUN export CUDA_HOME="/usr/local/cuda"

# pytorch + ml libraries
RUN pip install tensorboard cmake onnx   # cmake from apt-get is too old
RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install 'git+https://github.com/facebookresearch/fvcore'

ENV FORCE_CUDA="1"

RUN pip install pycocotools
RUN pip install ninja yacs cython matplotlib tqdm shapely
RUN pip install opencv-python==4.4.0.40
RUN pip install sklearn
RUN pip install scikit-image
RUN pip install timm==0.4.12
RUN pip install setuptools==59.5.0
RUN pip install tensorboardX

# Install detectron2
RUN pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

# Install aistron
RUN pip install git+https://github.com/trqminh/aistron@v0.1.0
