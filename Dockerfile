FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04
# FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu20.04

ENV PYTHONUNBUFFERED TRUE

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    ca-certificates \
    g++ \
    python3-dev \
    python3-distutils \
    python3-venv \
    openjdk-11-jre-headless \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && cd /tmp \
    && curl -O https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py \
    && rm get-pip.py

RUN python3 -m venv /home/venv

ENV PATH="/home/venv/bin:$PATH"

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN update-alternatives --install /usr/local/bin/pip pip /usr/local/bin/pip3 1

RUN pip install -U pip setuptools

# For CUDA  install 
RUN export USE_CUDA=1
ARG CUDA=1
RUN if [ $CUDA==1 ]; then \ 
        pip install nvgpu; \
    fi

# Extra libraries for opencv
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install --no-install-recommends -y  \
    bzip2 \
    g++ \
    git \
    git-lfs \
    graphviz \
    libgl1-mesa-glx \
    libhdf5-dev \
    openmpi-bin \
    wget \
    python3-tk && \
    rm -rf /var/lib/apt/lists/*

# Pip dependencies
COPY requirements.txt /home/model-server/requirements.txt
RUN pip install --no-cache-dir -r /home/model-server/requirements.txt

# Add user for execute the commands 
RUN useradd -m model-server

# Install extra packages
RUN pip install torchserve torch-model-archiver
RUN pip install pyyaml

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y ffmpeg

# INSTALLING CMAKE
# RUN apt-get update && apt-get -y install build-essential libtool autoconf unzip wget
# RUN apt-get -y install libssl-dev
# RUN apt purge --auto-remove cmake
# RUN mkdir ~/temp && \
#     cd ~/temp && \
#     wget https://cmake.org/files/v3.30/cmake-3.30.5.tar.gz && \
#     tar -xzvf cmake-3.30.5.tar.gz && \
#     cd cmake-3.30.5/ && \
#     ./bootstrap && \
#     make -j$(nproc) && \
#     make install
# RUN hash -r
# RUN ln -s /usr/local/bin/cmake /usr/bin/cmake

# INSTALLING FFMPEG NVIDIA
# RUN apt-get -y install build-essential yasm cmake libtool libc6 libc6-dev unzip \
#     wget libnuma1 libnuma-dev pkgconf
# RUN wget https://www.nasm.us/pub/nasm/releasebuilds/2.15.05/nasm-2.15.05.tar.gz && \
#     tar xzf nasm-2.15.05.tar.gz && \
#     cd nasm-2.15.05 && \
#     ./configure && make -j$(nproc) && make install && \
#     cd .. && rm -rf nasm-2.15.05 nasm-2.15.05.tar.gz

# RUN git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
# RUN cd nv-codec-headers && make && make install && cd ..

# RUN git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg/
# RUN cd ffmpeg && PKG_CONFIG_PATH=/usr/lib/pkgconfig ./configure --enable-nonfree --enable-cuda-nvcc --enable-nvenc --enable-libnpp \
#     --extra-cflags="-I/usr/local/cuda/include -I/usr/local/include" \
#     --extra-ldflags=-L/usr/local/cuda/lib64 --disable-static --enable-shared && \
#     make -j 8 && make install

RUN mkdir /home/dependencies
RUN cd /home/dependencies && \
    git clone https://github.com/hhj1897/face_detection.git && \
    cd /home/dependencies/face_detection && \
    git lfs pull && \
    pip install -e .

RUN pip install setuptools==69.5.1

# Create two folder for models 
RUN mkdir /home/model-server/model-store
# Copy all required models and pipelines inside docker 
COPY model_store /home/model-server/model-store

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
COPY config.properties /home/model-server/config.properties

# Giving rights for execute for entrypoint
RUN chmod +x /usr/local/bin/docker-entrypoint.sh \
    && mkdir -p /home/model-server/tmp \
    && chown -R model-server /home/model-server

# RUN mkdir -p /home/model-server/tmp \
#     && chown -R model-server /home/model-server


# GIVING rights to execute 
RUN chown -R model-server /home/model-server/model-store

EXPOSE 8080 8081 8082

USER model-server
WORKDIR /home/model-server
ENV TEMP=/home/model-server/tmp
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["serve", "curl"]

# # run Torchserve HTTP serve to respond to prediction requests
# ENTRYPOINT ["torchserve", \
#      "--start", \
#      "--ncs", \
#     #  "--ts-config=/home/model-server/config.properties", \
#      "--models", \
#      "mono.mar", \
#      "--model-store", \
#      "/home/model-server/model-store"]
