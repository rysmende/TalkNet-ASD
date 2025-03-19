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

RUN mkdir /home/dependencies
RUN cd /home/dependencies && \
    git clone https://github.com/hhj1897/face_detection.git && \
    cd /home/dependencies/face_detection && \
    git lfs pull && \
    pip install -e .

RUN pip install setuptools==69.5.1

# Install FastAPI dependencies
RUN pip install fastapi uvicorn python-multipart pydantic

# FastAPI Configuration
# Create app directory
RUN mkdir -p /home/app
WORKDIR /home/app

# Copy application code
COPY src/ /home/app/src/
COPY app.py /home/app/

# Create user for running the application
RUN useradd -m appuser && \
    chown -R appuser:appuser /home/app

# Switch to non-root user
USER appuser

# Expose port for FastAPI
EXPOSE 8080

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
