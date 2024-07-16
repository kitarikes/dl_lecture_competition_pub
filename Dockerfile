FROM FROM nvidia/cuda:12.0.0-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.3/compat

COPY requirements.txt /tmp/requirements.txt

RUN apt update
RUN apt upgrade -y

# ubuntu package install
RUN apt install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        libffi-dev \
        libssl-dev \
        libbz2-dev \
        python3-pip \
        python3-setuptools \
        wget \
        git \
        tzdata \
        libgl1-mesa-dev \
        libjpeg-dev \
        zlib1g-dev \
        libncurses5-dev \
        libgdbm-dev \
        libnss3-dev \
        libreadline-dev \
        libffi-dev \
        libsqlite3-dev \
        libglib2.0-0 \
        && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        && sh Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda \
        && rm Miniconda3-latest-Linux-x86_64.sh \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

# Set the environment path to include the Conda bin directory
ENV PATH="/opt/miniconda/bin:$PATH"

WORKDIR /workspace

COPY ./requirements.txt requirements.txt

# Install Python packages and setup Conda environment in a single RUN command
RUN conda create -y -n vqa-env python=3.8 \
    && echo "conda activate vqa-env" >> ~/.bashrc \
    && echo "conda activate vqa-env" > /etc/profile.d/conda.sh \
    && . /opt/miniconda/etc/profile.d/conda.sh \
    && conda activate vqa-env \
    && ls -a \
    && pip install -r /tmp/requirements.txt

# Set the default command for the container
CMD ["/bin/bash"]
CMD ["tail", "-f", "/dev/null"]