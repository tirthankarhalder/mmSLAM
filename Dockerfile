FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

RUN apt-get update --yes --quiet && DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    software-properties-common \
    build-essential apt-utils \
    wget curl vim git ca-certificates kmod \
    nvidia-driver-465 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update --yes --quiet
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    python3.8 \
    python3.8-dev \
    python3-pip \ 
    python3.8-distutils \
    python3.8-lib2to3 \
    python3.8-gdbm \
    python3.8-tk \
    pip \
    gfortran musl-dev \
    gcc \
    iputils-ping \
    nano


RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 999 \
    && update-alternatives --config python3 && ln -s /usr/bin/python3 /usr/bin/python

# RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.8
COPY requirements.txt /requirements.txt
COPY . /app

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r /requirements.txt
RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113 --root-user-action=ignore

#special installation of torch geometric with specific missing file installation
RUN pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.11.0+cu113.html \
&& pip install torch-sparse==0.6.15 -f https://data.pyg.org/whl/torch-1.11.0+cu113.html \
&& pip install torch-cluster==1.6.0 -f https://data.pyg.org/whl/torch-1.11.0+cu113.html \
&& pip install torch-spline-conv==1.2.1 -f https://data.pyg.org/whl/torch-1.11.0+cu113.html \
&& pip install torch-geometric==2.0.4

# Add the library path to .bashrc
# RUN echo 'export LD_LIBRARY_PATH=/usr/local/lib/python3.8/dist-packages/torch/lib:$LD_LIBRARY_PATH' >> ~/.bashrc

# Make the environment variable persistent for all future sessions
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.8/dist-packages/torch/lib:$LD_LIBRARY_PATH

# RUN cd Emd && python setup.py install
# ENTRYPOINT [ "python3", "hello.py" ]