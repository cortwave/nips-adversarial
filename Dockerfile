FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python3 \
        python3-dev \
        python3-pip \
        rsync \
        software-properties-common \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install setuptools
RUN pip3 --no-cache-dir install \
        Pillow \
        h5py \
        ipython \
        ipykernel \
        matplotlib \
        numpy \
        pandas \
        scipy \
        scikit-learn \
        Fire \
        Keras==2.0.6 \
        tensorflow-gpu \
        foolbox

RUN ln -s /usr/bin/python3 /usr/bin/python

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

CMD "/bin/sh"
