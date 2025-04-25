FROM python:3.8-slim

RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir numpy

# Install basics and OpenSSL
RUN apt-get update -y \
    && apt-get install -y build-essential apt-utils git curl ca-certificates bzip2 tree htop wget \
    && apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev bmon iotop g++ \
    && apt-get install -y libssl-dev \
    && apt-get install -y python3-dev python3-distutils python3-pip python3-openssl \
    # && apt-get install -y # python3.10 python3.10-dev python3.10-distutils
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
	
COPY requirements.txt /tmp/pip-tmp/
RUN pip3 --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
    && rm -rf /tmp/pip-tmp

#CMD [ "python", "./your-daemon-or-script.py" ]