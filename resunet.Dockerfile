FROM tensorflow/tensorflow:2.0.0-gpu-py3

COPY ./requirements.txt /build/

# RUN DEBIAN_FRONTEND=noninteractive apt install python3-pip -y

RUN python --version
RUN pip --version


# 1) especificar que trabajamos con python 3: python3, pip3
# 2) Actualizar python a 3.7

RUN pip install -r /build/requirements.txt

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt update && apt install -y libsm6 libxext6
RUN apt-get install -y libxrender-dev

# RUN apt-get install libcudnn7=7.4.1.5-1+cuda10.0 -y
# RUN apt-get install libcudnn7-dev=7.4.1.5-1+cuda10.0 -y