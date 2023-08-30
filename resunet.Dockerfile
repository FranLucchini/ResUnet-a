FROM tensorflow/tensorflow:2.0.0-gpu-py3

COPY ./requirements.txt /build/

# RUN DEBIAN_FRONTEND=noninteractive apt install python3-pip -y

RUN python --version
RUN pip --version


# 1) especificar que trabajamos con python 3: python3, pip3
# 2) Actualizar python a 3.7

RUN pip install -r /build/requirements.txt