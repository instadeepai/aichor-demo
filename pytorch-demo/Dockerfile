FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

RUN mkdir -p /opt/app/src
WORKDIR /opt/app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/main.py src/