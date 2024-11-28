FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED "yes"

WORKDIR /opt/app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src .