FROM python:3.8-slim

RUN pip install tensorboardX boto3

WORKDIR /app
COPY ./src ./src
COPY main.py .