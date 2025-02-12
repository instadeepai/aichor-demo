FROM python:3.11-slim

RUN apt update && apt install -y wget

COPY build/requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /app
COPY main.py .