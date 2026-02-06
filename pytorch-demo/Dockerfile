FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN mkdir -p /opt/app/src
WORKDIR /opt/app

RUN python3 -m venv /opt/venv && \
    pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/