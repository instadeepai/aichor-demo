FROM rayproject/ray:2.36.1-py39-gpu
WORKDIR /app

ENV TZ=Europe/Paris
ENV LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY main.py .
