FROM rayproject/ray:2.2.0-cpu

RUN pip install tensorboardX boto3

WORKDIR /app
COPY ./src ./src
COPY main.py .