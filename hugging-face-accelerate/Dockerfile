FROM alpine/curl AS vscode-installer

RUN mkdir /aichor
RUN curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output /aichor/vscode_cli.tar.gz
RUN tar -xf /aichor/vscode_cli.tar.gz -C /aichor

FROM astral/uv:0.8-python3.11-trixie-slim
COPY --from=vscode-installer /aichor /aichor

WORKDIR /opt/app

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y iputils-ping \
    net-tools \
    libnuma1 \
    libsubunit0 \
    libpci-dev \
    libibverbs1 \
    libibverbs-dev \
    ibverbs-utils \
    librdmacm1 \
    librdmacm-dev \
    libibumad3 \
    libibumad-dev \
    infiniband-diags \
    ibutils \
    numactl \
    perftest \
    iproute2 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED="1"

COPY pyproject.toml uv.lock ./
RUN uv sync --locked
ENV PATH="/opt/app/.venv/bin:$PATH"

# Finally copy source code
COPY src .

# Setup NCCL variables
ENV NCCL_DEBUG=INFO
ENV NCCL_IB_GID_INDEX=1
