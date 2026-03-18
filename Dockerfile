# Build and run on macOS arm64:
#   docker build --platform linux/amd64 -t crmrr .
#   docker compose run crmrr python main.py
#
# Note: torch 2.0.0 wheels for arm64 are not on PyPI;
# using --extra-index-url for CPU wheels (amd64 emulation via Rosetta).

FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY . .
