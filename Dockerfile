FROM --platform=linux/amd64 python:3.12-slim

WORKDIR /app

# Build tools for Cython compilation of strategy binaries
# Note: python:3.12-slim already includes Python.h headers
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libc6-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt cython setuptools

COPY . .

# Auto-compile student strategies (best-effort)
RUN bash tools/build_students.sh || true
