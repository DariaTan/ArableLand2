# Use NVIDIA CUDA base image for GPU support
# FROM nvidia/cuda:12.2.2-base-ubuntu22.04 as builder
FROM python:3.10-slim

# Set non-interactive shell to avoid prompts during build
# ARG DEBIAN_FRONTEND=noninteractive

# Remove any third-party apt sources to avoid issues with expiring keys
RUN rm -f /etc/apt/sources.list.d/*.list

# Install some basic utilities
RUN apt-get update \
    && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY poetry.lock pyproject.toml ./

# Enable venv
RUN python -m pip install --no-cache-dir poetry==1.8.1 \
    && poetry config virtualenvs.in-project true \
    && poetry install --no-interaction --no-ansi
ENV PATH="${PATH}:/app/.venv/bin"

# Create and set the working directory
RUN mkdir /app 
WORKDIR /app

# Define default command use the new environment:
RUN echo "poetry shell" >> ~/.bashrc
# SHELL ["/bin/bash", "--login", "-c"]
