# Base image with Conda
FROM continuumio/miniconda3

# Set working directory
WORKDIR /workspace

# Copy environment and source code
COPY environment.yaml ./environment.yaml
COPY . .

# Install system packages (for pygame, ffmpeg, image/video processing)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create Conda env from exported config
RUN conda env create -f environment.yaml

# Use the created environment by default
ENV PATH="/opt/conda/envs/qwen2.5VL-R1/bin:$PATH"
ENV CONDA_DEFAULT_ENV=qwen2.5VL-R1

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONPATH="/workspace/src:$PYTHONPATH" \
    TOKENIZERS_PARALLELISM=false

# Default command (optional)
CMD ["python", "video_generator.py", "--help"]
