# syntax=docker/dockerfile:1

# --- Base image ---
FROM python:3.9-slim AS base

# Set working directory
WORKDIR /app

# Install system dependencies (minimal for OpenCV and PyTorch)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libfontconfig1 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    gfortran \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# --- Install Python dependencies first for better caching ---
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Copy only the application code ---
COPY app-menu.py ./
COPY utils.py ./
COPY config.yml ./
COPY models/ ./models/
COPY runs/ ./runs/

# --- Create necessary directories ---
RUN mkdir -p videos feedback_data data

# --- Set environment variables ---
ENV PYTHONPATH=/app
ENV DISPLAY=:99

# --- Use a non-root user for security ---
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# --- Expose port if needed for web interface ---
EXPOSE 8080

# --- Entrypoint and default command ---
ENTRYPOINT ["python"]
CMD ["app-menu.py"] 