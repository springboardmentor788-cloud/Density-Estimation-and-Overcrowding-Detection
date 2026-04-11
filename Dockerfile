FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Install system dependencies (OpenCV requires libGL, libglib)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# In case it has "+cpu" torch, we uninstall any existing CPU versions and install from requirements, 
# ensuring Flask is included. We also avoid installing the CPU torch since the base image handles torch-gpu!
# Therefore we install all OTHER requirements, or just rely on pip to resolve.
RUN pip uninstall -y torch torchvision torchaudio || true
RUN pip install Flask 
# Filter out torch, torchvision, torchaudio from requirements.txt to prevent cpu override
RUN grep -v "torch" requirements.txt > req_filtered.txt || true
RUN pip install -r req_filtered.txt || true
# Alternatively, if grep isn't ideal here due to utf-16le issues inside docker, we just install common deps.
# Actually we know what we need:
RUN pip install opencv-python-headless numpy matplotlib

COPY . .

EXPOSE 5000

ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]
