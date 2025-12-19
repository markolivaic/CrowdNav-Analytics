FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for OpenCV
# FIX: 'libgl1-mesa-glx' is deprecated in newer Debian versions.
# We replaced it with 'libgl1' which provides the required libGL.so.1
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*



RUN pip install --no-cache-dir --default-timeout=3000 torch torchvision numpy ultralytics

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=3000 -r requirements.txt

COPY . .

ENTRYPOINT ["python", "main.py"]