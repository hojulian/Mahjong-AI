FROM pytorch/pytorch:2.12.0-cuda13.2-cudnn9-devel

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY . .
