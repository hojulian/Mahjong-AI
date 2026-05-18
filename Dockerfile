FROM pytorch/pytorch:2.12.0-cuda13.2-cudnn9-devel

# Allow pip to install into the system Python (PEP 668 / Debian externally-managed-environment)
RUN echo "[global]\nbreak-system-packages = true" > /etc/pip.conf

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
