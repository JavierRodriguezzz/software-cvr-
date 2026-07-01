# Cambiamos a una imagen que ya incluye las dependencias del sistema necesarias para OpenCV y PyTorch
FROM continuumio/miniconda3:latest

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ENV=production

WORKDIR /app

# Instalamos las librerías necesarias de Python directamente sin usar apt-get
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.0.0" "torchvision>=0.15.0" \
    && pip install -r requirements.txt

COPY app.py wsgi.py ./
COPY templates ./templates
COPY models ./models

RUN mkdir -p uploads previews output_processed

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=3).read()"

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "2", "--timeout", "300", "wsgi:app"]
