FROM continuumio/miniconda3:latest

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENV=production \
    LD_LIBRARY_PATH=/opt/conda/lib

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.0.0" "torchvision>=0.15.0" \
    && pip install -r requirements.txt \
    && pip uninstall -y opencv-python || true \
    && pip install opencv-python-headless

COPY . .

# Download actual YOLO model from GitHub LFS (pointer file was copied, replace it)
RUN python -c "\
import urllib.request, sys; \
url = 'https://media.githubusercontent.com/media/JavierRodriguezzz/software-cvr-/main/models/yolo_cervix_best.pt'; \
print('Downloading YOLO model (~125 MB)...'); \
urllib.request.urlretrieve(url, 'models/yolo_cervix_best.pt'); \
print('Download complete. Verifying...'); \
import zipfile; \
zipfile.ZipFile('models/yolo_cervix_best.pt'); \
print('YOLO model verified OK')"

RUN mkdir -p uploads previews output_processed

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=3).read()"

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "2", "--timeout", "300", "wsgi:app"]