# Despliegue en Portainer

## Requisitos

- Docker con soporte suficiente para instalar PyTorch CPU.
- Salida a internet desde el servidor Docker/Portainer para descargar `python:3.11-slim` y dependencias de Python.
- Los archivos de modelo deben existir en `models/`:
  - `yolo_cervix_best.pt`
  - `unet_fugc_best.pth`
  - `unet_model.py`

## Variables

Crea una variable `SECRET_KEY` en el stack de Portainer o en un archivo `.env` junto al `docker-compose.yml`.

Ejemplo:

```env
SECRET_KEY=cambia_esto_por_un_valor_largo_y_aleatorio
SESSION_COOKIE_SECURE=false
APP_PORT=5000
IMAGE_NAME=secretaria-salud-front:latest
```

Usa `SESSION_COOKIE_SECURE=false` si entraras por `http://IP_DEL_SERVIDOR:5000`. Cambiala a `true` solo cuando publiques la aplicacion detras de HTTPS.

`APP_PORT` controla el puerto publicado en el servidor. Si el puerto `5000` ya esta ocupado, usa por ejemplo `APP_PORT=8080` y entra por `http://IP_DEL_SERVIDOR:8080`.

`IMAGE_NAME` es opcional. Sirve para etiquetar la imagen construida por Docker/Portainer.

## Opcion 1: Stack desde repositorio

1. Sube este proyecto a un repositorio Git.
   - Si usas Git LFS, confirma que Portainer o el servidor descargue los archivos reales de `models/` y no solo punteros LFS.
   - Los modelos actuales pesan aproximadamente 31 MB (`unet_fugc_best.pth`) y 125 MB (`yolo_cervix_best.pt`).
2. En Portainer, entra a `Stacks` y crea un stack nuevo.
3. Selecciona `Repository`.
4. Apunta al repositorio y usa `docker-compose.yml`.
5. Define las variables `SECRET_KEY` y `SESSION_COOKIE_SECURE`.
6. Despliega el stack.

Nota: usa un stack Docker Compose/Standalone. Si tu Portainer esta conectado a un entorno Swarm, la clave `build:` no se usa igual; en ese caso construye y publica la imagen primero, luego despliega apuntando a `IMAGE_NAME`.

## Opcion 2: Build local en servidor

En el servidor donde corre Docker:

```bash
docker compose up -d --build
```

La aplicacion quedara disponible en:

```text
http://IP_DEL_SERVIDOR:5000
```

## Verificacion local

Antes de subirlo a Portainer puedes validar:

```bash
docker compose config
docker compose build
docker compose up -d
```

Tambien puedes validar el estado del contenedor:

```bash
docker compose ps
docker compose logs -f
```

Si `docker compose build` falla al resolver `registry-1.docker.io`, el problema esta en la conexion/DNS de Docker Desktop o del servidor, no en el `Dockerfile`. Revisa que Docker tenga acceso a internet y que no falte configurar proxy o DNS.

El `Dockerfile` instala PyTorch y Torchvision desde el indice CPU oficial para evitar descargar dependencias CUDA/GPU innecesarias. Esto hace el build mas liviano y compatible con servidores Portainer sin GPU.

## Persistencia

El stack crea volumenes Docker para:

- `/app/uploads`
- `/app/previews`
- `/app/output_processed`

Esto mantiene archivos cargados y resultados aunque se recree el contenedor.

Nota: el indice `DICOM_FILES` sigue estando en memoria dentro de la app. Si el contenedor se reinicia, los archivos fisicos permanecen, pero la lista del dashboard se vacia hasta implementar persistencia en base de datos o archivo.
