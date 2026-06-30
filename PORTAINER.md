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
```

Usa `SESSION_COOKIE_SECURE=false` si entraras por `http://IP_DEL_SERVIDOR:5000`. Cambiala a `true` solo cuando publiques la aplicacion detras de HTTPS.

## Opcion 1: Stack desde repositorio

1. Sube este proyecto a un repositorio Git.
2. En Portainer, entra a `Stacks` y crea un stack nuevo.
3. Selecciona `Repository`.
4. Apunta al repositorio y usa `docker-compose.yml`.
5. Define las variables `SECRET_KEY` y `SESSION_COOKIE_SECURE`.
6. Despliega el stack.

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

Si `docker compose build` falla al resolver `registry-1.docker.io`, el problema esta en la conexion/DNS de Docker Desktop o del servidor, no en el `Dockerfile`. Revisa que Docker tenga acceso a internet y que no falte configurar proxy o DNS.

## Persistencia

El stack crea volumenes Docker para:

- `/app/uploads`
- `/app/previews`
- `/app/output_processed`

Esto mantiene archivos cargados y resultados aunque se recree el contenedor.

Nota: el indice `DICOM_FILES` sigue estando en memoria dentro de la app. Si el contenedor se reinicia, los archivos fisicos permanecen, pero la lista del dashboard se vacia hasta implementar persistencia en base de datos o archivo.
