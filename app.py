import os
import uuid
import time
from datetime import datetime, timedelta
from collections import defaultdict
from functools import wraps
from pathlib import Path

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, jsonify, abort, send_file, send_from_directory
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import pydicom
from PIL import Image
import numpy as np
import cv2
import torch
from ultralytics import YOLO

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(64).hex())

BASE_DIR = Path(__file__).parent

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("SESSION_COOKIE_SECURE", "false").lower() == "true",
    PERMANENT_SESSION_LIFETIME=timedelta(hours=8),
    UPLOAD_FOLDER=BASE_DIR / "uploads",
    PREVIEW_FOLDER=BASE_DIR / "previews",
    OUTPUT_FOLDER=BASE_DIR / "output_processed",
    MAX_CONTENT_LENGTH=500 * 1024 * 1024,
)

# Asegurar la existencia de los directorios esenciales
app.config["UPLOAD_FOLDER"].mkdir(exist_ok=True)
app.config["PREVIEW_FOLDER"].mkdir(exist_ok=True)
app.config["OUTPUT_FOLDER"].mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".dcm", ".dicom"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff", ".tif"}
ALL_EXTENSIONS = ALLOWED_EXTENSIONS | IMAGE_EXTENSIONS

# --- Lazy Loading de Modelos Inteligentes ---
_yolo_model = None
_unet_model = None
_unet_device = None

def get_yolo():
    global _yolo_model
    if _yolo_model is None:
        model_path = BASE_DIR / "models" / "yolo_cervix_best.pt"
        if model_path.exists():
            _yolo_model = YOLO(str(model_path))
            print("[IA] YOLOv11 cargado con éxito.")
        else:
            print("[IA] Warning: Modelo YOLO no encontrado en disco.")
    return _yolo_model

def get_unet():
    global _unet_model, _unet_device
    if _unet_model is None:
        model_path = BASE_DIR / "models" / "unet_fugc_best.pth"
        if model_path.exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "unet_model", BASE_DIR / "models" / "unet_model.py"
            )
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                _unet_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                _unet_model = mod.UNetSegmenter(model_path, _unet_device)
                print(f"[IA] U-Net cargado dinámicamente en {_unet_device}")
            else:
                print("[IA] Error crucial: No se pudo cargar el módulo estructural de U-Net.")
        else:
            print("[IA] Warning: Pesos del modelo U-Net no encontrados.")
    return _unet_model, _unet_device

# --- Almacenamiento Estático y Variables de Control (En Memoria) ---
USERS = {
    "javier": generate_password_hash("javier123"),
    "admin": generate_password_hash("admin2024"),
    "marcos": generate_password_hash("marcos123"),
    "anahi": generate_password_hash("anahi123"),
    "saul": generate_password_hash("saul123"),
    "pablo": generate_password_hash("jpsr2026")
}

DICOM_FILES = {}

RATE_LIMIT_WINDOW = 60
RATE_LIMIT_MAX = 30
_rate_store = defaultdict(list)

LOCKOUT_MAX_ATTEMPTS = 5
LOCKOUT_DURATION = timedelta(minutes=15)
_failed_attempts = {}

# --- Funciones Helper e Infraestructura ---

def allowed_file(filename):
    return Path(filename).suffix.lower() in ALL_EXTENSIONS

def is_dicom(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def dicom_to_preview(dicom_path, png_path):
    ds = pydicom.dcmread(str(dicom_path))
    if not hasattr(ds, "pixel_array"):
        return False
    arr = ds.pixel_array
    if hasattr(ds, "PhotometricInterpretation") and ds.PhotometricInterpretation == "MONOCHROME1":
        arr = np.amax(arr) - arr
    if arr.ndim == 2:
        arr = arr.astype(float)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
        arr = arr.astype(np.uint8)
        img = Image.fromarray(arr, mode="L")
    else:
        img = Image.fromarray(arr)
    img.save(str(png_path))
    return True

def rate_limit(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        ip = request.remote_addr or "127.0.0.1"
        now = time.time()
        window_start = now - RATE_LIMIT_WINDOW
        _rate_store[ip] = [t for t in _rate_store[ip] if t > window_start]
        if len(_rate_store[ip]) >= RATE_LIMIT_MAX:
            abort(429)
        _rate_store[ip].append(now)
        return f(*args, **kwargs)
    return decorated

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

def is_locked_out(username):
    if username in _failed_attempts:
        count, first_fail = _failed_attempts[username]
        if count >= LOCKOUT_MAX_ATTEMPTS:
            if datetime.now() - first_fail < LOCKOUT_DURATION:
                return True
            del _failed_attempts[username]
    return False

def record_failed_attempt(username):
    now = datetime.now()
    if username in _failed_attempts:
        count, first_fail = _failed_attempts[username]
        if now - first_fail > LOCKOUT_DURATION:
            _failed_attempts[username] = [1, now]
        else:
            _failed_attempts[username] = [count + 1, first_fail]
    else:
        _failed_attempts[username] = [1, now]

# --- Rutas de Autenticación ---

@app.route("/login", methods=["GET", "POST"])
@rate_limit
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "")
        if not username or not password:
            error = "Credenciales incorrectas. Intenta de nuevo."
        elif is_locked_out(username):
            error = "Cuenta bloqueada temporalmente. Intenta más tarde."
        elif USERS.get(username) and check_password_hash(USERS[username], password):
            session["user"] = username
            session.permanent = True
            _failed_attempts.pop(username, None)
            return redirect(url_for("dashboard"))
        else:
            record_failed_attempt(username)
            error = "Credenciales incorrectas. Intenta de nuevo."
    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

# --- Rutas del Panel Unificado ---

@app.route("/")
@login_required
def dashboard():
    files = list(DICOM_FILES.values())
    files.sort(key=lambda f: f["uploaded_at"], reverse=True)
    return render_template("dashboard.html", files=files)

@app.route("/upload", methods=["POST"])
@login_required
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No se envió ningún archivo"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Archivo vacío"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Formato no soportado"}), 400

    file_id = str(uuid.uuid4())
    original_name = secure_filename(file.filename)
    stem = Path(original_name).stem
    ext = Path(file.filename).suffix.lower()

    metadata = {
        "id": file_id,
        "original_name": original_name,
        "stem": stem,
        "uploaded_by": session["user"],
        "uploaded_at": datetime.now().isoformat(),
        "has_result": False,
        "yolo_ok": False,
        "unet_ok": False
    }

    if is_dicom(file.filename):
        dicom_path = app.config["UPLOAD_FOLDER"] / f"{file_id}.dcm"
        preview_path = app.config["PREVIEW_FOLDER"] / f"{file_id}.png"
        file.save(str(dicom_path))

        try:
            ds = pydicom.dcmread(str(dicom_path))
            metadata.update({
                "is_image": False,
                "ext": ".dcm",
                "patient_name": str(getattr(ds, "PatientName", "Desconocido")),
                "patient_id": str(getattr(ds, "PatientID", "N/A")),
                "study_date": str(getattr(ds, "StudyDate", "N/A")),
                "modality": str(getattr(ds, "Modality", "N/A")),
                "study_description": str(getattr(ds, "StudyDescription", "")),
                "series_description": str(getattr(ds, "SeriesDescription", "")),
                "rows": str(getattr(ds, "Rows", "N/A")),
                "columns": str(getattr(ds, "Columns", "N/A")),
            })
            if hasattr(ds, "pixel_array"):
                dicom_to_preview(dicom_path, preview_path)
                metadata["has_preview"] = True
            else:
                metadata["has_preview"] = False

        except Exception as e:
            dicom_path.unlink(missing_ok=True)
            return jsonify({"error": f"Error al procesar archivo DICOM: {str(e)}"}), 400
    else:
        img_path = app.config["UPLOAD_FOLDER"] / f"{file_id}{ext}"
        file.save(str(img_path))
        metadata.update({
            "is_image": True,
            "ext": ext,
            "has_preview": True
        })

    DICOM_FILES[file_id] = metadata
    return redirect(url_for("view_file", file_id=file_id))

@app.route("/view/<file_id>")
@login_required
def view_file(file_id):
    meta = DICOM_FILES.get(file_id)
    if not meta:
        abort(404)
    return render_template("viewer.html", meta=meta)

@app.route("/analyze/<file_id>")
@login_required
def analyze_file(file_id):
    meta = DICOM_FILES.get(file_id)
    if not meta:
        return jsonify({"error": "Archivo no encontrado"}), 404

    if not meta["is_image"]:
        img_path = app.config["PREVIEW_FOLDER"] / f"{file_id}.png"
    else:
        img_path = app.config["UPLOAD_FOLDER"] / f"{file_id}{meta['ext']}"

    if not img_path.exists():
        return jsonify({"error": "Imagen base no encontrada"}), 404

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return jsonify({"error": "Error al leer la imagen"}), 400

    # YOLOv11 Inferencia
    yolo_ok = False
    try:
        yolo = get_yolo()
        if yolo:
            yolo_results = yolo(img_bgr, imgsz=640, conf=0.25, verbose=False)[0]
            yolo_overlay = img_bgr.copy()
            if yolo_results.masks is not None:
                for j, mask_data in enumerate(yolo_results.masks.data):
                    cls_id = int(yolo_results.boxes.cls[j])
                    color = (0, 0, 255) if cls_id == 0 else (0, 255, 0)
                    mask_np = (mask_data.cpu().numpy() * 255).astype(np.uint8)
                    mask_resized = cv2.resize(mask_np, (img_bgr.shape[1], img_bgr.shape[0])) > 0
                    overlay = np.zeros_like(img_bgr, dtype=np.uint8)
                    overlay[mask_resized] = color
                    yolo_overlay = cv2.addWeighted(yolo_overlay, 1, overlay, 0.4, 0)
                    cnts, _ = cv2.findContours(mask_resized.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(yolo_overlay, cnts, -1, color, 2)
            yolo_out = app.config["OUTPUT_FOLDER"] / f"{file_id}_yolo.png"
            cv2.imwrite(str(yolo_out), yolo_overlay)
            yolo_ok = True
    except Exception as e:
        print(f"[IA] Error en YOLO: {e}")

    # U-Net Inferencia
    unet_ok = False
    try:
        unet_seg, _ = get_unet()
        if unet_seg:
            unet_pred = unet_seg.predict(img_bgr)
            unet_overlay = img_bgr.copy()
            for cls_val, color in [(1, (0, 0, 255)), (2, (0, 255, 0))]:
                mask_cls = (unet_pred == cls_val)
                overlay = np.zeros_like(img_bgr, dtype=np.uint8)
                overlay[mask_cls] = color
                unet_overlay = cv2.addWeighted(unet_overlay, 1, overlay, 0.4, 0)
                cnts, _ = cv2.findContours(mask_cls.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(unet_overlay, cnts, -1, color, 2)
            unet_out = app.config["OUTPUT_FOLDER"] / f"{file_id}_unet.png"
            cv2.imwrite(str(unet_out), unet_overlay)
            unet_ok = True
    except Exception as e:
        print(f"[IA] Error en U-Net: {e}")

    # Mosaico de comparación
    try:
        h_img = img_bgr.shape[0]
        cmp_panels = []
        orig_small = cv2.resize(img_bgr, (256, int(256 * h_img / img_bgr.shape[1])))
        cmp_panels.append(orig_small)
        target_h = orig_small.shape[0]

        for ok_flag, out_path, label in [
            (yolo_ok, app.config["OUTPUT_FOLDER"] / f"{file_id}_yolo.png", "YOLO: N/A"),
            (unet_ok, app.config["OUTPUT_FOLDER"] / f"{file_id}_unet.png", "U-Net: N/A"),
        ]:
            if ok_flag and out_path.exists():
                panel = cv2.imread(str(out_path))
                panel = cv2.resize(panel, (256, int(256 * panel.shape[0] / panel.shape[1])))
            else:
                panel = np.full((target_h, 256, 3), 64, dtype=np.uint8)
                cv2.putText(panel, label, (20, target_h // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cmp_panels.append(panel)

        max_h = max(p.shape[0] for p in cmp_panels)
        panels_padded = []
        for p in cmp_panels:
            if p.shape[0] < max_h:
                pad = np.zeros((max_h - p.shape[0], p.shape[1], 3), dtype=np.uint8)
                p = np.vstack([p, pad])
            panels_padded.append(p)
        
        comparison = np.hstack(panels_padded)
        for i, label in enumerate(["Original", "YOLOv11", "U-Net"]):
            x = i * 256 + 10
            cv2.putText(comparison, label, (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        
        cmp_out = app.config["OUTPUT_FOLDER"] / f"{file_id}_comparison.png"
        cv2.imwrite(str(cmp_out), comparison)
    except Exception as e:
        print(f"[IA] Error en mosaico: {e}")

    meta["has_result"] = True
    meta["yolo_ok"] = yolo_ok
    meta["unet_ok"] = unet_ok
    return redirect(url_for("view_file", file_id=file_id))

@app.route("/media/<file_id>/<img_type>")
@login_required
def serve_media(file_id, img_type):
    meta = DICOM_FILES.get(file_id)
    if not meta:
        abort(404)

    if img_type == "original":
        if meta["is_image"]:
            return send_from_directory(str(app.config["UPLOAD_FOLDER"]), f"{file_id}{meta['ext']}")
        else:
            return send_from_directory(str(app.config["PREVIEW_FOLDER"]), f"{file_id}.png")
            
    elif img_type in ("yolo", "unet", "comparison"):
        path = app.config["OUTPUT_FOLDER"] / f"{file_id}_{img_type}.png"
        if path.exists():
            return send_file(str(path), mimetype="image/png")
    
    abort(404)

@app.route("/download/<file_id>")
@login_required
def download_file(file_id):
    meta = DICOM_FILES.get(file_id)
    if not meta:
        abort(404)
    return send_from_directory(
        str(app.config["UPLOAD_FOLDER"]),
        f"{file_id}{meta['ext']}",
        download_name=meta["original_name"],
    )

@app.route("/delete/<file_id>", methods=["POST"])
@login_required
def delete_file(file_id):
    meta = DICOM_FILES.get(file_id)
    if not meta:
        abort(404)
        
    (app.config["UPLOAD_FOLDER"] / f"{file_id}{meta['ext']}").unlink(missing_ok=True)
    (app.config["PREVIEW_FOLDER"] / f"{file_id}.png").unlink(missing_ok=True)
    for suffix in ("yolo", "unet", "comparison"):
        (app.config["OUTPUT_FOLDER"] / f"{file_id}_{suffix}.png").unlink(missing_ok=True)
        
    del DICOM_FILES[file_id]
    return redirect(url_for("dashboard"))

if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG") == "1"
    app.run(debug=debug, host="0.0.0.0", port=5000)
