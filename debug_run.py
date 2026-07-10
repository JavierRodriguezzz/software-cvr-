import sys, cv2
from pathlib import Path
sys.path.insert(0, '.')
from core.pipeline_factory import run_pipeline

img = cv2.imread(r"ruta\a\tu\imagen.png")   # <-- cambia por tu archivo
ctx = run_pipeline(img, Path('.'), {'file_id': 'debug_test'})
report = ctx.pipeline_report
print("=== PIPELINE REPORT ===")
for name, ms in report.stage_times_ms.items():
    print(f"  {name}: {ms} ms")
print(f"  TOTAL: {report.total_ms} ms")
print(f"  Errors: {ctx.errors if ctx.errors else 'none'}")