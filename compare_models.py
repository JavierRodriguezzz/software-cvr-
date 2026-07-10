"""Compare two U-Net weights on a folder of images — prints NUMBERS ONLY.

PRIVACY: this reads images locally and prints only aggregate metrics.  No
image ever leaves your machine, so it is safe to run on confidential data;
you only share the printed summary if you want to.

Two modes, auto-selected:
  * If a sibling ``labels/`` folder exists (ground-truth masks with the same
    file names), it reports Dice per class for each model — the rigorous
    before-vs-after comparison.
  * Otherwise it reports mask-free proxies: how often each model produces a
    plausible two-lip segmentation, and how clean (un-fragmented) each mask
    is.  Useful for OOD images you have not annotated.

Examples:
    # OOD images without masks
    python compare_models.py --before models/unet_fugc_best.pth \\
        --after models/unet_fugc_aug.pth --images "C:/ruta/ood_imgs"

    # dataset test split (has labels -> Dice)
    python compare_models.py --after models/unet_fugc_aug.pth \\
        --images "C:/.../unet_dataset/test/images"
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

from models.unet_model import UNetSegmenter

_CONN8 = np.ones((3, 3), dtype=int)


def _sibling_labels(images_dir: Path):
    sib = images_dir.parent / "labels"
    return sib if sib.is_dir() else None


def _largest_fraction(mask_bool: np.ndarray) -> float:
    """Largest connected component / total area.  1.0 = single clean blob."""
    if not mask_bool.any():
        return 0.0
    labeled, n = ndi.label(mask_bool, structure=_CONN8)
    if n == 0:
        return 0.0
    sizes = ndi.sum(mask_bool, labeled, range(1, n + 1))
    return float(sizes.max() / mask_bool.sum())


def _dice(a: np.ndarray, b: np.ndarray):
    a, b = a > 0, b > 0
    s = int(a.sum()) + int(b.sum())
    return None if s == 0 else 2.0 * int((a & b).sum()) / s


class Acc:
    """Accumulates per-image metrics for one model."""

    def __init__(self):
        self.n = 0
        self.both = 0
        self.d1, self.d2 = [], []           # Dice per class (if GT)
        self.frag1, self.frag2 = [], []     # mask cleanliness
        self.area1, self.area2 = [], []     # area fraction of the image

    def add(self, pred: np.ndarray, gt):
        self.n += 1
        ant, post = (pred == 1), (pred == 2)
        if ant.any() and post.any():
            self.both += 1
        self.area1.append(float(ant.mean()))
        self.area2.append(float(post.mean()))
        self.frag1.append(_largest_fraction(ant))
        self.frag2.append(_largest_fraction(post))
        if gt is not None:
            d1, d2 = _dice(pred == 1, gt == 1), _dice(pred == 2, gt == 2)
            if d1 is not None:
                self.d1.append(d1)
            if d2 is not None:
                self.d2.append(d2)

    @staticmethod
    def _mean(xs):
        return float(np.mean(xs)) if xs else None


def _fmt(x):
    return " n/a" if x is None else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default="models/unet_fugc_best.pth")
    ap.add_argument("--after", default="models/unet_fugc_aug.pth")
    ap.add_argument("--images", required=True, help="folder of .png/.jpg images")
    ap.add_argument("--labels", default="", help="optional explicit labels folder")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    for tag, w in (("--before", args.before), ("--after", args.after)):
        if not Path(w).is_file():
            print(f"No existe el archivo de pesos {tag}: {w}")
            if w.endswith("aug.pth"):
                print("  Sugerencia: primero corre train_unet.py para generar los pesos afinados.")
            return

    files = sorted(glob.glob(f"{args.images}/*.png")) + sorted(glob.glob(f"{args.images}/*.jpg"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        print("No se encontraron imagenes en:", args.images)
        return

    images_dir = Path(args.images)
    labels_dir = Path(args.labels) if args.labels else _sibling_labels(images_dir)
    has_gt = labels_dir is not None and labels_dir.is_dir()

    print(f"Imagenes: {len(files)}  |  ground-truth: "
          f"{'SI (' + str(labels_dir) + ')' if has_gt else 'NO -> metricas proxy'}")
    print("Cargando modelos...")
    seg_before = UNetSegmenter(args.before)
    seg_after = UNetSegmenter(args.after)

    accB, accA = Acc(), Acc()
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        gt = None
        if has_gt:
            g = cv2.imread(str(labels_dir / Path(f).name), cv2.IMREAD_UNCHANGED)
            if g is not None:
                gt = g[..., 0] if g.ndim == 3 else g
        accB.add(seg_before.predict(img), gt)
        accA.add(seg_after.predict(img), gt)

    print()
    print(f"{'metrica':30s}{'ANTES':>10s}{'DESPUES':>10s}")
    print("-" * 50)
    if has_gt:
        print(f"{'Dice anterior':30s}{_fmt(Acc._mean(accB.d1)):>10s}{_fmt(Acc._mean(accA.d1)):>10s}")
        print(f"{'Dice posterior':30s}{_fmt(Acc._mean(accB.d2)):>10s}{_fmt(Acc._mean(accA.d2)):>10s}")
        bm = Acc._mean(accB.d1 + accB.d2)
        am = Acc._mean(accA.d1 + accA.d2)
        print(f"{'Dice medio':30s}{_fmt(bm):>10s}{_fmt(am):>10s}")
        if len(accB.d1) == len(accA.d1) and accB.d1:
            imp = sum(1 for a, b in zip(accA.d1, accB.d1) if a - b > 0.01)
            wor = sum(1 for a, b in zip(accA.d1, accB.d1) if b - a > 0.01)
            print(f"  (anterior) mejoraron: {imp}   empeoraron: {wor}   igual: {len(accB.d1)-imp-wor}")
    print(f"{'% ambos labios detectados':30s}"
          f"{100*accB.both/max(1,accB.n):>9.0f}%{100*accA.both/max(1,accA.n):>9.0f}%")
    print(f"{'Limpieza masc. ant (1=lisa)':30s}{Acc._mean(accB.frag1):>10.3f}{Acc._mean(accA.frag1):>10.3f}")
    print(f"{'Limpieza masc. post (1=lisa)':30s}{Acc._mean(accB.frag2):>10.3f}{Acc._mean(accA.frag2):>10.3f}")
    print(f"{'Area ant (frac imagen)':30s}{Acc._mean(accB.area1):>10.4f}{Acc._mean(accA.area1):>10.4f}")
    print(f"{'Area post (frac imagen)':30s}{Acc._mean(accB.area2):>10.4f}{Acc._mean(accA.area2):>10.4f}")
    print()
    if has_gt:
        print("Guia: Dice mas alto = mejor. Si DESPUES >= ANTES, el fine-tune no dano la calidad.")
    else:
        print("Sin ground-truth no hay Dice. Guia: en tus imagenes OOD, '% ambos labios' y")
        print("'limpieza' MAS ALTOS en DESPUES = segmentacion mas robusta y limpia (mejora OOD).")


if __name__ == "__main__":
    main()
