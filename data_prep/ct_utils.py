"""Shared helpers for turning LIDC-IDRI scans + annotations into eval cases."""
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "LIDC-IDRI"

# Standard lung window: width 1500 HU, level -600 HU.
LUNG_WINDOW = (-1350.0, 150.0)


def configure_pylidc():
    """Point pylidc at our downloaded DICOM folder (writes ~/.pylidcrc)."""
    cfg = Path(os.path.expanduser("~/.pylidcrc"))
    cfg.write_text(f"[dicom]\npath = {RAW_DIR}\nwarn = False\n")


def load_scan(patient_id):
    from lidc_compat import pl

    configure_pylidc()
    scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).all()
    scans = [s for s in scans if 1.0 <= s.slice_thickness <= 2.5]
    if len(scans) != 1:
        raise ValueError(f"expected one thin-slice scan for {patient_id}, got {len(scans)}")
    return scans[0]


def load_volume_hu(scan):
    """Return the CT volume in Hounsfield units, indexed [row, col, slice].

    Also checks the in-plane orientation is the standard axial one
    (ImageOrientationPatient = 1,0,0,0,1,0), i.e. the patient's RIGHT is on
    the LEFT of the image (radiological convention), which the task prompts state.
    """
    images = scan.load_all_dicom_images(verbose=False)
    iop = [round(float(v)) for v in images[0].ImageOrientationPatient]
    if iop != [1, 0, 0, 0, 1, 0]:
        raise ValueError(f"{scan.patient_id}: non-standard orientation {iop}")
    vol = np.stack(
        [
            img.pixel_array.astype(np.float32) * float(img.RescaleSlope)
            + float(img.RescaleIntercept)
            for img in images
        ],
        axis=-1,
    )
    zs = [float(img.ImagePositionPatient[2]) for img in images]
    return vol, zs


def to_png_array(hu_slice, window=LUNG_WINDOW):
    lo, hi = window
    arr = np.clip((hu_slice - lo) / (hi - lo), 0, 1)
    return (arr * 255).round().astype(np.uint8)


def save_png(hu_slice, path):
    Image.fromarray(to_png_array(hu_slice)).save(path, optimize=True)


def annotation_slice_mask(ann, shape, k):
    """Boolean mask (full image size) of one radiologist's outline on slice k."""
    mask = np.zeros(shape[:2], dtype=bool)
    ann_mask = ann.boolean_mask()
    bbox = ann.bbox()
    if not (bbox[2].start <= k < bbox[2].stop):
        return mask
    mask[bbox[0], bbox[1]] = ann_mask[:, :, k - bbox[2].start]
    return mask


def max_feret_mm(mask, spacing):
    """Longest in-plane distance across a 2D mask, in mm (pixel-edge corrected)."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0.0
    # Use pixel corners so a 1-pixel object measures 1 pixel, not 0.
    pts = np.concatenate(
        [np.c_[xs + dx, ys + dy] for dx in (0, 1) for dy in (0, 1)]
    ).astype(float)
    from scipy.spatial import ConvexHull

    try:
        hull = pts[ConvexHull(pts).vertices]
    except Exception:
        hull = pts
    d = np.sqrt(((hull[:, None, :] - hull[None, :, :]) ** 2).sum(-1)).max()
    return float(d * spacing)


def body_and_lung_masks(hu_slice):
    """Rough body and (hole-filled) lung masks for one axial slice."""
    body = hu_slice > -500
    body = ndimage.binary_opening(body, iterations=2)
    lab, n = ndimage.label(body)
    if n:
        sizes = ndimage.sum(body, lab, range(1, n + 1))
        body = lab == (1 + int(np.argmax(sizes)))
    body = ndimage.binary_fill_holes(body)
    air = (hu_slice < -400) & body
    lab, n = ndimage.label(air)
    lungs = np.zeros_like(air)
    if n:
        sizes = ndimage.sum(air, lab, range(1, n + 1))
        for idx in np.argsort(sizes)[::-1][:2]:
            if sizes[idx] > 2000:
                lungs |= lab == idx + 1
    lungs = ndimage.binary_closing(lungs, iterations=3)
    lungs = ndimage.binary_fill_holes(lungs)
    return body, lungs


def patient_side(mask, hu_slice):
    """'right' or 'left' (patient side) for a nodule mask on an axial slice."""
    body, _ = body_and_lung_masks(hu_slice)
    body_center_x = np.nonzero(body)[1].mean()
    nodule_x = np.nonzero(mask)[1].mean()
    # Radiological convention: image left = patient's right.
    return "right" if nodule_x < body_center_x else "left"


def pleural_contact_fraction(mask, hu_slice):
    """Fraction of the ring just outside the nodule that is chest wall /
    mediastinum (soft tissue outside the hole-filled lung)."""
    _, lungs = body_and_lung_masks(hu_slice)
    ring = ndimage.binary_dilation(mask, iterations=2) & ~mask
    if ring.sum() == 0:
        return 0.0
    outside = ring & ~lungs & (hu_slice > -400)
    return float(outside.sum() / ring.sum())
