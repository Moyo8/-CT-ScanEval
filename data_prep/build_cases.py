"""Build the CT-ScanEval tasks from downloaded LIDC-IDRI scans.

For every task this writes:
    tasks/<task_id>/task.json         prompt + image labels only (exactly what the model sees)
    tasks/<task_id>/images/*.png      5 consecutive axial slices (lung window)
    reference/<task_id>/reference.json  correct answer + grading parameters (hidden)
    reference/<task_id>/masks.npz       radiologist outlines on the centre slice (hidden)
    (deliberately wrong answers for the grader tests live in tests/test_graders.py)

Run after data_prep/download_lidc.py:
    python data_prep/build_cases.py
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ct_utils import (  # noqa: E402
    ROOT,
    annotation_slice_mask,
    body_and_lung_masks,
    load_scan,
    load_volume_hu,
    max_feret_mm,
    patient_side,
    to_png_array,
)
from lidc_compat import consensus  # noqa: E402

TASKS_DIR = ROOT / "tasks"
REF_DIR = ROOT / "reference"
N_SLICES = 5          # images per task
TARGET_SPACING = 2.5  # mm between the images we show
LOCATION_TOL_MM = 4.0
SIZE_TOL_MM = 1.5

# ---------------------------------------------------------------- prompts --

PREAMBLE = """You are taking part in a research evaluation of chest CT interpretation. This is not clinical use.

You are given {n} consecutive axial images from one non-contrast chest CT scan (lung window: width 1500 HU, level -600 HU).
- The images are ordered from superior (Image 1, toward the head) to inferior (Image {n}, toward the feet). Adjacent images are {spacing:.1f} mm apart. Image {c} is the CENTRE image.
- Orientation follows the standard radiological convention: the patient's RIGHT side is on the LEFT side of the image, and anterior (front of the chest) is at the top.
- Every image is {w} x {h} pixels. Pixel spacing is {ps:.3f} mm per pixel in both directions.
- Pixel coordinates: x is the column (0 at the left edge, increasing to the right) and y is the row (0 at the top edge, increasing downward), in this {w} x {h} pixel grid.

Definition used in this evaluation (based on the LIDC-IDRI reading protocol): a pulmonary nodule is a focal, roughly round or oval SOLID opacity (soft-tissue density, as bright as a blood vessel on this window) within the lung, measuring at least 3 mm and at most 30 mm. It may touch the chest wall or the mediastinum. Hazy ground-glass opacities do not count as nodules in this evaluation. A blood vessel seen in cross-section is NOT a nodule: across adjacent images a vessel continues as a tubular structure that shifts position or branches, whereas a nodule appears and disappears over a few images."""

FIND_QUESTION = """Question: Does the CENTRE image (Image {c}) contain a pulmonary nodule?
If yes, give the pixel coordinates of the centre of the nodule on Image {c} (if there is more than one, report the largest) and say which of the patient's lungs it is in.

Reply with only a JSON object in exactly this format:
{{"nodule_present": true or false, "x": integer or null, "y": integer or null, "patient_side": "right" or "left" or null}}
Use null for x, y and patient_side when nodule_present is false."""

CHOICE_QUESTION = """Three candidate structures on the CENTRE image (Image {c}) are marked with yellow circles labelled A, B and C. Each circle is drawn at the same position on every image so you can follow what is inside it from image to image. The circles are larger than the structures inside them.

Exactly one of the three candidates is a pulmonary nodule. The other two are blood vessels seen in cross-section.

Question: Which candidate is the pulmonary nodule?

Reply with only a JSON object in exactly this format:
{{"nodule_label": "A" or "B" or "C"}}"""

COUNT_QUESTION = """Question: How many distinct pulmonary nodules are visible on the CENTRE image (Image {c})? Give the pixel coordinates of the centre of each one on Image {c}.

Reply with only a JSON object in exactly this format:
{{"count": integer, "nodules": [{{"x": integer, "y": integer}}, ...]}}
The "nodules" list must have exactly "count" entries."""

MEASURE_QUESTION = """One pulmonary nodule on the CENTRE image (Image {c}) is marked with a yellow circle centred at pixel (x={mx}, y={my}). The circle is drawn on every image and is larger than the nodule.

Question: Measure the nodule's longest in-plane diameter on Image {c}, in millimetres, as a radiologist would with electronic calipers (outer edge to outer edge, along the longest axis, in the plane of the image). Use the pixel spacing given above.

Reply with only a JSON object in exactly this format:
{{"long_axis_mm": number}}"""

# ----------------------------------------------------------------- tasks ---

TASKS = [
    dict(
        id="t01_find_large_nodule",
        kind="find_positive",
        patient="LIDC-IDRI-0294",
        group=1,
        expect=dict(n=4, diam=(19, 21)),
        title="Detect and locate a large (20 mm) solid nodule",
        why_hard="Near-control task: a 20 mm solid nodule in the left lung that abuts the mediastinal border. Shows whether the model can do the basic job at all and whether it follows the radiological left/right convention. The only difficulty is telling the nodule from the mediastinal structures it touches.",
    ),
    dict(
        id="t02_find_small_nodule",
        kind="find_positive",
        patient="LIDC-IDRI-0803",
        group=0,
        expect=dict(n=4, diam=(5, 6)),
        title="Detect and locate a small (5 mm) solid nodule",
        why_hard="A 5 mm nodule is about 8 pixels across on a 512-pixel image and looks like the many vessel cross-sections around it. Telling them apart takes comparing adjacent slices. Nodules of this size are the most often missed in screening.",
    ),
    dict(
        id="t03_find_juxtapleural_nodule",
        kind="find_positive",
        patient="LIDC-IDRI-0133",
        group=1,
        expect=dict(n=4, diam=(16, 18)),
        title="Detect and locate a juxtapleural nodule (attached to the chest wall)",
        why_hard="The nodule touches the pleura, so it has no dark lung rim on one side and merges with the chest wall soft tissue. Juxtapleural nodules are a known blind spot for both radiologists and CAD systems.",
    ),
    dict(
        id="t04_no_nodule_hilum",
        kind="find_negative",
        patient="LIDC-IDRI-0963",
        center_k=300,
        title="Correctly report no nodule at the level of the hila",
        why_hard="Many vessels and bronchi in cross-section near the hila look like round opacities on a single image. The correct answer is 'no nodule', and a model that over-calls vessels fails.",
    ),
    dict(
        id="t05_no_nodule_lower_lobes",
        kind="find_negative",
        patient="LIDC-IDRI-0803",
        center_k=95,
        title="Correctly report no nodule in the lower lobes",
        why_hard="Lower-lobe vessels run up and down, so on axial images they appear as dozens of small round dots, which are classic nodule mimics.",
    ),
    dict(
        id="t06_nodule_vs_vessels",
        kind="choice",
        patient="LIDC-IDRI-0067",
        group=1,
        expect=dict(n=4, diam=(7, 9)),
        title="Pick the true nodule among three marked candidates (two are vessels)",
        why_hard="On the centre image the nodule and the two vessel cross-sections look alike. Only following each structure through the adjacent images separates them: a vessel persists and moves, the nodule ends.",
    ),
    dict(
        id="t07_count_nodules",
        kind="count",
        patient="LIDC-IDRI-0121",
        groups=[0, 1],
        title="Count and locate two nodules on the same image",
        why_hard="Two nodules in the same lung on one image: a 21 mm juxtapleural nodule and an 8 mm nodule. A model that stops after the obvious one undercounts.",
    ),
    dict(
        id="t08_measure_nodule",
        kind="measure",
        patient="LIDC-IDRI-0067",
        group=0,
        expect=dict(n=4, diam=(11, 12.5)),
        title="Measure a marked 11 mm nodule in millimetres",
        why_hard="The size sets screening management: Lung-RADS changes category at 6, 8 and 15 mm. The model must turn pixels into millimetres using the stated pixel spacing rather than guess a 'typical' size.",
    ),
]

# --------------------------------------------------------------- helpers ---


def stack_indices(scan, center_k, n_slices):
    step = max(1, int(round(TARGET_SPACING / scan.slice_thickness)))
    half = n_slices // 2
    # pylidc volumes go inferior -> superior (ascending z); show superior first.
    ks = [center_k + step * i for i in range(half, -half - 1, -1)]
    return ks, step * scan.slice_thickness


def all_annotation_slices(scan):
    out = {}
    for ann in scan.annotations:
        b = ann.bbox()
        for k in range(b[2].start, b[2].stop):
            out.setdefault(k, set()).add(ann.id)
    return out


def group_center_slice(group, shape):
    cmask, bbox, _ = consensus(group, clevel=0.5)
    areas = cmask.sum(axis=(0, 1))
    kk = int(np.argmax(areas))
    k = bbox[2].start + kk
    full = np.zeros(shape[:2], dtype=bool)
    full[bbox[0], bbox[1]] = cmask[:, :, kk]
    return k, full


def group_masks_on_slice(group, shape, k):
    cmask, bbox, _ = consensus(group, clevel=0.5)
    full = np.zeros(shape[:2], dtype=bool)
    if bbox[2].start <= k < bbox[2].stop:
        full[bbox[0], bbox[1]] = cmask[:, :, k - bbox[2].start]
    readers = [annotation_slice_mask(a, shape, k) for a in group]
    return full, readers


def centroid(mask):
    ys, xs = np.nonzero(mask)
    return int(round(xs.mean())), int(round(ys.mean()))


def draw_markers(gray, markers, radius):
    img = Image.fromarray(gray).convert("RGB")
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=20)
    for label, (x, y) in markers:
        d.ellipse([x - radius, y - radius, x + radius, y + radius], outline=(255, 220, 0), width=2)
        if label:
            d.text((x + radius + 3, y - radius - 18), label, fill=(255, 220, 0), font=font)
    return img


def save_images(task_dir, vol, ks, markers=None, radius=0):
    img_dir = task_dir / "images"
    img_dir.mkdir(parents=True)
    names = []
    for i, k in enumerate(ks, start=1):
        gray = to_png_array(vol[:, :, k])
        img = draw_markers(gray, markers, radius) if markers else Image.fromarray(gray)
        name = f"image_{i}.png"
        img.save(img_dir / name, optimize=True)
        names.append(f"images/{name}")
    return names


def vessel_candidates(vol, k, ks, exclude_mask, nodule_area, rng):
    """Round bright blobs on slice k that are vessels: their 3-D connected
    component runs through every image of the stack (a nodule does not)."""
    hu = vol[:, :, k]
    _, lungs = body_and_lung_masks(hu)
    lungs = ndimage.binary_erosion(lungs, iterations=6)  # stay off the pleura
    kmin, kmax = min(ks), max(ks)
    sub = vol[:, :, kmin : kmax + 1] > -500
    lab3, _ = ndimage.label(sub)
    blobs = (hu > -500) & lungs & ~ndimage.binary_dilation(exclude_mask, iterations=25)
    lab2, n2 = ndimage.label(blobs)
    found = []
    for i in range(1, n2 + 1):
        comp = lab2 == i
        area = comp.sum()
        if not (0.5 * nodule_area <= area <= 2.0 * nodule_area):
            continue
        ys, xs = np.nonzero(comp)
        h, w = np.ptp(ys) + 1, np.ptp(xs) + 1
        if max(h, w) / min(h, w) > 1.6:  # roughly round on this image
            continue
        cx, cy = int(round(xs.mean())), int(round(ys.mean()))
        l3 = lab3[cy, cx, k - kmin]
        if l3 == 0:
            continue
        z_present = np.nonzero((lab3 == l3).any(axis=(0, 1)))[0]
        if len(z_present) != (kmax - kmin + 1):
            continue  # does not run through the whole stack -> not clearly a vessel
        found.append(dict(x=cx, y=cy, area=int(area)))
    rng.shuffle(found)
    return found


# ----------------------------------------------------------------- build ---


def build_task(spec, rng):
    scan = load_scan(spec["patient"])
    vol, zs = load_volume_hu(scan)
    shape = vol.shape
    ps = float(scan.pixel_spacing)
    groups = scan.cluster_annotations(verbose=False)
    ann_slices = all_annotation_slices(scan)

    def check_group(g):
        exp = spec["expect"]
        d = np.mean([a.diameter for a in g])
        assert len(g) == exp["n"] and exp["diam"][0] <= d <= exp["diam"][1], (spec["id"], len(g), d)

    kind = spec["kind"]
    ref = dict(task_id=spec["id"], kind=kind)
    masks = {}
    markers, radius = None, 0

    if kind in ("find_positive", "choice", "measure"):
        g = groups[spec["group"]]
        check_group(g)
        k, cmask = group_center_slice(g, shape)
        ks, spacing = stack_indices(scan, k, N_SLICES)
        cons, readers = group_masks_on_slice(g, shape, k)
        target_ids = {a.id for a in g}
        # The centre image must contain no other annotated nodule.
        others = ann_slices.get(k, set()) - target_ids
        assert not others, (spec["id"], "other annotations on centre slice", others)
        x, y = centroid(cons)
        union = np.any(readers, axis=0)
        masks = dict(consensus=cons, union=union, **{f"reader_{i}": m for i, m in enumerate(readers)})
        reader_long_axes = [round(max_feret_mm(m, ps), 2) for m in readers]
        ref.update(
            nodule=dict(
                x=x,
                y=y,
                patient_side=patient_side(cons, vol[:, :, k]),
                reader_long_axis_mm=reader_long_axes,
                consensus_long_axis_mm=round(max_feret_mm(cons, ps), 2),
                lidc_mean_diameter_mm=round(float(np.mean([a.diameter for a in g])), 2),
                n_readers=len(g),
                subtlety_median=float(np.median([a.subtlety for a in g])),
                texture_median=float(np.median([a.texture for a in g])),
            )
        )
        if kind == "find_positive":
            ref["answer"] = dict(nodule_present=True, x=x, y=y, patient_side=ref["nodule"]["patient_side"])
            ref["grading"] = dict(location_tolerance_mm=LOCATION_TOL_MM)
        elif kind == "measure":
            radius = 22
            markers = [("", (x, y))]
            ref["answer"] = dict(long_axis_mm=ref["nodule"]["consensus_long_axis_mm"])
            lo = round(min(reader_long_axes) - SIZE_TOL_MM, 2)
            hi = round(max(reader_long_axes) + SIZE_TOL_MM, 2)
            ref["grading"] = dict(accept_range_mm=[lo, hi], size_tolerance_mm=SIZE_TOL_MM)
        else:  # choice
            nod_area = int(cons.sum())
            exclude = np.zeros(shape[:2], bool)
            for kk in ks:
                for a in scan.annotations:
                    exclude |= annotation_slice_mask(a, shape, kk)
            cands = vessel_candidates(vol, k, ks, exclude, nod_area, rng)
            assert len(cands) >= 2, (spec["id"], "not enough vessel candidates", len(cands))
            vessels = cands[:2]
            labels = ["A", "B", "C"]
            order = rng.permutation(3)
            positions = [(x, y)] + [(v["x"], v["y"]) for v in vessels]
            marker_list = [None] * 3
            for slot, src in zip(order, range(3)):
                marker_list[slot] = (labels[slot], positions[src])
            nodule_label = labels[int(order[0])]
            radius = 14
            markers = marker_list
            ref["candidates"] = {lab: dict(x=p[0], y=p[1]) for lab, p in marker_list}
            ref["vessel_candidates"] = vessels
            ref["answer"] = dict(nodule_label=nodule_label)
            ref["grading"] = {}

    elif kind == "find_negative":
        k = spec["center_k"]
        ks, spacing = stack_indices(scan, k, N_SLICES)
        for kk in range(min(ks) - 4, max(ks) + 5):
            assert kk not in ann_slices, (spec["id"], "annotation near negative stack", kk)
        nearest = min(abs(kk - k) for kk in ann_slices) * scan.slice_thickness
        ref["nearest_annotation_mm"] = round(float(nearest), 1)
        ref["answer"] = dict(nodule_present=False, x=None, y=None, patient_side=None)
        ref["grading"] = {}

    elif kind == "count":
        gs = [groups[i] for i in spec["groups"]]
        for g in gs:
            assert len(g) >= 3
        # centre slice: the one where the smaller nodule is largest while both are present
        common = set.intersection(*[set(range(a.bbox()[2].start, a.bbox()[2].stop)) for g in gs for a in g])
        best_k, best_area = None, -1
        for kk in sorted(common):
            areas = [group_masks_on_slice(g, shape, kk)[0].sum() for g in gs]
            if min(areas) > best_area:
                best_k, best_area = kk, min(areas)
        k = best_k
        ks, spacing = stack_indices(scan, k, N_SLICES)
        target_ids = {a.id for g in gs for a in g}
        others = ann_slices.get(k, set()) - target_ids
        assert not others, (spec["id"], "other annotations on centre slice", others)
        nodules = []
        for i, g in enumerate(gs):
            cons, readers = group_masks_on_slice(g, shape, k)
            x, y = centroid(cons)
            masks[f"nodule_{i}_consensus"] = cons
            masks[f"nodule_{i}_union"] = np.any(readers, axis=0)
            nodules.append(
                dict(
                    x=x,
                    y=y,
                    patient_side=patient_side(cons, vol[:, :, k]),
                    lidc_mean_diameter_mm=round(float(np.mean([a.diameter for a in g])), 2),
                    n_readers=len(g),
                )
            )
        ref["nodules"] = nodules
        ref["answer"] = dict(count=len(nodules), nodules=[dict(x=n["x"], y=n["y"]) for n in nodules])
        ref["grading"] = dict(location_tolerance_mm=LOCATION_TOL_MM)
    else:
        raise ValueError(kind)

    c = N_SLICES // 2 + 1
    fmt = dict(n=N_SLICES, spacing=spacing, c=c, w=shape[1], h=shape[0], ps=ps)
    question = {
        "find_positive": FIND_QUESTION,
        "find_negative": FIND_QUESTION,
        "choice": CHOICE_QUESTION,
        "count": COUNT_QUESTION,
        "measure": MEASURE_QUESTION,
    }[kind]
    if kind == "measure":
        fmt.update(mx=ref["nodule"]["x"], my=ref["nodule"]["y"])
    prompt = PREAMBLE.format(**fmt) + "\n\n" + question.format(**fmt)

    task_dir = TASKS_DIR / spec["id"]
    ref_dir = REF_DIR / spec["id"]
    for d in (task_dir, ref_dir):
        if d.exists():
            shutil.rmtree(d)
    ref_dir.mkdir(parents=True)
    images = save_images(task_dir, vol, ks, markers, radius)

    grader_kind = "find" if kind.startswith("find") else kind
    task = dict(
        id=spec["id"],
        grader=grader_kind,
        image_labels=[
            f"Image {i} of {N_SLICES}" + (" (CENTRE image)" if i == c else "")
            + (" - most superior" if i == 1 else "")
            + (" - most inferior" if i == N_SLICES else "")
            for i in range(1, N_SLICES + 1)
        ],
        images=images,
        prompt=prompt,
        image_size=[shape[1], shape[0]],
        pixel_spacing_mm=round(ps, 4),
        slice_gap_mm=round(spacing, 3),
    )
    (task_dir / "task.json").write_text(json.dumps(task, indent=2) + "\n")

    ref["grader"] = grader_kind
    ref["title"] = spec["title"]
    ref["why_hard"] = spec["why_hard"]
    ref["pixel_spacing_mm"] = round(ps, 4)
    ref["source"] = dict(
        collection="LIDC-IDRI",
        patient_id=scan.patient_id,
        study_instance_uid=scan.study_instance_uid,
        series_instance_uid=scan.series_instance_uid,
        slice_thickness_mm=float(scan.slice_thickness),
        centre_slice_index=int(k),
        slice_indices_superior_to_inferior=[int(v) for v in ks],
        centre_slice_z_mm=round(float(zs[k]), 2),
    )
    (ref_dir / "reference.json").write_text(json.dumps(ref, indent=2) + "\n")
    if masks:
        np.savez_compressed(ref_dir / "masks.npz", **{n: np.packbits(m) for n, m in masks.items()}, shape=np.array(shape[:2]))
    print(f"built {spec['id']}: centre slice {k}, images {ks}")
    return ref


def main():
    only = sys.argv[1:]
    for idx, spec in enumerate(TASKS):
        if only and spec["id"] not in only:
            continue
        build_task(spec, np.random.default_rng([20261107, idx]))


if __name__ == "__main__":
    main()
