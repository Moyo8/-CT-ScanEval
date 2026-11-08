"""Code-based graders for CT-ScanEval.

Every grader is deterministic: no LLM judge. Each returns a dict with
    passed   - bool, the task's pass/fail outcome (all checks true)
    score    - float in [0, 1], partial credit (mean of the checks)
    checks   - {check_name: bool}
    details  - numbers behind the checks (distances, errors) for the write-up
    error    - why the answer could not be graded (unparseable JSON, ...)

Graders only read the model's final JSON answer, never its reasoning, so any
approach that produces the right answer passes ("grade the outcome").
"""
import json
import re
from itertools import permutations
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "reference"
TASKS_DIR = ROOT / "tasks"


# ------------------------------------------------------------ loading ----

def load_reference(task_id):
    return json.loads((REF_DIR / task_id / "reference.json").read_text())


def load_masks(task_id):
    path = REF_DIR / task_id / "masks.npz"
    if not path.exists():
        return {}
    data = np.load(path)
    shape = tuple(int(v) for v in data["shape"])
    n = shape[0] * shape[1]
    return {
        k: np.unpackbits(data[k])[:n].reshape(shape).astype(bool)
        for k in data.files
        if k != "shape"
    }


# ------------------------------------------------------------ parsing ----

def _json_candidates(text):
    """Yield every balanced {...} substring, last one first."""
    spans, depth, start = [], 0, None
    in_str, esc = False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                spans.append(text[start : i + 1])
    return reversed(spans)


def parse_answer(text, required_key):
    """Return the last JSON object in the reply that contains required_key."""
    if not isinstance(text, str):
        return None
    cleaned = re.sub(r"```(?:json)?", "", text)
    for cand in _json_candidates(cleaned):
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and required_key in obj:
            return obj
    return None


def _as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in ("true", "yes"):
        return True
    if isinstance(v, str) and v.strip().lower() in ("false", "no"):
        return False
    return None


def _as_number(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, str):
        m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*(mm)?\s*", v)
        return float(m.group(1)) if m else None
    return None


def _as_point(obj, shape):
    x, y = _as_number(obj.get("x")), _as_number(obj.get("y"))
    if x is None or y is None:
        return None
    if not (0 <= x < shape[1] and 0 <= y < shape[0]):
        return None
    return x, y


def _distance_mm(point, mask, spacing):
    """Distance (mm) from a point to the nearest pixel of mask (0 if inside)."""
    dist = ndimage.distance_transform_edt(~mask) * spacing
    xi = int(round(min(max(point[0], 0), mask.shape[1] - 1)))
    yi = int(round(min(max(point[1], 0), mask.shape[0] - 1)))
    return float(dist[yi, xi])


def _result(checks, details=None, error=None, score=None):
    vals = list(checks.values())
    if score is None:
        score = float(np.mean(vals)) if vals else 0.0
    return dict(
        passed=bool(vals) and all(vals) and error is None,
        score=score,
        checks=checks,
        details=details or {},
        error=error,
    )


# ------------------------------------------------------------ graders ----

def grade_find(ref, masks, text):
    """Detection (+ location + laterality for positive cases)."""
    positive = ref["answer"]["nodule_present"]
    ans = parse_answer(text, "nodule_present")
    present = _as_bool(ans.get("nodule_present")) if ans else None
    if not positive:
        checks = {"correctly_reports_no_nodule": present is False}
        err = None if present is not None else "no valid nodule_present in reply"
        return _result(checks, {"answer": ans}, err)

    checks = {"detected": False, "located": False, "correct_side": False}
    details = {"answer": ans}
    if present is None:
        return _result(checks, details, "no valid nodule_present in reply")
    checks["detected"] = present is True
    if present:
        spacing = ref["pixel_spacing_mm"]
        point = _as_point(ans, masks["union"].shape)
        if point is not None:
            d = _distance_mm(point, masks["union"], spacing)
            details["distance_to_nodule_mm"] = round(d, 2)
            checks["located"] = d <= ref["grading"]["location_tolerance_mm"]
        side = str(ans.get("patient_side", "")).strip().lower()
        checks["correct_side"] = side == ref["answer"]["patient_side"]
    return _result(checks, details)


def grade_choice(ref, masks, text):
    ans = parse_answer(text, "nodule_label")
    label = str(ans.get("nodule_label", "")).strip().upper() if ans else ""
    checks = {"picked_nodule": label == ref["answer"]["nodule_label"]}
    err = None if label in ("A", "B", "C") else "no valid nodule_label (A/B/C) in reply"
    return _result(checks, {"answer": ans}, err)


def grade_count(ref, masks, text):
    ans = parse_answer(text, "count")
    n_ref = ref["answer"]["count"]
    checks = {"count_correct": False, "all_nodules_located": False, "no_extra_points": False}
    details = {"answer": ans}
    if not ans:
        return _result(checks, details, "no valid count in reply")
    count = _as_number(ans.get("count"))
    checks["count_correct"] = count is not None and int(count) == count == n_ref
    shape = masks["nodule_0_union"].shape
    pts = ans.get("nodules") if isinstance(ans.get("nodules"), list) else []
    pts = [_as_point(p, shape) if isinstance(p, dict) else None for p in pts]
    spacing = ref["pixel_spacing_mm"]
    tol = ref["grading"]["location_tolerance_mm"]
    unions = [masks[f"nodule_{i}_union"] for i in range(n_ref)]
    hit = [[p is not None and _distance_mm(p, u, spacing) <= tol for u in unions] for p in pts]
    # best one-to-one matching of predicted points to reference nodules
    best = 0
    for perm in permutations(range(len(pts)), min(len(pts), n_ref)):
        best = max(best, sum(hit[pi][ni] for ni, pi in enumerate(perm)))
    details["nodules_matched"] = best
    checks["all_nodules_located"] = best == n_ref
    false_points = sum(1 for h in hit if not any(h))
    checks["no_extra_points"] = len(pts) == n_ref and false_points == 0
    # partial credit: right count, fraction of nodules found, fraction of points that are real
    precision = (len(pts) - false_points) / len(pts) if pts else 0.0
    score = float(np.mean([checks["count_correct"], best / n_ref, precision]))
    return _result(checks, details, score=score)


def grade_measure(ref, masks, text):
    ans = parse_answer(text, "long_axis_mm")
    value = _as_number(ans.get("long_axis_mm")) if ans else None
    lo, hi = ref["grading"]["accept_range_mm"]
    checks = {"size_within_reader_range": value is not None and lo <= value <= hi}
    details = {"answer": ans, "accept_range_mm": [lo, hi]}
    if value is not None:
        details["abs_error_vs_consensus_mm"] = round(abs(value - ref["answer"]["long_axis_mm"]), 2)
    err = None if value is not None else "no numeric long_axis_mm in reply"
    return _result(checks, details, err)


GRADERS = {
    "find": grade_find,
    "choice": grade_choice,
    "count": grade_count,
    "measure": grade_measure,
}

_CACHE = {}


def grade(task_id, response_text):
    if task_id not in _CACHE:
        _CACHE[task_id] = (load_reference(task_id), load_masks(task_id))
    ref, masks = _CACHE[task_id]
    return GRADERS[ref["grader"]](ref, masks, response_text)
