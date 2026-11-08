"""Grader validation: every reference answer passes, deliberately wrong
answers fail, near-misses inside the stated tolerance still pass, and the
model-facing task files never contain the hidden answer.

    python -m pytest -q tests/
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graders.grader import grade, load_masks, load_reference  # noqa: E402

TASK_IDS = sorted(p.name for p in (ROOT / "tasks").iterdir() if (p / "task.json").exists())
FIND_POS = ["t01_find_large_nodule", "t02_find_small_nodule", "t03_find_juxtapleural_nodule"]
FIND_NEG = ["t04_no_nodule_hilum", "t05_no_nodule_lower_lobes"]


def j(obj):
    return json.dumps(obj)


def ref(task_id):
    return load_reference(task_id)


def test_eight_tasks_present():
    assert len(TASK_IDS) == 8


# ------------------------------------------------ reference solutions pass --

@pytest.mark.parametrize("task_id", TASK_IDS)
def test_reference_passes(task_id):
    g = grade(task_id, j(ref(task_id)["answer"]))
    assert g["passed"] and g["score"] == 1.0, g


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_reference_passes_inside_prose_and_code_fence(task_id):
    text = "Looking at image 3 carefully...\n\n```json\n" + json.dumps(ref(task_id)["answer"], indent=2) + "\n```\n"
    assert grade(task_id, text)["passed"]


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_unparseable_reply_fails(task_id):
    for text in ["", "I cannot determine this from the images.", "{not json}", None]:
        g = grade(task_id, text)
        assert not g["passed"] and g["error"]


# ----------------------------------------------------- find (positive) ----

@pytest.mark.parametrize("task_id", FIND_POS)
def test_find_wrong_answers_fail(task_id):
    a = ref(task_id)["answer"]
    other = "right" if a["patient_side"] == "left" else "left"
    wrong = {
        "missed": {"nodule_present": False, "x": None, "y": None, "patient_side": None},
        "mirror_location": {**a, "x": 511 - a["x"], "patient_side": other},
        "wrong_side_only": {**a, "patient_side": other},
        "image_side_word": {**a, "patient_side": "image-left"},
        "no_coordinates": {**a, "x": None, "y": None},
        "out_of_image": {**a, "x": 900, "y": -5},
        "normalised_0_1000": {**a, "x": round(a["x"] / 512 * 1000), "y": round(a["y"] / 512 * 1000)},
    }
    for name, ans in wrong.items():
        g = grade(task_id, j(ans))
        assert not g["passed"], (name, g)


@pytest.mark.parametrize("task_id", FIND_POS)
def test_find_partial_credit(task_id):
    a = ref(task_id)["answer"]
    other = "right" if a["patient_side"] == "left" else "left"
    assert grade(task_id, j({**a, "patient_side": other}))["score"] == pytest.approx(2 / 3)
    assert grade(task_id, j({"nodule_present": False}))["score"] == 0.0


@pytest.mark.parametrize("task_id", FIND_POS)
def test_find_tolerance_boundary(task_id):
    """A point just inside the 4 mm tolerance passes; one well outside fails."""
    r = ref(task_id)
    union = load_masks(task_id)["union"]
    ps = r["pixel_spacing_mm"]
    from scipy import ndimage

    dist = ndimage.distance_transform_edt(~union) * ps
    near = np.argwhere((dist > 2.0) & (dist < 3.5))
    far = np.argwhere((dist > 6.0) & (dist < 9.0))
    y, x = near[0]
    assert grade(task_id, j({**r["answer"], "x": int(x), "y": int(y)}))["passed"]
    y, x = far[0]
    assert not grade(task_id, j({**r["answer"], "x": int(x), "y": int(y)}))["passed"]


def test_find_accepts_string_booleans_and_case():
    a = ref("t01_find_large_nodule")["answer"]
    assert grade("t01_find_large_nodule", j({**a, "nodule_present": "true", "patient_side": "LEFT"}))["passed"]


# ----------------------------------------------------- find (negative) ----

@pytest.mark.parametrize("task_id", FIND_NEG)
def test_negative_wrong_answers_fail(task_id):
    for ans in [
        {"nodule_present": True, "x": 256, "y": 256, "patient_side": "right"},
        {"nodule_present": True, "x": None, "y": None, "patient_side": None},
        {"nodule_present": "maybe"},
    ]:
        assert not grade(task_id, j(ans))["passed"], ans


@pytest.mark.parametrize("task_id", FIND_NEG)
def test_negative_reference_with_stray_coordinates_still_passes(task_id):
    # grading is on the outcome (no nodule), not on how tidy the JSON is
    assert grade(task_id, j({"nodule_present": False, "x": 0, "y": 0, "patient_side": "left"}))["passed"]


# ------------------------------------------------------------ choice ------

def test_choice_wrong_labels_fail():
    t = "t06_nodule_vs_vessels"
    right = ref(t)["answer"]["nodule_label"]
    for lab in "ABC":
        if lab != right:
            assert not grade(t, j({"nodule_label": lab}))["passed"]
    for hedge in [f"{right} or A", f"{right}/C", "ABC", ""]:
        assert not grade(t, j({"nodule_label": hedge}))["passed"], hedge
    assert grade(t, j({"nodule_label": right.lower()}))["passed"]


# ------------------------------------------------------------- count ------

def test_count_wrong_answers_fail():
    t = "t07_count_nodules"
    n0, n1 = ref(t)["answer"]["nodules"]
    wrong = {
        "undercount": {"count": 1, "nodules": [n0]},
        "both_points_on_one_nodule": {"count": 2, "nodules": [n0, {"x": n0["x"] + 1, "y": n0["y"]}]},
        "extra_point": {"count": 3, "nodules": [n0, n1, {"x": 100, "y": 300}]},
        "right_count_wrong_places": {"count": 2, "nodules": [{"x": 100, "y": 300}, {"x": 400, "y": 300}]},
        "count_without_points": {"count": 2, "nodules": []},
        "list_mismatch": {"count": 2, "nodules": [n0, n1, n1]},
        "spray_points": {"count": 40, "nodules": [{"x": x, "y": y} for x in range(0, 512, 80) for y in range(0, 512, 80)]},
    }
    for name, ans in wrong.items():
        assert not grade(t, j(ans))["passed"], name


def test_count_order_does_not_matter_and_partial_credit():
    t = "t07_count_nodules"
    n0, n1 = ref(t)["answer"]["nodules"]
    assert grade(t, j({"count": 2, "nodules": [n1, n0]}))["passed"]
    g = grade(t, j({"count": 1, "nodules": [n1]}))
    assert 0 < g["score"] < 1


# ----------------------------------------------------------- measure ------

def test_measure_range():
    t = "t08_measure_nodule"
    lo, hi = ref(t)["grading"]["accept_range_mm"]
    for v in [lo, hi, (lo + hi) / 2, f"{(lo + hi) / 2:.1f} mm"]:
        assert grade(t, j({"long_axis_mm": v}))["passed"], v
    nodule_px = ref(t)["answer"]["long_axis_mm"] / ref(t)["pixel_spacing_mm"]
    for v in [lo - 0.2, hi + 0.2, 5.0, 20.0, round(nodule_px, 1), None, "about 1 cm"]:
        assert not grade(t, j({"long_axis_mm": v}))["passed"], v


# ------------------------------------------------- spec / leakage checks --

@pytest.mark.parametrize("task_id", TASK_IDS)
def test_prompt_states_what_is_graded(task_id):
    task = json.loads((ROOT / "tasks" / task_id / "task.json").read_text())
    p = task["prompt"]
    assert "radiological convention" in p and "Pixel spacing" in p and "x is the column" in p
    assert "JSON" in p
    for key in ref(task_id)["answer"]:
        assert f'"{key}"' in p, key


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_answer_not_leaked_into_task_files(task_id):
    task_dir = ROOT / "tasks" / task_id
    files = {p.name for p in task_dir.rglob("*") if p.is_file()}
    assert files <= {"task.json"} | {f"image_{i}.png" for i in range(1, 6)}, files
    text = (task_dir / "task.json").read_text()
    assert set(json.loads(text)) == {"id", "grader", "image_labels", "images", "prompt",
                                     "image_size", "pixel_spacing_mm", "slice_gap_mm"}
    r = ref(task_id)
    assert r["source"]["patient_id"] not in text
    if r["grader"] == "find" and r["answer"]["nodule_present"]:
        assert f'x={r["answer"]["x"]}' not in text and f'"x": {r["answer"]["x"]}' not in text


def test_find_prompts_identical_for_positive_and_negative_cases():
    """Positive and negative detection tasks ask the exact same question, so the
    prompt gives no hint which kind of case it is."""
    qs = set()
    for t in FIND_POS + FIND_NEG:
        p = json.loads((ROOT / "tasks" / t / "task.json").read_text())["prompt"]
        qs.add(p.split("Question:")[1])
    assert len(qs) == 1
