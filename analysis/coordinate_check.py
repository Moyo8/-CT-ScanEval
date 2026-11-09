"""Diagnostic (not a grader): did a model find the nodule but report its
position in the wrong coordinate frame?

For every positive detection answer it reports the distance from the given
point to the radiologists' outline as given, and again after re-scaling y (or
x and y) from a 0-1000 normalised frame to the 512-pixel grid the prompt asks
for. Grading never uses these re-scaled numbers; they only explain failures.

    python analysis/coordinate_check.py results/<run_id>
"""
import json
import sys
from pathlib import Path

from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from graders.grader import load_masks, load_reference  # noqa: E402


def dist_mm(dist, x, y):
    xi, yi = int(round(x)), int(round(y))
    if 0 <= xi < dist.shape[1] and 0 <= yi < dist.shape[0]:
        return round(float(dist[yi, xi]), 1)
    return None  # outside the image


def main(run_dir):
    run_dir = Path(run_dir)
    print(f"{'task':<30}{'trial':>6}  {'given (x,y)':>14}  {'mm as given':>11}  {'mm, y*512/1000':>14}  {'mm, x,y*512/1000':>16}")
    for line in (run_dir / "results.jsonl").read_text().splitlines():
        r = json.loads(line)
        ans = r["details"].get("answer") or {}
        if not ans.get("nodule_present") or ans.get("x") is None or ans.get("y") is None:
            continue
        ref = load_reference(r["task_id"])
        if ref["grader"] != "find" or not ref["answer"]["nodule_present"]:
            continue
        union = load_masks(r["task_id"])["union"]
        dist = ndimage.distance_transform_edt(~union) * ref["pixel_spacing_mm"]
        x, y = float(ans["x"]), float(ans["y"])
        s = 512 / 1000
        print(f"{r['task_id']:<30}{r['trial']:>6}  {str((ans['x'], ans['y'])):>14}  {str(dist_mm(dist, x, y)):>11}  "
              f"{str(dist_mm(dist, x, y * s)):>14}  {str(dist_mm(dist, x * s, y * s)):>16}")


if __name__ == "__main__":
    main(sys.argv[1])
