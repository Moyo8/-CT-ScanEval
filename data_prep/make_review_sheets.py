"""Draw one review sheet per task: the 5 images the model sees, plus the centre
image with the hidden reference overlaid (radiologist outlines, reference
point, accepted zone). Used for the author's manual check of every task.

    python data_prep/make_review_sheets.py   ->  review/<task_id>.png
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from graders.grader import load_masks  # noqa: E402


def main():
    out_dir = ROOT / "review"
    out_dir.mkdir(exist_ok=True)
    for task_dir in sorted((ROOT / "tasks").iterdir()):
        task = json.loads((task_dir / "task.json").read_text())
        ref = json.loads((ROOT / "reference" / task["id"] / "reference.json").read_text())
        masks = load_masks(task["id"])
        imgs = [np.array(Image.open(task_dir / p)) for p in task["images"]]
        fig, axes = plt.subplots(1, 6, figsize=(30, 5.6))
        for i, (ax, img) in enumerate(zip(axes, imgs), start=1):
            ax.imshow(img, cmap="gray")
            ax.set_title(task["image_labels"][i - 1], fontsize=12)
            ax.axis("off")
        ax = axes[5]
        ax.imshow(imgs[2], cmap="gray")
        for name, m in masks.items():
            if "union" in name:
                ax.contour(m, levels=[0.5], colors="cyan", linewidths=0.8)
            if "consensus" in name:
                ax.contour(m, levels=[0.5], colors="red", linewidths=0.8)
        ans = ref["answer"]
        pts = []
        if "x" in ans and ans.get("x") is not None:
            pts = [(ans["x"], ans["y"])]
        if "nodules" in ans:
            pts = [(n["x"], n["y"]) for n in ans["nodules"]]
        if "candidates" in ref:
            lab = ans["nodule_label"]
            pts = [(ref["candidates"][lab]["x"], ref["candidates"][lab]["y"])]
        for x, y in pts:
            ax.plot(x, y, "r+", markersize=14)
        ax.set_title(f"REFERENCE: {json.dumps(ans)}"[:90], fontsize=10)
        ax.axis("off")
        fig.suptitle(f"{task['id']} - {ref['title']}  (red = 50% consensus, cyan = any reader)", fontsize=14)
        plt.tight_layout()
        fig.savefig(out_dir / f"{task['id']}.png", dpi=70)
        plt.close(fig)
        print("wrote", out_dir / f"{task['id']}.png")


if __name__ == "__main__":
    main()
