"""Build the CT-ScanEval walkthrough video (slides + synthetic narration).

Needs: ffmpeg, espeak-ng with the mbrola us2 voice (apt-get install ffmpeg espeak-ng mbrola-us2)
    python video/make_video.py   ->  video/CT-ScanEval_video.mp4
"""
import json
import subprocess
import sys
import wave
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from graders.grader import load_masks, load_reference  # noqa: E402

OUT = ROOT / "video"
BUILD = OUT / "build"
BG, FG, ACC, MUTED = "#0d1321", "#f1f3f8", "#ffd23f", "#9aa5b8"
GOOD, BAD, BLUE = "#3ddc97", "#ff5c5c", "#4ea8ff"
VOICE, SPEED = "mb-us2", "150"


def img(task, i):
    return np.array(Image.open(ROOT / "tasks" / task / "images" / f"image_{i}.png"))


def new_slide(title=None):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
    if title:
        fig.text(0.05, 0.9, title, color=FG, fontsize=30, weight="bold")
        fig.add_artist(Rectangle((0.05, 0.875), 0.08, 0.006, color=ACC, transform=fig.transFigure))
    return fig


def text(fig, x, y, s, size=19, color=FG, **kw):
    fig.text(x, y, s, color=color, fontsize=size, va="top", linespacing=1.5, **kw)


def ct_axes(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_facecolor(BG)
    ax.axis("off")
    return ax


# ------------------------------------------------------------------ slides --

def s_title():
    fig = new_slide()
    ax = ct_axes(fig, [0.58, 0.12, 0.38, 0.76])
    ax.imshow(img("t01_find_large_nodule", 3), cmap="gray")
    fig.text(0.06, 0.66, "CT-ScanEval", color=FG, fontsize=54, weight="bold")
    text(fig, 0.06, 0.55, "Can a general-purpose AI model\nread a chest CT like a radiologist?", 24, MUTED)
    text(fig, 0.06, 0.25, "micro1 Eval Hackathon  ·  github.com/Moyo8/-CT-ScanEval", 15, MUTED)
    return fig


def s_problem():
    fig = new_slide("Why this matters")
    items = [
        ("Find small solid nodules", "a miss delays a cancer diagnosis"),
        ("Don't call vessels nodules", "false alarms mean extra scans and anxiety"),
        ("Measure accurately", "Lung-RADS follow-up changes at 6, 8 and 15 mm"),
    ]
    for i, (h, sub) in enumerate(items):
        y = 0.72 - i * 0.18
        fig.text(0.07, y, f"{i + 1}", color=ACC, fontsize=40, weight="bold", va="top")
        text(fig, 0.13, y, h, 24, FG, weight="bold")
        text(fig, 0.13, y - 0.065, sub, 18, MUTED)
    text(fig, 0.07, 0.17, "People already upload scans to chatbots. A fluent report can still get all three wrong.", 17, MUTED)
    return fig


def s_eval():
    fig = new_slide("The eval: 8 tasks, 5 CT slices each")
    for i in range(5):
        ax = ct_axes(fig, [0.05 + i * 0.18, 0.47, 0.17, 0.3])
        ax.imshow(img("t02_find_small_nodule", i + 1), cmap="gray")
        ax.set_title(f"slice {i + 1}" + (" (centre)" if i == 2 else ""), color=MUTED, fontsize=11)
    rows = [
        "Find: a 20 mm nodule · a 5 mm nodule · one stuck to the chest wall",
        "Rule out: two vessel-rich areas with no nodule",
        "Choose: the real nodule among three look-alikes",
        "Count two nodules  ·  Measure one in millimetres",
    ]
    for i, r in enumerate(rows):
        text(fig, 0.06, 0.38 - i * 0.065, "•  " + r, 18)
    text(fig, 0.06, 0.1, "Public LIDC-IDRI scans · every target outlined by all four radiologists", 15, MUTED)
    return fig


def s_task():
    fig = new_slide("One task, end to end: measure the nodule")
    ax = ct_axes(fig, [0.04, 0.08, 0.44, 0.76])
    ax.imshow(img("t08_measure_nodule", 3)[60:360, 230:512])
    text(fig, 0.52, 0.8, "Prompt gives:", 18, MUTED)
    text(fig, 0.52, 0.74, "pixel spacing 0.781 mm · orientation\ncoordinate system · JSON format", 17)
    text(fig, 0.52, 0.58, "Model replies:", 18, MUTED)
    fig.text(0.52, 0.52, '{"long_axis_mm": 6.0}', color=ACC, fontsize=22, family="monospace", va="top")
    text(fig, 0.52, 0.4, "Grader (plain Python, no AI judge):", 18, MUTED)
    text(fig, 0.52, 0.34, "4 radiologists measured 10.0-11.1 mm\naccept 8.5-12.6 mm  ->  6.0 = FAIL", 18)
    text(fig, 0.52, 0.17, "Finding tasks: click must land within\n4 mm of the radiologists' outlines", 16, MUTED)
    return fig


def s_validation():
    fig = new_slide("Is the grader fair?")
    stats = [("60", "automated grader tests pass"), ("8/8", "reference answers pass"), ("21%", "random answers (chance level)")]
    for i, (n, lab) in enumerate(stats):
        x = 0.07 + i * 0.31
        fig.text(x, 0.66, n, color=ACC, fontsize=56, weight="bold", va="top")
        text(fig, x, 0.48, lab, 18, MUTED)
    text(fig, 0.07, 0.32, "Wrong answers that must fail: mirror-image location, wrong side,\n"
         "size in pixels, coordinates off the image, 40 scattered points, hedged labels", 18)
    return fig


def s_results(summary):
    fig = new_slide("Results (Gemini, free tier)")
    ax = fig.add_axes([0.25, 0.14, 0.33, 0.66], facecolor=BG)
    names = ["Gemini 3.6 Flash", "Gemini 3.5 Flash-Lite", "Random answers"]
    vals = [summary["gemini-3.6-flash"], summary["gemini-3.5-flash-lite"], summary["random"]]
    bars = ax.barh(names[::-1], [v * 100 for v in vals[::-1]], color=[MUTED, BAD, GOOD])
    for b, v in zip(bars, vals[::-1]):
        ax.text(b.get_width() + 2, b.get_y() + b.get_height() / 2, f"{v * 100:.0f}%", color=FG, va="center", fontsize=20, weight="bold")
    ax.set_xlim(0, 110)
    ax.tick_params(colors=FG, labelsize=15)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([])
    ax.set_title("share of trials passed", color=MUTED, fontsize=14, loc="left")
    text(fig, 0.62, 0.74, "3.6 Flash found even the\n5 mm nodule every time", 18)
    text(fig, 0.62, 0.55, "Flash-Lite barely beats\nrandom guessing", 18)
    text(fig, 0.62, 0.36, "Both failed where it\nmatters most:", 18, ACC, weight="bold")
    text(fig, 0.62, 0.25, "measuring", 22, ACC, weight="bold")
    return fig


def s_measure():
    fig = new_slide("Failure 1: neither model can measure")
    ref = load_reference("t08_measure_nodule")
    m = load_masks("t08_measure_nodule")
    x, y = ref["answer"].get("x", ref["nodule"]["x"]), ref["nodule"]["y"]
    h = 40
    crop = img("t08_measure_nodule", 3)[y - h:y + h, x - h:x + h]
    ax = ct_axes(fig, [0.04, 0.1, 0.42, 0.74])
    ax.imshow(crop, extent=(x - h, x + h, y + h, y - h), interpolation="bicubic")
    ax.contour(np.arange(512), np.arange(512), m["union"], levels=[0.5], colors=GOOD, linewidths=2)
    px10 = 10 / ref["pixel_spacing_mm"]
    ax.add_patch(Rectangle((x - h + 2, y + h - 13), px10 + 8, 11, color=BG, alpha=0.8))
    ax.plot([x - h + 6, x - h + 6 + px10], [y + h - 4] * 2, color=FG, lw=4)
    ax.text(x - h + 6, y + h - 7, "10 mm", color=FG, fontsize=14)
    ax.text(x - h + 4, y - h + 6, "green = radiologists' outline", color=GOOD, fontsize=13)
    ax.set_xlim(x - h, x + h)
    ax.set_ylim(y + h, y - h)
    text(fig, 0.5, 0.8, "Radiologists:  10.0 - 11.1 mm", 22, GOOD, weight="bold")
    text(fig, 0.5, 0.7, "Flash-Lite:     5.0,  5.0,  5.0 mm", 21, BAD)
    text(fig, 0.5, 0.62, "3.6 Flash:      6.0,  7.1 mm", 21, BAD)
    text(fig, 0.5, 0.48, "Lung-RADS, first screen:", 18, MUTED)
    text(fig, 0.5, 0.42, "11 mm  ->  4A: repeat CT in 3 months\n6-7 mm ->  3: wait 6 months\n5 mm   ->  2: wait 12 months", 18,
         family="monospace")
    text(fig, 0.5, 0.18, "Every answer too small. No pixel-to-mm maths.", 18, ACC, weight="bold")
    return fig


def s_coords():
    fig = new_slide("Failure 2: right nodule, wrong coordinates")
    ax = ct_axes(fig, [0.03, 0.06, 0.46, 0.8])
    ax.imshow(img("t01_find_large_nodule", 3), cmap="gray")
    ax.add_patch(Circle((314, 205), 14, fill=False, color=GOOD, lw=2))
    ax.plot(300, 387, "x", color=BAD, ms=18, mew=4)
    ax.plot(300, 198, "o", color=ACC, ms=12, mfc="none", mew=3)
    text(fig, 0.53, 0.8, "Flash-Lite, large nodule (t01):", 18, MUTED)
    text(fig, 0.53, 0.74, '"nodule present, patient\'s left"  - correct', 18, GOOD)
    text(fig, 0.53, 0.64, "but answered  x = 300,  y = 387", 18, BAD)
    text(fig, 0.53, 0.54, "x in pixels, y on a 0-1000 scale\n(other answers: y = 551, 681 on a 512-px image)", 17)
    text(fig, 0.53, 0.37, "rescale y  ->  lands on the nodule", 18, ACC)
    text(fig, 0.53, 0.3, "green circle = nodule   red X = its answer   yellow o = rescaled", 13, MUTED)
    text(fig, 0.53, 0.2, "A tool using these numbers would point\nat the wrong part of the lung.", 17, MUTED)
    return fig


def s_falsepos():
    fig = new_slide("Failure 3: a vessel called a nodule")
    strip = np.hstack([img("t04_no_nodule_hilum", i)[231:311, 129:209] for i in range(1, 6)])
    ax = ct_axes(fig, [0.05, 0.4, 0.9, 0.38])
    ax.imshow(strip, cmap="gray")
    for i in range(5):
        ax.add_patch(Circle((i * 80 + 49, 40), 9, fill=False, color=BAD, lw=2))
        ax.text(i * 80 + 4, 76, f"slice {i + 1}", color=MUTED, fontsize=11)
    text(fig, 0.06, 0.32, "Hilum, no nodule anywhere. Flash-Lite said \"nodule\" here in 3 of 3 trials:", 19)
    text(fig, 0.06, 0.25, "a round vessel cross-section that continues through every slice.", 19, MUTED)
    text(fig, 0.06, 0.15, "3.6 Flash made the same kind of mistake once, on a tiny vessel dot.", 19, MUTED)
    return fig


def s_takeaway():
    fig = new_slide("Takeaway")
    text(fig, 0.07, 0.76, "Bigger models find nodules much better,\nbut still can't measure them.", 28, FG, weight="bold")
    text(fig, 0.07, 0.52, "The errors look reasonable: right format, plausible size.", 20, MUTED)
    text(fig, 0.07, 0.4, "Fix with grounding, not prompting:", 20, ACC, weight="bold")
    text(fig, 0.07, 0.33, "•  a real measuring tool (pixel extent -> mm in code)\n"
         "•  coordinates checked against the image size\n•  check neighbouring slices before calling a nodule", 19)
    return fig


def s_end():
    fig = new_slide()
    fig.text(0.5, 0.58, "CT-ScanEval", color=FG, fontsize=48, weight="bold", ha="center")
    fig.text(0.5, 0.45, "github.com/Moyo8/-CT-ScanEval", color=ACC, fontsize=24, ha="center")
    fig.text(0.5, 0.36, "tasks · graders · tests · every transcript", color=MUTED, fontsize=17, ha="center")
    return fig


SCRIPT = [
    (s_title, "C T Scan Eval. Can a general purpose A I model read a chest C T scan the way a radiologist does? "
              "This is my entry to the micro one Eval Hackathon."),
    (s_problem, "People already upload scans to chatbots, and developers are building vision models into triage tools. "
                "Lung cancer screening is a good test of whether that is safe, because it comes down to three concrete skills. "
                "First, seeing small solid nodules, because a miss delays a cancer diagnosis. "
                "Second, not mistaking blood vessels for nodules, because false alarms mean extra scans and anxiety. "
                "Third, measuring nodules accurately, because the Lung RADS follow up plan changes at six, eight and fifteen millimetres. "
                "A model can write a fluent report and still get all three wrong."),
    (s_eval, "C T Scan Eval has eight tasks built from the public L I D C dataset, where four radiologists outlined every nodule. "
             "Each task shows the model five consecutive C T slices, the way a radiologist scrolls through a scan. "
             "The model has to find a large nodule, a small five millimetre nodule, and one stuck to the chest wall. "
             "Rule out a nodule in two areas full of vessels. Pick the real nodule among three look alikes. "
             "Count two nodules. And measure one in millimetres."),
    (s_task, "Here is the measuring task. The prompt gives the pixel spacing, the orientation and the coordinate system, "
             "and the model must reply with a short jason answer. Grading is plain Python code, with no A I judge. "
             "The four radiologists measured this nodule at ten to eleven millimetres, so the grader accepts eight and a half "
             "to twelve point six. An answer of six fails. For the finding tasks, the clicked point must land within four "
             "millimetres of the radiologists' outlines."),
    (s_validation, "To make sure the grader is fair, sixty automated tests check that every correct answer passes, "
                   "and that plausible wrong answers fail, such as the mirror image location, a size given in pixels, "
                   "or forty points scattered everywhere. The reference answers score eight out of eight, "
                   "and random answers score twenty one percent."),
    (s_results, "I ran two Google Gemini models on the free tier. Gemini three point six Flash passed seventy eight percent of trials. "
                "The smaller Gemini three point five Flash Lite passed twenty five percent, barely above random guessing. "
                "The bigger model found even the five millimetre nodule every time. "
                "But both models failed where it matters most."),
    (s_measure, "The biggest failure is measuring. For this eleven millimetre nodule, Flash Lite answered five millimetres, three times. "
                "Gemini three point six Flash answered six, and seven point one. Every answer was too small. "
                "On a first screening scan, Lung RADS puts an eleven millimetre solid nodule in category four A, a repeat C T in three months. "
                "The models' answers would put it in category two or three, with a wait of six to twelve months. "
                "Neither model converted pixels into millimetres. They gave a plausible small number instead."),
    (s_coords, "The smaller model has a second problem. On the large nodule, it correctly said there is a nodule in the patient's left lung, "
               "but its coordinates were far off. Its x values were in pixels, while its y values were on a zero to one thousand scale, "
               "sometimes off the image entirely. Rescale y, and the point lands right on the nodule. "
               "A tool that uses these coordinates would point at the wrong part of the lung, and nothing in the answer would look wrong."),
    (s_falsepos, "Near the hilum, where there is no nodule at all, Flash Lite called a vessel cross section a nodule in all three trials. "
                 "Gemini three point six Flash made the same kind of mistake once, on a tiny vessel dot."),
    (s_takeaway, "The takeaway. Bigger models get much better at finding nodules, but not at measuring them, and the errors look perfectly reasonable. "
                 "Better prompting will not fix this. Measurement needs a real tool, coordinates need checking against the image, "
                 "and models need to check neighbouring slices before calling something a nodule."),
    (s_end, "The tasks, graders, tests and every transcript are on GitHub, under Moyo eight, C T Scan Eval. Thank you."),
]


def pass_rates():
    out = {}
    for d in (ROOT / "results").iterdir():
        f = d / "results.jsonl"
        if not f.exists():
            continue
        recs = [json.loads(line) for line in f.read_text().splitlines() if line]
        meta = json.loads((d / "run.json").read_text())
        key = meta["model"] if meta["provider"] == "gemini" else meta["provider"]
        if key not in out or len(recs) > out[key][1]:
            out[key] = (sum(r["passed"] for r in recs) / len(recs), len(recs))
    out["random"] = out.pop("random")
    return {k: v[0] for k, v in out.items()}


def wav_seconds(path):
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    rates = pass_rates()
    parts, total = [], 0.0
    for i, (fn, narration) in enumerate(SCRIPT, start=1):
        fig = fn(rates) if fn is s_results else fn()
        png = BUILD / f"s{i:02d}.png"
        fig.savefig(png, facecolor=BG)
        plt.close(fig)
        wav = BUILD / f"s{i:02d}.wav"
        subprocess.run(["espeak-ng", "-v", VOICE, "-s", SPEED, "-w", str(wav), narration], check=True)
        dur = wav_seconds(wav) + 1.0
        total += dur
        mp4 = BUILD / f"s{i:02d}.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(png), "-i", str(wav),
            "-af", "apad=pad_dur=1.0,aresample=44100", "-t", f"{dur:.2f}",
            "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p", "-r", "25",
            "-c:a", "aac", "-b:a", "128k", "-ac", "1", str(mp4),
        ], check=True)
        parts.append(mp4)
        print(f"slide {i:2d}: {dur:5.1f}s")
    lst = BUILD / "list.txt"
    lst.write_text("".join(f"file '{p.name}'\n" for p in parts))
    out = OUT / "CT-ScanEval_video.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", "-movflags", "+faststart", str(out)], check=True)
    print(f"total {total / 60:.1f} min -> {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
