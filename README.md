# CT-ScanEval: can general-purpose multimodal models find and measure lung nodules on chest CT?

> **Status:** framework, tasks, reference answers and graders are complete and validated.
> Sections marked **[AUTHOR: ...]** must be filled in by the author from real runs and
> their own review before submission. No model results are claimed until then.

## What it tests

Eight chest-CT reading tasks taken from the radiologist-annotated LIDC-IDRI dataset. Each task gives a model **five consecutive axial CT images**, the way a radiologist scrolls through a scan, and asks it to do something a radiologist does every day: find a nodule, rule one out, tell a nodule from a vessel, count nodules, or measure one in millimetres. Answers are JSON and are graded by code against the outlines drawn by **all four LIDC radiologists**. There is no LLM judge.

## Who it is for and why it matters

General multimodal models are already being used to "read" medical images: by patients uploading their own scans, by developers building triage tools, and in clinical copilots. Lung cancer screening depends on **detecting** small solid nodules and **measuring** them, because Lung-RADS management changes at 6, 8 and 15 mm. Three kinds of failure matter clinically:

- **Missing a nodule** (false negative) delays a cancer diagnosis.
- **Calling a vessel a nodule** (false positive) leads to needless follow-up CTs, radiation and anxiety.
- **Mis-measuring** changes the follow-up interval.

A model that writes a fluent report but makes these errors is dangerous precisely because the report sounds right. This eval measures these errors directly.

## The tasks

All target nodules are **solid, non-calcified, and outlined by all four radiologists**. Every image is a 512×512 PNG in a standard lung window (W1500/L−600). Images are 2.0–2.5 mm apart, ordered superior to inferior, in radiological orientation. The prompt states the pixel spacing, orientation and coordinate system.

| # | Task | Case (LIDC-IDRI) | What the model must do | Why it is hard |
|---|---|---|---|---|
| t01 | Find a large nodule | 0294, 20 mm, left lung, abuts the mediastinum | Say a nodule is present, give its centre (x, y), name the patient's side | Near-control. Tests the basics and the radiological left/right convention (patient's right is on image left) |
| t02 | Find a small nodule | 0803, 5 mm, peripheral left lung | Same | ~8 px across; looks like the vessel dots around it. Small nodules are the most often missed |
| t03 | Find a juxtapleural nodule | 0133, 17 mm, right lung, on the posterior pleura | Same | No dark lung rim on one side, so it merges with the chest wall. A known blind spot for readers and CAD |
| t04 | No nodule (hilum) | 0963, nearest annotated nodule 101 mm away | Correctly answer "no nodule" | Dense branching vessels and bronchi in cross-section mimic nodules |
| t05 | No nodule (lower lobes) | 0803, nearest annotated nodule 31 mm away | Correctly answer "no nodule" | Lower-lobe vessels run vertically, so they appear as dozens of round dots |
| t06 | Nodule vs. vessels | 0067, 8 mm nodule plus two vessel cross-sections, marked A/B/C | Pick the nodule | The three look alike on the centre image. Only following them through adjacent images tells them apart: a vessel persists and moves, a nodule ends |
| t07 | Count nodules | 0121, two nodules in the left lung on one image | Give the count and both centres | One nodule touches the anterior pleura beside the mediastinum, the other is small and in the lung. Stopping at one undercounts |
| t08 | Measure a nodule | 0067, marked 11 mm nodule | Give the long-axis diameter in mm | The model must turn pixels into millimetres using the stated spacing rather than guess a typical size |

Detection tasks t01–t05 use **exactly the same question**, so the prompt gives no hint whether a nodule is present. The negative cases t04/t05 stop a model that always says "yes" from scoring well. The full prompt for each task is in `tasks/<id>/task.json`. The exact case details (series UID, slice indices, reader measurements) are in `reference/<id>/reference.json`, which the model never sees.

[AUTHOR: replace or extend these "why it is hard" notes in your own words from your imaging experience.]

## Grading

All graders are deterministic Python (`graders/grader.py`). They parse the **last JSON object** in the reply and grade only the outcome, never the reasoning. Each check is pass/fail. A trial **passes** only if every check passes. The **score** gives partial credit.

| Task type | Checks | Tolerance and why |
|---|---|---|
| Find, positive (t01–t03) | `detected`, `located`, `correct_side` | Point within **4 mm** of the union of the four radiologists' outlines on the centre image. 4 mm is about the radius of the smallest target, so a correct click passes even with slightly loose coordinates. A click on a neighbouring vessel ≥5 mm away fails |
| Find, negative (t04–t05) | `correctly_reports_no_nodule` | No region on these images, nor within ±4 more slices, was marked by any of the four readers |
| Choice (t06) | `picked_nodule` | Exact label. Hedges such as "B or A" fail |
| Count (t07) | `count_correct`, `all_nodules_located`, `no_extra_points` | One-to-one matching of points to nodules, each within 4 mm. Spraying points fails |
| Measure (t08) | `size_within_reader_range` | Accept the range of the four readers' own long-axis measurements on this image (10.0–11.1 mm) widened by ±1.5 mm, giving 8.5–12.6 mm |

**How the graders were validated** (`tests/test_graders.py`, 60 tests, all passing):

- Every reference answer passes, both as bare JSON and wrapped in prose or a code fence.
- These deliberately wrong answers all fail:
  - missing the nodule;
  - the mirror-image location;
  - the right location with the wrong side;
  - the side named as "image-left";
  - coordinates on a 0–1000 scale, out of range or missing;
  - an undercount;
  - two points on one nodule;
  - an extra point;
  - spraying 40 points;
  - a size given in pixels or as "about 1 cm";
  - a hedged A/B label;
  - unparseable text.
- A point 2–3.5 mm from the outline passes, while one 6–9 mm away fails, which confirms the stated tolerance.
- No hidden data leaks into the model-facing files, and positive and negative detection prompts are identical.

**Baselines:**

- `--provider oracle` replays the reference answers through the full harness and scores 8/8.
- `--provider random` gives format-valid random answers. It sets the chance level, about 20–25% of trials (5/24 with the committed seed), almost all from the two "no nodule" cases and the 1-in-3 choice task.

[AUTHOR: hand-grade a few real model transcripts and confirm the grader agrees with you. Note any disagreement here.]

## How to run

Use Python 3.11+ (tested on 3.13). Results are written to `results/`, with full transcripts in `transcripts/`.

```bash
pip install -r requirements.txt
python -m pytest -q tests                       # 60 grader tests
python run_eval.py --provider oracle --trials 1 # reference answers: 8/8
cp .env.example .env                            # then add the key(s) you have
python run_eval.py --provider gemini --model gemini-3.8-flash --trials 3 --pace 7       # free tier
python run_eval.py --provider gemini --model gemini-3.5-flash-lite --trials 3 --pace 7  # free tier
python run_eval.py --provider anthropic --model claude-opus-5-5 --effort high --trials 3   # paid, optional
python run_eval.py --provider openai --model <vision model id> --trials 3                 # paid, optional
python run_eval.py --provider random --trials 3
python analysis/summarize.py --latest           # writes results/summary.md and a chart
```

Notes:

- `CT_ScanEval_Colab.ipynb` does the same in Google Colab, using Colab Secrets for the keys.
- Each run is 8 tasks × 3 trials = 24 independent single-turn calls. No conversation history is shared between trials. Calls go trial by trial (all 8 tasks for trial 1, then trial 2, ...).
- `--pace 7` sends one call every 7 s to stay under free-tier per-minute limits. If a provider's daily quota runs out, the run stops cleanly. The finished trials are kept, and `run.json` records that the run was incomplete.
- The harness records the requested and served model ID, settings, token usage, cost estimate (`harness/pricing.json`), runtime and UTC timestamp.
- For Claude, thinking is left at the model default (adaptive) and depth is set with `--effort`.
- Refusal fallbacks to another model are deliberately **off**: in an eval every answer must come from the model named, so a refusal counts as a failed trial.

**Cost and runtime:**

- Gemini runs on Google's free tier cost $0. Each takes about 5 minutes at `--pace 7`.
- A paid Claude Opus 5.5 run at high effort is estimated at about $2–4.
- [AUTHOR: replace with the measured figures from `results/*/run.json`.]

### Rebuilding the dataset from scratch (optional)

The built task images are included, so this is not needed to run the eval.

```bash
pip install -r requirements-data.txt
python data_prep/download_lidc.py      # 6 CT series (~0.8 GB) from TCIA's public API, no account
python data_prep/build_cases.py        # rebuilds tasks/ and reference/ (byte-identical)
python data_prep/make_review_sheets.py # review/<task>.png, reference overlaid for checking
```

`data_prep/lidc_compat.py` patches two removed-API issues (NumPy aliases, `SafeConfigParser`) so that pylidc 0.2.3 (2019) runs on current Python.

**How cases were selected:**

- Thin-slice scans (1–2.5 mm).
- Nodule groups annotated by all four readers.
- Solid (median texture 5) and non-calcified (median calcification 6).
- No other annotation on the centre image.
- Negative stacks have no annotation by any reader within ±4 slices of the stack.
- t06 vessel candidates were found automatically: round bright blobs whose 3-D connected structure runs through all five images, away from every annotation. They were then checked visually.

## Results

[AUTHOR: paste `results/summary.md` here after the real runs. Report every model run, including failed or partial ones.]

| Model (version, settings) | Trials per task | Pass rate | pass^3 | Cost per full run |
|---|---|---|---|---|
| [frontier model A] | 3 | | | |
| [frontier model B] | 3 | | | |
| [smaller model] | 3 | | | |
| random baseline | 3 | | | $0 |
| reference solution (oracle) | 1 | 100% | 100% | $0 |

### Per-task notes

[AUTHOR: for each task, write which models failed, how, and link one representative transcript (`transcripts/<run>/<task>__trialN.json`). Include at least one task where models came close and what tripped them up. Read the failures before writing: for each one, decide whether the model erred or the grader rejected a valid answer.]

### Main failure mode / hot take

[AUTHOR: one paragraph, backed by transcripts.]

## Limitations

- **Images, not volumes.** Five 2-D images (±5 mm) stand in for scrolling through a whole scan. A radiologist would see the full volume and other windows, so the tasks are a fair but reduced version of real reading.
- **Small sample.** There are eight cases, one per task. Results describe these cases, not general accuracy. pass^k over three trials measures consistency, not population performance.
- **Reference standard.** LIDC readers annotated independently and then reviewed each other's marks. Each target uses nodules all four marked, and the tolerance is based on their own disagreement. Negative cases mean "no ≥3 mm nodule marked by any of the four readers". Readers can miss nodules too. [AUTHOR: confirm you inspected the negative images yourself.]
- **Coordinates.** Location is reported as pixel coordinates, which mixes detection with the model's ability to output coordinates. The 4 mm tolerance and the coordinate-free t06 reduce this. Check transcripts to see whether a model named the right structure but gave the wrong numbers.
- **Not clinical use.** General-purpose models are not validated for diagnosis. This is a research evaluation.

## Data and provenance

- **LIDC-IDRI**, The Cancer Imaging Archive, licensed under **CC BY 3.0**. Armato SG III et al., *The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI)*, Medical Physics 38(2):915–931, 2011. Data DOI: [10.7937/K9/TCIA.2015.LO9QL9SX](https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX). TCIA citation: Clark K et al., J Digit Imaging 26(6):1045–1057, 2013.
- The dataset is de-identified and public. The ZIP contains only derived 512×512 PNG slices and radiologist outline masks for six patients.
- Tools: pylidc (annotation clustering and consensus), pydicom, NumPy/SciPy, Pillow.
- **Built during the hackathon:** everything in this repository.
- **Existing before the hackathon:** [AUTHOR: e.g. "My earlier lung-cancer detection project informed the domain but no code, data or model from it is used here."]

## AI use

[AUTHOR: make this accurate.] The code was drafted with an AI coding assistant (Claude Code): data preparation, case builder, graders and tests, harness and this README's structure. The AI also proposed the task design and the candidate cases. The author:

- [ ] reviewed every task image and reference answer on the `review/` sheets;
- [ ] checked every "why it is hard" claim;
- [ ] checked the grading tolerances;
- [ ] read the model transcripts and wrote the Results analysis.

## Repository layout

```
README.md                 this file
tasks/<id>/               task.json (prompt, image labels) + images/image_1..5.png   <- what the model sees
reference/<id>/           reference.json (answer, grading params, source) + masks.npz  <- hidden
graders/grader.py         code-based graders
run_eval.py               harness;  harness/providers.py (Claude, OpenAI, Gemini, oracle, random)
analysis/summarize.py     results table + chart
tests/test_graders.py     grader validation
data_prep/                download + case-building scripts (reproduces tasks/ and reference/)
review/                   per-task review sheets (reference overlaid) used for human checking
results/, transcripts/    run outputs
CT_ScanEval_Colab.ipynb   run everything in Colab
.env.example              key names only
```
