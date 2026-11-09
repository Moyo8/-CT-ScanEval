# CT-ScanEval

Can a general-purpose multimodal model read a chest CT well enough to find a lung nodule, rule one out, tell a nodule from a blood vessel, count nodules and measure one? This eval checks exactly that, on eight cases from the public LIDC-IDRI dataset, using the outlines drawn by four radiologists as the answer key.

## Why I built it

People are already uploading scans to chatbots, and developers are wiring vision models into triage tools. Lung cancer screening is a good test of whether that's safe, because it comes down to three very concrete skills.

- **Seeing small solid nodules.** A miss delays a cancer diagnosis.
- **Not mistaking vessels for nodules.** A false call means extra CT scans, radiation and anxiety.
- **Measuring accurately.** Lung-RADS changes the follow-up plan at 6, 8 and 15 mm, so a wrong size means the wrong plan.

A model can write a confident, fluent report and still get all three wrong. I wanted something that catches that automatically, without an AI judge deciding whether the answer "sounds right".

## The tasks

Each task shows the model five consecutive axial CT slices, the way a radiologist scrolls through a scan. The slices are 512×512, in a lung window, ordered from head to feet and shown in the usual radiological orientation. The prompt spells out the pixel spacing, orientation and coordinate system, and asks for a short JSON answer.

Every target nodule is solid, non-calcified and was outlined by all four LIDC radiologists, so there's no argument about whether it's real.

| # | Task | Case | What the model has to do | Why it's hard |
|---|---|---|---|---|
| t01 | Find a large nodule | LIDC-IDRI-0294: 20 mm, left lung, touching the mediastinum | Say whether there's a nodule, click its centre, name the patient's side | Close to a control task. It mainly checks the basics and the radiological left/right convention (the patient's right is on the image's left) |
| t02 | Find a small nodule | 0803: 5 mm, peripheral left lung | Same | About 8 pixels across, and it looks like the vessel dots around it. Small nodules are the ones people miss |
| t03 | Find a juxtapleural nodule | 0133: 17 mm, right lung, on the back wall of the chest | Same | It sits against the pleura, so on one side there's no dark lung around it and it blends into the chest wall |
| t04 | Rule out a nodule at the hilum | 0963: nearest annotated nodule 101 mm away | Answer "no nodule" | The hilum is full of vessels and bronchi cut across, and many look round |
| t05 | Rule out a nodule in the lower lobes | 0803: nearest annotated nodule 31 mm away | Answer "no nodule" | Lower-lobe vessels run up and down, so on axial slices they appear as dozens of small dots |
| t06 | Nodule vs. vessels | 0067: an 8 mm nodule and two vessel cross-sections, marked A, B and C | Pick the nodule | On the middle slice they look the same. You have to follow each one through the neighbouring slices: a vessel carries on and drifts, a nodule ends |
| t07 | Count nodules | 0121: two nodules in the left lung on one slice | Give the count and both centres | One is a 21 mm nodule tucked against the front of the chest beside the mediastinum, the other is 8 mm. It's easy to stop at one |
| t08 | Measure a nodule | 0067: a marked 11 mm nodule | Give its longest diameter in mm | The model has to convert pixels to millimetres with the spacing it's given, not guess a typical size |

The five detection tasks (t01–t05) ask the identical question, so nothing in the prompt hints whether a nodule is present. The two "no nodule" cases stop a model that always says yes from scoring well.

The model only ever sees `tasks/<id>/task.json` and the five images. The answers, the radiologist outlines and the source details live in `reference/<id>/`.

## How answers are graded

Grading is plain Python (`graders/grader.py`), with no AI judge. The grader reads the last JSON object in the reply and checks only the final answer. A trial passes when every check passes, and a separate score gives partial credit.

- **Locating a nodule:** the clicked point must fall within 4 mm of the area any of the four radiologists outlined. That is about the radius of the smallest nodule, so an honest click passes, while a click on the vessel next door doesn't.
- **"No nodule" cases:** the right answer is simply "no". None of the four readers marked anything on those slices or for four slices either side.
- **Nodule vs. vessels:** the exact letter is required, so "B or A" fails.
- **Counting:** the count must be right, and each point is matched to a different nodule. Scattering points everywhere fails.
- **Measuring:** the four radiologists measured this nodule at 10.0–11.1 mm on this slice. I accept their range plus 1.5 mm either side, which gives 8.5–12.6 mm.

To make sure the grader is fair, `tests/test_graders.py` runs 60 checks.

- Every reference answer passes, whether it's bare JSON or wrapped in text or a code block.
- A long list of plausible wrong answers fails. It includes:
  - a missed nodule, the mirror-image spot, and the wrong side;
  - coordinates on a 0–1000 scale or outside the image;
  - an undercount, two points on one nodule, and 40 points scattered everywhere;
  - a size given in pixels;
  - a hedged label.
- A click 2–3.5 mm off the outline passes and one 6–9 mm off fails, which confirms the tolerance does what it says.
- The tests also check that nothing from the answer key leaks into the files the model sees.

There are two baselines. Replaying the reference answers through the whole pipeline (`--provider oracle`) scores 8/8. Random answers in the right format (`--provider random`) pass 5 of 24 trials, almost all from the two "no nodule" cases and the one-in-three choice.

## Running it

You need Python 3.11 or newer (I tested on 3.13).

```bash
pip install -r requirements.txt
python -m pytest -q tests                        # grader checks, 60 should pass
python run_eval.py --provider oracle --trials 1  # reference answers, 8/8
cp .env.example .env                             # add whichever API keys you have
python run_eval.py --provider gemini --model gemini-3.5-flash-lite --trials 3 --pace 7
python run_eval.py --provider gemini --model gemini-3.8-flash --trials 3 --pace 7
python run_eval.py --provider anthropic --model claude-opus-5-5 --effort high --trials 3   # paid
python run_eval.py --provider openai --model <vision model id> --trials 3                 # paid
python run_eval.py --provider random --trials 3
python analysis/summarize.py --latest            # results/summary.md and a chart
```

Each run makes 24 separate calls: 8 tasks, 3 trials each, with no shared history. Results go to `results/` and full transcripts to `transcripts/`. Each run records the requested and served model, the settings, the tokens used, the cost, the runtime and a UTC timestamp.

`--pace 7` spaces the calls 7 seconds apart to stay under Gemini's free-tier limits. If the daily quota runs out, the run stops cleanly and keeps the trials that finished. The Flash-Lite run took 204 seconds and cost nothing on the free tier.

If you'd rather use Colab, open `CT_ScanEval_Colab.ipynb` and add your keys as Colab Secrets.

### Rebuilding the cases from raw data

You don't need to do this to run the eval, because the task images are included. To rebuild them:

```bash
pip install -r requirements-data.txt
python data_prep/download_lidc.py       # 6 CT series (~0.8 GB) from TCIA's public API, no account needed
python data_prep/build_cases.py         # rebuilds tasks/ and reference/ byte for byte
python data_prep/make_review_sheets.py  # review/<task>.png with the answer overlaid
```

These were the selection rules:

- thin-slice scans (1–2.5 mm);
- nodules all four readers outlined;
- solid and non-calcified;
- nothing else annotated on the middle slice;
- no annotation anywhere near the "no nodule" slices.

The two vessels in t06 were found automatically: round bright spots whose 3-D structure runs through all five slices. Then they were checked by eye.

pylidc 0.2.3 is from 2019, so `data_prep/lidc_compat.py` patches two things newer Python removed to keep it working.

## Results

All models get the same 8 tasks, the same graders and 3 independent trials per task. The full table is in `results/summary.md` and a chart is in `results/charts/per_task_scores.png`.

| Model | Trials per task | Pass rate | Mean score | pass@3 | pass^3 | Cost per run |
|---|---|---|---|---|---|---|
| Gemini 3.5 Flash-Lite (`gemini-3.5-flash-lite`, default settings, free tier, 9 Nov 2026) | 3 | 25% (6/24) | 0.46 | 25% | 25% | $0 |
| Random answers | 3 | 21% (5/24) | 0.33 | 50% | 0% | $0 |
| Reference answers | 1 | 100% | 1.00 | 100% | 100% | $0 |

I also tried Gemini 3.8 Flash on the free tier, but that run didn't produce results, so it isn't reported here.

| Task | Flash-Lite passed | Mean score |
|---|---|---|
| t01 large nodule | 0/3 | 0.67 |
| t02 small nodule | 0/3 | 0.33 |
| t03 juxtapleural nodule | 0/3 | 0.67 |
| t04 no nodule (hilum) | 0/3 | 0.00 |
| t05 no nodule (lower lobes) | 3/3 | 1.00 |
| t06 nodule vs. vessels | 3/3 | 1.00 |
| t07 count nodules | 0/3 | 0.00 |
| t08 measure | 0/3 | 0.00 |

Flash-Lite only just beats random answers, and all its passes come from t05 and t06. It replied with bare JSON every time, with no explanation, so everything below is worked out from what it answered and where it pointed. All 24 transcripts were read, and each verdict was checked against the review sheets. In none of the failures did the grader reject an answer that was actually right.

### What went wrong, task by task

**t01, the large nodule, came closest.** All three times it said "nodule, patient's left", which is correct. It then gave coordinates 84–87 mm away from the nodule: (300, 387), (303, 393) and (305, 393), when the right answer is (314, 205). The x values are fine, but the y values are almost exactly double. If you rescale just y from a 0–1000 scale to the 512-pixel grid, all three points land inside the nodule. It saw the nodule and described its position in a coordinate system nobody asked for. The prompt defines the 512-pixel grid clearly, so the grader is right to fail it. `analysis/coordinate_check.py` reproduces this. [Transcript](transcripts/20261109T093716Z__gemini__gemini-3.5-flash-lite/t01_find_large_nodule__trial1.json)

**t03, the juxtapleural nodule.** It said "nodule, patient's right" three times, which is correct. But it gave y = 681 on an image only 512 pixels tall. Rescaled, the points land 7–8 mm from the nodule's edge: near it, but not on it. [Transcript](transcripts/20261109T093716Z__gemini__gemini-3.5-flash-lite/t03_find_juxtapleural_nodule__trial1.json)

**t02, the small nodule.** The three trials disagreed with each other. One pointed into the wrong lung, one was within 6 mm once y was rescaled (but y = 551 is off the image), and one said there was no nodule at all. Same images, three different answers. That's why pass^3 matters. [Transcript](transcripts/20261109T093716Z__gemini__gemini-3.5-flash-lite/t02_find_small_nodule__trial1.json)

**t04, the hilum.** It reported a nodule at the same spot all three times, (169, 529). Rescaled, (169, 271) is a round cross-section of a right hilar vessel that runs through all five slices. Nobody annotated anything there. It's the textbook vessel-called-a-nodule mistake. [Transcript](transcripts/20261109T093716Z__gemini__gemini-3.5-flash-lite/t04_no_nodule_hilum__trial1.json)

**t05 and t06** were right every time.

**t07, counting.** It answered 0, 0 and 1. The one point it gave isn't on either nodule. So it missed both the 21 mm nodule against the front of the chest and the 8 mm one, though both are plainly visible on that slice. [Transcript](transcripts/20261109T093716Z__gemini__gemini-3.5-flash-lite/t07_count_nodules__trial1.json)

**t08, measuring.** It answered 5.0 mm every time. The radiologists measured 10.0–11.1 mm. The nodule is about 14 pixels long at 0.78 mm per pixel, so 5.0 matches neither the pixel count (~14) nor the actual size (~11 mm). It looks like a generic "small nodule" number rather than a measurement. This matters clinically: for a solid nodule on a first screen, Lung-RADS calls 5 mm category 2 (see you next year) and 11 mm category 4A (repeat CT in 3 months, maybe a PET scan). [Transcript](transcripts/20261109T093716Z__gemini__gemini-3.5-flash-lite/t08_measure_nodule__trial1.json)

### The main takeaway

The model often knows roughly where the nodule is, but it can't tell you where it is, and it doesn't measure.

In all eight answers where Flash-Lite placed a nodule, x was in pixels but y ran from 386 to 681, which fits a 0–1000 scale. Four of those points are outside the image, and so is the t04 false positive. Rescale y and the t01 answers sit right on the nodule.

That should worry anyone building on top of these models. A tool that uses the coordinates to drop a marker, crop a region or measure would quietly point at the wrong part of the lung, and nothing in the neat JSON answer would look off. The other mistakes are the classic ones: a vessel called a nodule, two nodules missed, and a size that ignores the pixel spacing.

I don't think better prompting fixes this. It needs grounded output, such as coordinates checked against the image size or an actual measuring tool, and a habit of checking the neighbouring slices before calling something a nodule.

## Limitations

**Slices, not a full scan.** Five 2-D slices covering about ±5 mm stand in for scrolling through a whole scan. A radiologist would have the full volume and other windows. The tasks are fair, but they're a cut-down version of real reading.

**Only eight cases.** There's one case per task, so the results describe these eight cases, not overall accuracy. pass^3 measures consistency, not performance across a population.

**The answer key is human too.** I only used nodules all four readers outlined, and the tolerances are based on how much those readers disagreed with each other. "No nodule" means none of the four marked a nodule of 3 mm or more there. Readers can miss things, so the negative slices were also checked visually.

**Coordinates mix two skills.** Asking for pixel coordinates tests seeing the nodule and reporting a position at the same time. The 4 mm tolerance, the coordinate-free t06 task and `analysis/coordinate_check.py` help separate the two, and the results above show why that separation matters.

**Research only.** None of these models are approved for diagnosis. This is not a clinical tool.

## Data and credit

The images come from LIDC-IDRI on The Cancer Imaging Archive, licensed CC BY 3.0:

- Armato SG III et al., *The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI)*, Medical Physics 38(2):915–931, 2011. Data DOI: [10.7937/K9/TCIA.2015.LO9QL9SX](https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX).
- Clark K et al., *The Cancer Imaging Archive (TCIA)*, J Digit Imaging 26(6):1045–1057, 2013.

The data is public and de-identified. This repository only contains 512×512 PNG slices and outline masks for six patients. I used pylidc, pydicom, NumPy, SciPy and Pillow to process it.

Everything in this repository was built during the hackathon.

## How I built this

I used Claude Code, an AI coding assistant, throughout. It wrote most of the code (data preparation, case builder, graders and tests, harness) and the first draft of this README. It also suggested the task list and the candidate cases. I picked the problem area and the dataset, and I ran the models.

Every case was checked visually against the radiologists' outlines using the sheets in `review/`. The model transcripts were read one by one before the results above were written.
