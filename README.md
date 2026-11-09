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
python run_eval.py --provider gemini --model gemini-3.6-flash --trials 3 --pace 7
python run_eval.py --provider gemini --model gemini-3.5-flash-lite --trials 3 --pace 7
python run_eval.py --provider anthropic --model claude-opus-5-5 --effort high --trials 3   # paid
python run_eval.py --provider openai --model <vision model id> --trials 3                 # paid
python run_eval.py --provider random --trials 3
python analysis/summarize.py --latest            # results/summary.md and a chart
```

Each run makes 24 separate calls: 8 tasks, 3 trials each, with no shared history. Results go to `results/` and full transcripts to `transcripts/`. Each run records the requested and served model, the settings, the tokens used, the cost, the runtime and a UTC timestamp.

`--pace 7` spaces the calls 7 seconds apart to stay under Gemini's free-tier limits. If the daily quota runs out, the run stops cleanly and keeps the trials that finished. Both Gemini runs used the free tier and cost nothing. Flash-Lite's 24 calls took 204 seconds. Gemini 3.6 Flash thinks before answering, so its 18 calls took 780 seconds.

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

All models get the same 8 tasks and the same graders. The full table is in `results/summary.md` and a chart is in `results/charts/per_task_scores.png`.

| Model | Trials per task | Pass rate | Mean score | pass@k | pass^k | Cost per run |
|---|---|---|---|---|---|---|
| Gemini 3.6 Flash (`gemini-3.6-flash`, default settings, free tier, 9 Nov 2026) | 2–3* | 78% (14/18) | 0.80 | 88% | 62% | $0 |
| Gemini 3.5 Flash-Lite (`gemini-3.5-flash-lite`, default settings, free tier, 9 Nov 2026) | 3 | 25% (6/24) | 0.46 | 25% | 25% | $0 |
| Random answers | 3 | 21% (5/24) | 0.33 | 50% | 0% | $0 |
| Reference answers | 1 | 100% | 1.00 | 100% | 100% | $0 |

\*Gemini 3.6 Flash hit the free tier's limit of 20 requests a day after 18 calls. So it got three trials on t01 and t02 and two on everything else, and pass@k and pass^k use whatever trials each task got. Two later attempts that day got one call each (both on t01, both correct) before hitting the limit. They're kept in `results/` but not mixed into these numbers. I also tried Gemini 3.8 Flash, but that run didn't produce results.

| Task | 3.6 Flash | 3.5 Flash-Lite |
|---|---|---|
| t01 large nodule | 3/3 | 0/3 |
| t02 small nodule | 3/3 | 0/3 |
| t03 juxtapleural nodule | 2/2 | 0/3 |
| t04 no nodule (hilum) | 1/2 | 0/3 |
| t05 no nodule (lower lobes) | 2/2 | 3/3 |
| t06 nodule vs. vessels | 2/2 | 3/3 |
| t07 count nodules | 1/2 | 0/3 |
| t08 measure | 0/2 | 0/3 |

The bigger model does much better, which is a good sign the tasks measure something real. A scrambled eval would not separate the two models this cleanly, and Flash-Lite only just beats random answers.

3.6 Flash still fails in the places that matter most clinically. It got the measurement wrong both times, called a vessel a nodule once and missed a nodule once. Both models replied with bare JSON and no explanation, so everything below is worked out from what they answered and where they pointed. All the transcripts were read, and each verdict was checked against the review sheets. In none of the failures did the grader reject an answer that was actually right.

### What went wrong, task by task

**t08, measuring: neither model passed.** The four radiologists measured this nodule at 10.0–11.1 mm. Flash-Lite said 5.0 mm all three times; 3.6 Flash said 6.0 and 7.1 mm. Every answer was too small, by 4 to 6 mm. The nodule is about 14 pixels long at 0.78 mm per pixel, and none of the answers fits either number, so they look like a typical "small nodule" size rather than a measurement. This matters for follow-up. On a first screen, Lung-RADS puts a solid nodule of 11 mm in category 4A (repeat CT in 3 months, maybe a PET scan). An answer of 6–7 mm would put it in category 3 (repeat in 6 months), and 5 mm in category 2 (next year). [Flash-Lite](transcripts/20261109T093716Z__gemini__gemini-3.5-flash-lite/t08_measure_nodule__trial1.json), [3.6 Flash](transcripts/20261109T114327Z__gemini__gemini-3.6-flash/t08_measure_nodule__trial2.json)

**t01–t03, finding a nodule.** 3.6 Flash nailed all three: the large one, the 5 mm one (all three trials within 0.7 mm) and the one against the chest wall. Flash-Lite always named the right lung for t01 and t03, but gave positions in the wrong coordinate system. Its x values were in pixels, but its y values ran from 386 to 681 on an image only 512 pixels tall. That fits a 0–1000 scale. Rescale just y and all three t01 answers land inside the nodule, while the t03 ones land 7–8 mm away. So it saw roughly where the nodule was but reported the position in a frame nobody asked for. The prompt defines the 512-pixel grid clearly, so the grader is right to fail it. `analysis/coordinate_check.py` reproduces this. On the small nodule (t02), Flash-Lite gave three different answers to the same images: the wrong lung, roughly the right spot (off the image once y is taken at face value) and "no nodule". [Flash-Lite t01](transcripts/20261109T093716Z__gemini__gemini-3.5-flash-lite/t01_find_large_nodule__trial1.json), [Flash-Lite t02](transcripts/20261109T093716Z__gemini__gemini-3.5-flash-lite/t02_find_small_nodule__trial1.json)

**t04, ruling out a nodule at the hilum.** Flash-Lite reported a nodule at the same spot all three times. Rescaled, that's a round cross-section of a right hilar vessel that runs through all five slices. 3.6 Flash got it right once. The other time it called a tiny round vessel dot in the left upper lung a nodule; it's about 2 mm, under the 3 mm cut-off. Both are versions of the classic vessel-called-a-nodule mistake. [Flash-Lite](transcripts/20261109T093716Z__gemini__gemini-3.5-flash-lite/t04_no_nodule_hilum__trial1.json), [3.6 Flash](transcripts/20261109T114327Z__gemini__gemini-3.6-flash/t04_no_nodule_hilum__trial2.json)

**t07, counting two nodules.** 3.6 Flash found both nodules in one trial. In the other it reported only the 8 mm nodule and missed the 21 mm one tucked against the front of the chest beside the mediastinum, which is the bigger of the two. Flash-Lite answered 0, 0 and 1, and its one point wasn't on either nodule. [3.6 Flash](transcripts/20261109T114327Z__gemini__gemini-3.6-flash/t07_count_nodules__trial1.json), [Flash-Lite](transcripts/20261109T093716Z__gemini__gemini-3.5-flash-lite/t07_count_nodules__trial1.json)

**t05 and t06** were right every time for both models. t06 turned out easier than I meant it to be. Both models picked the nodule (B) in every trial, so the task may not force the slice-by-slice check it was designed to test. A harder version would use vessel candidates that match the nodule more closely in size and brightness on the middle slice.

### The main takeaway

A bigger model gets much better at finding nodules, but not at measuring them. On t08 every answer from both models was too small, and each would have pushed the nodule into a lower Lung-RADS category with a longer wait before the next scan. Neither model seems to turn pixels into millimetres. They give a plausible "small nodule" size instead. That's the dangerous kind of error: the answer is in the right format and sounds reasonable, and nothing about it looks wrong unless you measure the nodule yourself.

The smaller model adds a second trap. It often knows roughly where the nodule is, but reports y on a 0–1000 scale while x stays in pixels. Any tool that uses those coordinates to drop a marker, crop a region or measure would quietly point at the wrong part of the lung.

I don't think better prompting fixes either problem. Measurement needs a real tool, for example asking the model for the nodule's pixel extent and doing the millimetre maths in code. Coordinates need checking against the image size. Both models would also benefit from a habit of checking neighbouring slices before calling something a nodule.

## Limitations

**Slices, not a full scan.** Five 2-D slices covering about ±5 mm stand in for scrolling through a whole scan. A radiologist would have the full volume and other windows. The tasks are fair, but they're a cut-down version of real reading.

**Only eight cases.** There's one case per task, so the results describe these eight cases, not overall accuracy. pass^3 measures consistency, not performance across a population.

**The answer key is human too.** I only used nodules all four readers outlined, and the tolerances are based on how much those readers disagreed with each other. "No nodule" means none of the four marked a nodule of 3 mm or more there. Readers can miss things, so I also checked the negative slices myself.

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

I checked every case visually against the radiologists' outlines using the sheets in `review/`. The model transcripts were read one by one before the results above were written.
