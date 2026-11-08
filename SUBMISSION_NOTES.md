# What's left before you submit (not part of the ZIP)

Deadline: **Nov 9, 05:15 GMT = 06:15 Nigerian time**. Aim to submit by midnight on Nov 8.

## 1. Run the models (about 30 min)
1. Download `dist/CT-ScanEval_submission.zip`.
2. Open `CT_ScanEval_Colab.ipynb` in Colab (File, Upload notebook).
3. Add your key(s) as Colab Secrets and run all cells.
4. You need **at least one frontier model with 3 trials**. A second model makes "results show a clear spread across models" possible. Good options:
   - two Claude models (`claude-opus-5-5` and `claude-haiku-5-5`), or
   - Claude plus whichever OpenAI or Gemini key you have.
5. Download `ctscaneval_runs.zip` at the end.

Sanity checks from the guide:
- If a strong model scores 0% everywhere, look for a broken task before claiming a finding.
- A stronger model or higher effort should generally do better. If a small model beats a large one, check the transcripts for grader problems first.

## 2. Read the transcripts (about 1 h, the part judges value most)
For every failed trial, open `transcripts/<run>/<task>__trialN.json` and read `response_text`. Then ask:
- Did the model make a mistake, or did the grader reject a valid answer?
- What exactly went wrong? For example: a vessel was called a nodule, image-left was confused with patient-left, the nodule was right but the coordinates were off, pixels were reported as mm, or the count was 1.

## 3. Review every task yourself (about 30 min, worth 10 points)
Open `review/<task>.png`. The last panel shows the reference: red is the 50% consensus outline, cyan is any reader's outline, and the + is the reference point. For each task, confirm:
- the nodule is where the reference says, and is solid;
- on t04/t05, you see no nodule ≥3 mm on the centre image;
- on t06, A and C really are vessels and B is the nodule (scroll images 1–5);
- the measurement range for t08 looks right to you.

If you disagree with any task, tell me and I'll rebuild it.

## 4. Fill the [AUTHOR: ...] parts of README.md, then rebuild the ZIP
Replace the results table, per-task notes, hot take, cost/runtime, prior work and AI use. Copy `ctscaneval_runs.zip`'s `results/` and `transcripts/` into the project, then run:

```bash
python analysis/summarize.py --latest
```

```bash
python make_submission.py
```

The ZIP builder refuses to build if it finds a `.env` file or anything that looks like an API key, and checks the 50 MB limit.

## 5. Video (5 min max; an AI voice is fine; no face needed)
Use the guide's structure:
- **0:00–1:00:** the problem. Screening depends on finding and measuring small nodules, and general AI models are being used to read scans. Show one CT image.
- **1:00–2:30:** one task end to end. Show the 5 images, the prompt, the JSON answer and how the grader checks it against the 4 radiologists' outlines. Use t06 or t02.
- **2:30–4:30:** the results table, then walk through one or two failing transcripts.
- **4:30–5:00:** the takeaway.

Make sure the link opens without a permission request (YouTube unlisted or Drive "anyone with the link").

## 6. Written description (draft; fill the brackets from your results)
> CT-ScanEval tests whether general-purpose multimodal models can read chest CT the way a radiologist does: find a solid lung nodule, rule one out, tell a nodule from a blood vessel, count nodules and measure one in millimetres. It has 8 tasks built from the public LIDC-IDRI dataset, each with 5 consecutive CT slices. Every answer is graded by code against the outlines of all four LIDC radiologists, with no LLM judge. I tested [models] with 3 trials per task. [Headline finding, e.g. "the best model passed X% of trials; all models failed the small-nodule and nodule-vs-vessel tasks because ..."].
