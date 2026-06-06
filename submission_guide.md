# 🚀 Submission Guide — ARC Prize 2026 (ARC-AGI-3)

> **TL;DR:** You cannot submit directly from this terminal. Submissions happen through Kaggle's notebook runner. Here's the exact 5-step process.

---

## How Submission Works

Kaggle evaluates your agent by running a **private Jupyter notebook** inside their infrastructure. The notebook:
1. Installs `arc-agi` from the offline competition dataset
2. Writes your `agent/my_agent.py` into the Kaggle runtime
3. Connects to a live **Gateway Sidecar API** that runs actual game levels
4. Plays all 25 games, records scores, and writes `submission.parquet`

Your local machine **cannot** connect to this sidecar — it only runs inside Kaggle's private evaluation environment.

---

## Prerequisites (One-time Setup)

### 1. Create a Kaggle API Token

Go to → [https://www.kaggle.com/settings](https://www.kaggle.com/settings) → **API** → **Create New Token**

This downloads a `kaggle.json` file. Extract the token value from it:
```json
{"username":"abhilashmanchala","key":"YOUR_API_KEY_HERE"}
```

Save just the `key` value as a one-line file in the project:
```bash
mkdir -p .kaggle
echo "YOUR_API_KEY_HERE" > .kaggle/access_token
```

### 2. Update Your Kaggle Username in the Notebook Metadata

Edit [`notebooks/kernel-metadata.json`](file:///Users/sunnyluffy/test/kagglle/ARC/notebooks/kernel-metadata.json):

```json
{
  "id": "abhilashmanchala/arc-prize-2026-arc-agi-3-starter",
  "title": "ARC Prize 2026 — ARC-AGI-3 Starter",
  ...
}
```

Replace `REPLACE_WITH_YOUR_USERNAME` with `abhilashmanchala`.

> ⚠️ **This is required only once.** After the first push, Kaggle tracks the kernel by this ID.

---

## Submission Workflow

### Step 1 — Test Locally First (Always do this)

```bash
# Quick smoke test (50 steps, 2 games):
make verify-local

# Full local benchmark (200 steps, all 25 games):
make play-local

# Single game debug:
make play-local GAME=vc33 STEPS=200
```

**Current local scores:**
| Game | Levels Completed |
|------|-----------------|
| `lp85` | 1 ✅ |
| `vc33` | 0 (but progresses significantly) |
| All others | 0 |
| **Aggregate score** | **~0.10 (10%)** |

### Step 2 — Build the Submission Notebook

```bash
make notebook
```

This splices `agent/my_agent.py` into `notebooks/submission.ipynb` automatically. You don't need to edit the notebook manually.

### Step 3 — Submit to Kaggle

```bash
make submit
```

This runs `make notebook` then pushes to Kaggle via the CLI. The kernel will be queued for evaluation.

### Step 4 — Monitor the Run

```bash
make status
```

Or go to your Kaggle kernels page:  
`https://www.kaggle.com/abhilashmanchala/arc-prize-2026-arc-agi-3-starter`

The run takes **~9 hours** on the private test set. Check back after.

### Step 5 — Check the Leaderboard

After the kernel completes, your score will appear at:  
`https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard`

---

## From GitHub to Kaggle — The Full Flow

```
[Local Dev]                    [GitHub]                [Kaggle]
agent/my_agent.py  →  git push origin main  →  make submit  →  Kaggle kernel runs
     ↑                                                              ↓
 Edit & test locally                                    submission.parquet → score
```

**Yes, your GitHub repo and Kaggle submission are independent.** The Makefile builds the notebook from your local `agent/my_agent.py`. It does NOT pull from GitHub automatically — so always make sure your local file matches what you want to submit.

---

## Can We Submit Right Now?

**Almost.** You need:

| Requirement | Status |
|-------------|--------|
| `agent/my_agent.py` coded & tested | ✅ Done |
| Code pushed to GitHub | ✅ `https://github.com/abhilashmanchala/ARC_2026_3` |
| `notebooks/kernel-metadata.json` username updated | ❌ Need to add `abhilashmanchala` |
| `.kaggle/access_token` file created | ❌ Need your Kaggle API key |

Once those two items are done:
```bash
make submit
```

---

## Accelerator Choice

The notebook is configured to use **Dual T4 GPUs** (default). Since our agent is pure Python (no ML inference), CPU is sufficient and won't burn your GPU quota:

Edit `scripts/build_notebook.py`, line:
```python
ACCELERATOR = "cpu"   # Change from "t4" to "cpu" to save GPU quota
```

---

## Daily Submission Limits

Kaggle limits submissions per day (usually 5–10 for most competitions). Plan accordingly:
- Use `make play-local` to validate **before** every submission
- Only submit when you have a meaningful improvement

---

## Quick Reference

```bash
make verify-local              # Fast test: 50 steps on vc33 + ls20
make play-local                # Full test: 200 steps on all 25 games
make play-local GAME=vc33      # Debug a single game
make notebook                  # Build submission.ipynb from my_agent.py
make submit                    # Build + push to Kaggle (requires .kaggle/access_token)
make status                    # Check Kaggle run status
```
