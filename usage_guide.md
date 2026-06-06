# ARC-AGI-3 Agent — Usage Guide

This guide explains how to use, iterate, and submit the **Graph-Based Exploration Agent** implemented in this repository.

---

## 1. How the Agent Works

The agent is implemented in `agent/my_agent.py`. It uses a hybrid search and exploration architecture:

```
  +--------------------------------------------------------------+
  |                   High-Level Macro Search                    |
  |  - Hashing states using the 3D grid                          |
  |  - Constructing a directed graph of transition states        |
  |  - Backtracking to untried nodes using BFS in memory         |
  +--------------------------------------------------------------+
                                 │
                                 ▼
  +--------------------------------------------------------------+
  |                   Low-Level Micro Navigation                 |
  |  - Dynamically learning wall collisions and barriers        |
  |  - Dynamically mapping controller layouts (ACTION1 -> Dir)   |
  |  - Pathfinding directly to target coordinates using A*       |
  +--------------------------------------------------------------+
```

---

## 2. Command Reference

All commands must be executed from the project root (`/Users/sunnyluffy/test/kagglle/ARC/`).

### Local Testing & Verification

| Command | Action |
|---|---|
| `make verify-local` | Runs a 30-second smoke test on games `vc33` and `ls20` (capped at 50 steps). |
| `make play-local` | Runs the agent against **all** 25 cached environments. |
| `make play-local GAME=ls20` | Runs the agent against a specific game (highly recommended during iteration). |
| `make play-local GAME=vc33,bp35` | Runs the agent against a comma-separated list of games. |
| `make list-games` | Lists all available games in the benchmark. |

---

## 3. Logs & Observability

### Console Output
When you run the agent, it outputs detailed step logs:
- `RESET`: Triggers level start or restart.
- `ACTION1-5,7`: Simple interactions and movement.
- `ACTION6`: Targeted centroids exploration.
- `levels completed`: Puzzles solved in this track.

### File Logs
The run writes complete logs to `logs.log` in the root directory. You can tail it to debug state hashing or path planning:
```bash
tail -f logs.log
```

---

## 4. How to Submit to Kaggle

To submit your agent to the Kaggle competition:

1. **Verify your Kaggle Username:**
   Open `notebooks/kernel-metadata.json` and replace `REPLACE_WITH_YOUR_USERNAME` with your official Kaggle handle.

2. **Select your Hardware Accelerator:**
   Open `scripts/build_notebook.py` and set the `ACCELERATOR` constant near the top:
   - `"cpu"`: No GPU.
   - `"t4"`: Dual Nvidia T4 (Default; matches Kaggle baseline).
   - `"rtx6000"`: Heavy GPU acceleration (ARC-AGI-3 exclusive).

3. **Deploy:**
   Run the following command to bundle your agent and upload it to Kaggle:
   ```bash
   make submit
   ```

4. **Monitor Rerun:**
   Track the upload using:
   ```bash
   make status
   ```
   Once `complete`, navigate to your notebook on kaggle.com and click **"Submit to Competition"** in the top right.
