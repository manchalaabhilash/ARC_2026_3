# Goal Description

Our goal is to optimize the step efficiency of the Graph-Based Exploration Agent to achieve a much higher score on the ARC-AGI-3 benchmark. Because level scores are computed as the square of human-to-agent action efficiency `(h / a)^2`, minimizing unnecessary steps is critical for a winning submission.

---

## User Review Required

We propose adding several advanced heuristics to prune the exploration space. These heuristics are based on the core knowledge priors of ARC:
1. **Obstacle Color Generalization:** Once the player bumps into a color (e.g., grey) and is blocked, the agent learns that this color represents a wall and treats *all* cells of this color as obstacles, avoiding them in A* pathfinding.
2. **Systematic Probing:** The agent will dedicate the first 4–5 steps of a level to cleanly execute simple movement actions sequentially, establishing controller bindings before doing any complex clicks.
3. **Click Space Pruning:** The agent will avoid clicking on colors identified as obstacles (walls) or background, and will prune any coordinates that result in state self-loops.
4. **Target Color Heuristics:** The agent will prioritize navigating to unique target colors (e.g. gold, green, pink) that typically represent goals in ARC puzzles.

> [!NOTE]
> These changes are completely training-free and compatible with the Kaggle offline container timeout constraints.

---

## Proposed Changes

### [Agent Component]

#### [MODIFY] [my_agent.py](file:///Users/sunnyluffy/test/kagglle/ARC/agent/my_agent.py)
We will modify `MyAgent` to implement:
- **`self.obstacle_colors` Set:** To track wall colors dynamically.
- **Dynamic Color-Wall Classification:** Learn wall colors when a movement action results in zero displacement.
- **Probing Queue:** Initial action queuing for directional probing.
- **Target Color Prioritization:** Identifying and heading towards candidate goal colors.
- **Self-Loop Pruning:** Tracking and filtering out useless clicks.

---

## Verification Plan

### Automated Tests
- Run `make verify-local` (smoke test on games `vc33` and `ls20`).
- Run `make play-local` to ensure levels completed increases and scores improve.
- Output the action count per level to verify that step efficiency increases.
