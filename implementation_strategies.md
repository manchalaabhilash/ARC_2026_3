# ARC-AGI-3 — Agent Implementation & Success Strategies

To achieve a winning score on the ARC-AGI-3 benchmark, our agent must solve 25+ distinct game tracks efficiently without human intervention. This document outlines multiple implementation options for each key stage of the agent pipeline.

---

## 1. Phase 1: Action Mapping & Exploration
*Goal: Determine which keys/actions map to which mechanics (e.g., movement, rotation, interaction) at the start of a level with minimal step cost.*

```carousel
### Option A: Systematic Action Probing
- **Mechanism:** At the start of a level, the agent executes each action sequentially once or twice, comparing the resulting frame to the initial frame.
- **Analysis:** By checking the coordinate differences of specific pixel clusters (e.g., the player sprite), the agent instantly maps directional buttons (`Move Up`, `Move Down`, `Move Left`, `Move Right`).
- **Pros:** Zero-dependency, fast, extremely low action footprint (usually 4–5 steps).
- **Cons:** If actions require state prerequisites (e.g., a key to open a door), simple probing won't reveal their mechanics.

<!-- slide -->
### Option B: Active-Learning Transition Discovery
- **Mechanism:** The agent maintains a transition dataset of `(state, action, next_state)`. It prioritizes actions that lead to the most "surprising" state updates (highest prediction error).
- **Analysis:** Promotes exploring boundaries, hitting walls, collecting items, and stepping into new regions to compile a comprehensive state-action library.
- **Pros:** Highly robust; naturally uncovers hidden mechanics (teleporters, traps).
- **Cons:** Higher step cost; can consume significant action budget before achieving goal mapping.

<!-- slide -->
### Option C: Local Vision LLM Hypothesizer
- **Mechanism:** A lightweight vision model (e.g., Qwen2.5-VL-7B running on GPU) takes the visual grid and writes natural language hypotheses of action mappings.
- **Analysis:** The LLM suggests specific test actions to confirm/refute hypotheses based on standard game conventions (e.g., "Press ACTION5 to pick up the green block").
- **Pros:** Incorporates strong prior knowledge about game UI and layout conventions.
- **Cons:** Very slow inference; risk of hallucination; offline resources are constrained.
```

---

## 2. Phase 2: Transition Modeling (The World Model)
*Goal: Represent the physics and mechanics of the grid so the agent can predict `next_grid = f(grid, action)` without executing it in the real game.*

### Option A: Sprite/Object Segmentation & Tracking (Recommended)
- **Algorithm:**
  1. Segment the grid into contiguous components (objects/sprites) using flood fill or connected-components labeling.
  2. Track each object's position, bounding box, color distribution, and velocity across frames.
  3. Formulate rules based on object intersections (e.g., "If Player overlaps with Yellow Block, Yellow Block moves with Player's velocity" -> pushing mechanic).
- **Pros:** Very natural mapping to standard ARC-AGI structures; highly interpretable; extremely fast execution.
- **Cons:** Fails if the game uses global rules like cellular automata (e.g., Conway's Life or maze fills) rather than object-based mechanics.

### Option B: Program Synthesis & Domain Specific Languages (DSL)
- **Algorithm:**
  1. Define a DSL representing grid operations (e.g., `shift_grid`, `fill_region`, `rotate_sprite`, `path_find`).
  2. Perform search (e.g., genetic algorithms or top-down enumeration) to find a DSL expression that perfectly maps `(grid, action) -> next_grid` for all collected transition pairs.
- **Pros:** Can model complex procedural transitions; guarantees exact correctness once synthesized.
- **Cons:** Search space grows exponentially; synthesizing models online during the 9-hour limit is difficult.

### Option C: Online-Trained Convolutional Transition Nets
- **Algorithm:**
  1. Train a small, lightweight fully convolutional network (FCN) or ResNet online.
  2. The input is the 3D grid tensor + one-hot action vector; the output is the predicted delta-grid.
- **Pros:** Handles arbitrary grid dynamics, cellular automata, and visual shifts without requiring object definitions.
- **Cons:** Requires significant training data (more steps than the action limit allows); vulnerable to overfitting on small state samples.

---

## 3. Phase 3: Goal & Reward Inference
*Goal: Determine what the level wants the agent to accomplish (e.g., escape, match a pattern, clear the board).*

```carousel
### Option A: Static Target Alignment
- **Method:** Deduce the target state by comparing the current grid layers with reference layers or specific static patterns.
- **Example:** If a static layer contains a distinct shape outline and another layer contains the movable shape, the goal is to align the shape with the outline.
- **Implementation:** Calculate the Hamming distance or intersection-over-union (IoU) of target colored blocks.

<!-- slide -->
### Option B: Level Progression Backpropagation
- **Method:** Treat the environment's `levels_completed` and `state` indicators as a reinforcement learning reward.
- **Example:** Whenever `levels_completed` increases or the state changes to `WIN`, backpropagate the successful action path to identify which state characteristics trigger a level clear.
- **Implementation:** Q-learning or path regression over key features (e.g., player position, key count, box locations).

<!-- slide -->
### Option C: Novelty & Density Maximization
- **Method:** Pursue the removal of specific color values (clearing targets) or alignment of heterogeneous blocks.
- **Example:** If the game starts with many scattered red blocks and a moving green block, and hitting red blocks deletes them, the goal is to clear all red blocks.
- **Implementation:** Track the entropy or counts of non-background colors in the grid.
```

---

## 4. Phase 4: Planning & Execution (Pathfinding & Optimization)
*Goal: Find the shortest sequence of actions to reach the goal, maximizing our RHAE score.*

```
                     [Transition Model f(s, a)]
                                 │
                                 ▼
                     [Search / Planning Engine]
                    /            │             \
                   /             │              \
                  ▼              ▼               ▼
           [Breadth-First]     [A* Search]     [Monte Carlo]
               (BFS)          (Heuristics)     (MCTS / Rollouts)
```

### Option A: Breadth-First / Dijkstra Search (Exact)
- **Use Case:** Best when the state space is relatively small (e.g., grid sizes < 32x32, fewer than 5 moving parts).
- **Execution:** We build a queue of states, apply our inferred transition model to generate neighbors, and stop once we find the goal state.
- **Pros:** Guaranteed to find the optimal path, maximizing our RHAE score.

### Option B: A* Search with Inferred Heuristics
- **Use Case:** Necessary for larger grids or complex pathfinding tasks (e.g., mazes, pushing boxes).
- **Heuristic Choices:**
  - Manhatten distance from player to target.
  - Number of remaining targets.
  - Grid connectivity / distance fields.
- **Pros:** Drastically reduces search time, allowing deep planning online.

### Option C: Monte Carlo Tree Search (MCTS)
- **Use Case:** Complex levels with high branching factors, stochastic elements, or long-horizon tasks.
- **Execution:** Run rollouts using our transition model combined with a simple default exploration policy.
- **Pros:** Excels in game environments; handles large branching factors well.

---

## 5. Winning Architectural Blueprint: Hybrid Search Agent

A robust, winning agent structure that runs completely offline within the Kaggle time budget:

1. **Initial Probing (Steps 1–5):** Detect standard controls (Move North, South, East, West) and track the player sprite.
2. **Object & Obstacle Mapping:** Segment the grid into:
   - **Static obstacles** (collidable walls, borders).
   - **Dynamic entities** (keys, player, boxes).
   - **Interactive triggers** (doors, teleporters, checkpoints).
3. **Transition Model Synthesizer:** Create state update rules based on bounding box collisions and movements.
4. **Planning Engine:**
   - Run a fast **BFS/A* search** in memory using the compiled transition rules.
   - If a plan is found, execute it immediately.
   - If the plan fails or the state deviates, trigger a **RESET** and refine the transition model with the new data.
