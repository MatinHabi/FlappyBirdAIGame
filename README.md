# Flappy Bird DQN

A from-scratch Flappy Bird environment paired with a Double-DQN agent
that learns to clear six levels of increasing difficulty — from "the
sky is the limit" to 40-pixel-wide pipes spawning every 1.2 seconds.

## Highlights

- **Custom environment** (`flappy_env.py`): bird physics, level-driven
  pipe generation, optional pygame rendering. Headless mode uses a
  deterministic synthetic clock so training is reproducible.
- **Double DQN** with batched on-the-fly target computation — no stale
  cached `q_next` values.
- **Dual replay buffer**: a large FIFO `storage` for all transitions
  plus a small `event_buffer` that only keeps deaths and pipe-pass
  moments. Every minibatch is 75 % main + 25 % events so the agent
  never forgets how to die.
- **Gap-aware Gaussian centring reward** — the bell width of the
  centring bonus scales with the pipe's gap height. Same shaping
  function works for L1 (huge gap, sloppy ok) and L5 (thread the
  needle).
- **Honest periodic eval** — every 1000 training episodes the agent
  pauses and plays 30 pure-greedy episodes per level. `sec_model.ckpt`
  is only overwritten when the worst-level mean improves.
- **Live training dashboard** — a matplotlib window with four panels
  (per-level averages, ε decay, running average, eval history) that
  refreshes every 50 episodes.

## Files

| File | Purpose |
|---|---|
| `flappy_env.py` | Gym-style Flappy Bird environment. |
| `flappy_clock.py` | Tiny clock wrapper — supports rendered and headless modes. |
| `mlp_regressor.py` | The Q-network (3-hidden-layer MLP) and its masked-MSE training step. |
| `config.yml` | Screen / physics / colours / six level definitions. |
| `sec_model.py` | The DQN agent + training driver. |
| `make_bird_image.py` | One-off helper that creates a placeholder bird sprite. |

## Install

```bash
pip install numpy pygame torch gymnasium pyyaml colorama matplotlib
```

## Run

Generate the placeholder sprite (optional — env falls back to a yellow
rectangle):

```bash
python make_bird_image.py
```

Train on all six levels with the live dashboard:

```bash
python sec_model.py --genTrain --csv
```

Train on a single level:

```bash
python sec_model.py --level 5 --csv
```

Watch a trained checkpoint play:

```bash
python sec_model.py --eval --level 5 --episodes 10
```

## Levels

Difficulty progresses along a **single axis** — gap size shrinks and the
gap-centre's vertical range grows. Everything else (pipe width, spawn
cadence, action rate, formation, game_length) is held constant across
all six levels so a single policy can in principle be optimal on all of
them.

| Level | Gap (px) | Y-offset | Notes |
|---|---|---|---|
| L1 | 300 | 60  | Warm-up — wide gap, near-centred. |
| L2 | 260 | 90  | |
| L3 | 230 | 110 | |
| L4 | 210 | 130 | |
| L5 | 190 | 150 | |
| L6 | 170 | 170 | Tightest — gap ≈ 5 bird-heights, full vertical range. |

All levels share: `pipe_width=60`, `pipe_frequency=1800ms`,
`formation=random`, `minimum_action_gap=2`, `game_length=10`.

## Outputs

Training writes:

- **`sec_model.ckpt`** — eval-verified best weights (the gold copy).
- **`sec_model_running.ckpt`** — running training-best, useful for
  `--load` resume runs.
- **`fun_train.csv`** — one row per training episode (when `--csv`).
- **`fun_eval.csv`** — one row per eval checkpoint (when `--csv`).

## State representation (6-dim)

The agent sees only six numbers per frame:

| Slot | Feature | Range |
|---|---|---|
| 0 | `bird_y / screen_h`              | 0 .. 1 |
| 1 | `bird_velocity / 10`             | clipped to [-1, 1] |
| 2 | `dist_to_pipe1 / screen_w`       | 0 .. 1 (1 = no pipe yet) |
| 3 | `(bird_cy − gap_centre_1) / h`   | roughly -1 .. 1 |
| 4 | `(bird_cy − gap_centre_2) / h`   | roughly -1 .. 1 |
| 5 | `gap_size_1 / screen_h`          | 0.25 .. 1.0 |

`gap_size` is the killer feature — it lets the network distinguish
"any altitude is fine" (L1) from "thread the needle" (L5) without
having to compromise to a one-size-fits-all policy.
