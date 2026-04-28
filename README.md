# COMP579 Project — Probabilistic Adversary in Cooperative MARL

This repository extends [EIR-MAPPO](https://github.com/DIG-Beihang/EIR-MAPPO) (ICLR 2024:
*Byzantine Robust Cooperative Multi-Agent Reinforcement Learning as a Bayesian Game*)
with a **step-level probabilistic adversary** for the COMP579 course project.

---

## 1. What I did

### 1.1 The change in one sentence

Original EIR-MAPPO models the adversary at the **per-episode** level: at the start of each
episode, a coin flip (`adv_prob = 0.5`) decides whether an agent becomes the adversary for
the *entire* episode — heads = every step is malicious, tails = every step is clean.

I changed it to **per-step**: the adversary's identity is still picked per episode, but at
**each timestep** an additional coin (`attack_prob ∈ {0.2, 0.5, 0.8}`) decides whether the
adversary actually fires its adversarial action this step or imitates a normal teammate.
This is meant to model a more realistic intermittent / lurking attacker that is silent
most of the time and strikes selectively.

### 1.2 Code-level summary

| File | Change |
|------|--------|
| [`eir_mappo/configs/algo/mappo_advt_belief.yaml`](eir_mappo/configs/algo/mappo_advt_belief.yaml) | New key `attack_prob: 1.0` (default = legacy behavior). |
| [`eir_mappo/configs/algo/mappo_traitor_belief.yaml`](eir_mappo/configs/algo/mappo_traitor_belief.yaml) | New key `attack_prob: 1.0`. |
| [`eir_mappo/runner/on_policy_ma_runner_advt_with_belief.py`](eir_mappo/runner/on_policy_ma_runner_advt_with_belief.py) | (a) `__init__` reads `attack_prob` and asserts it ∈ [0,1]; (b) training rollout combines `episode_adversary` with a per-step `step_attack` mask; (c) `_eval_adv` applies the same per-step masking and calls a new logger hook. |
| [`eir_mappo/common/base_logger.py`](eir_mappo/common/base_logger.py) | New method `log_attack_prob(adv_id, attack_prob, mean_return)` writes `[attack_prob] ...` lines to `progress.txt` and adds TensorBoard scalars. |
| [`scripts/run_attack_prob_training.sh`](scripts/run_attack_prob_training.sh) | Batch trainer for `{0.2, 0.5, 0.8} × {seed1,2,3}`. |
| [`scripts/aggregate_attack_prob_results.py`](scripts/aggregate_attack_prob_results.py) | Walks `eir_mappo/results/.../attack_prob_*/`, reads `config.json` + TensorBoard events, emits CSV + matplotlib curve. |
| [`tests/test_attack_prob_logic.py`](tests/test_attack_prob_logic.py) | Pure-NumPy unit tests of the mask math. |
| [`docs/experiment-instructions.md`](docs/experiment-instructions.md) | Step-by-step run instructions. |
| `README_EIR_MAPPO.md` | Original EIR-MAPPO README, preserved verbatim. |

The mask math:

```python
# In runner.run() training rollout, every step:
step_attack = (np.random.rand(self.n_rollout_threads) < self.attack_prob)
attack_mask = self.episode_adversary & step_attack
input_actions[attack_mask, self.agent_adversary] = adv_actions[attack_mask, self.agent_adversary]

# In runner._eval_adv() per evaluation step:
step_attack = (np.random.rand(n_eval_threads) < self.attack_prob)
eval_actions[step_attack, adv_id] = eval_adv_actions[step_attack, adv_id]
```

`attack_prob = 1.0` exactly reproduces upstream behavior, so all original EIR-MAPPO
experiments still work unchanged.

### 1.3 Experiment plan

Train at `attack_prob ∈ {0.2, 0.5, 0.8}` × `seed ∈ {1, 2, 3}` = 9 runs. Aggregate the final
evaluation return into a single curve of *return vs. attack probability*.

We run on **LBF (Level-Based Foraging)** rather than SMAC. Reasons:

- **Compute**: LBF is CPU-friendly and ~3-5× faster than SMAC for the same step count, so
  the cross-evaluation matrix in §3 is actually feasible.
- **Sparse cooperation reward**: LBF requires multiple agents to coordinate to harvest food
  (their levels must sum to the food's level). A single adversarial defection at a critical
  moment is often enough to kill the entire team's reward, which means the difference
  between `attack_prob = 0.2`, `0.5`, and `0.8` is amplified — the curve is more readable.
- **Precedent**: The original EIR-MAPPO paper's video comparisons use LBF `12x12-4p-3f-c`,
  so running on LBF stays close to upstream.

---

## 2. How to run

> Setup: follow upstream env setup (PyTorch, the relevant env packages) per
> [README_EIR_MAPPO.md](README_EIR_MAPPO.md) and the parent
> [HARL repo](https://github.com/PKU-MARL/HARL). The new code adds **no new
> Python deps beyond NumPy + TensorBoard + matplotlib**.

### 2.1 (Optional) Pretrain a victim defender

If you don't already have an `mappo_advt_belief` checkpoint:

```bash
python -u train.py --alg mappo_advt_belief --env lbforaging \
  --exp_name baseline --seed 1
```

### 2.2 Run the COMP579 batch experiment

```bash
bash scripts/run_attack_prob_training.sh lbforaging '' 5000000
```

This trains 9 models — 3 attack probabilities × 3 seeds — under
`eir_mappo/results/lbforaging/12x12-4p-3f/mappo_advt_belief/attack_prob_{0.2,0.5,0.8}/{1,2,3}/run1/`.

For SMAC or toy:

```bash
bash scripts/run_attack_prob_training.sh smac 4m_vs_3m 5000000
bash scripts/run_attack_prob_training.sh toy '' 2000000
```

### 2.3 Aggregate results

```bash
python scripts/aggregate_attack_prob_results.py \
  --env lbforaging --out-dir analysis/attack_prob_lbf
```

Outputs:
- `analysis/attack_prob_lbf/attack_prob_summary.csv` — one row per run
- `analysis/attack_prob_lbf/attack_prob_curve.png` — mean ± std return vs. `attack_prob`

If the script reports "none of the candidate tags matched", inspect TensorBoard tags via
the diagnostic command it prints, then rerun with `--tag <correct_tag>`.

### 2.4 Sanity-check the mask logic (no GPU needed)

```bash
pytest tests/test_attack_prob_logic.py -v
```

Should report 8 tests passing.

---

## 3. Critical self-review — three known limitations

After implementing the change above, I went back and reviewed my own diff. The result is
not as harmless as it looks. Three issues will make any results harder to interpret unless
they are addressed. Listed by severity.

### 3.1 Limitation A — belief network's supervision signal gets corrupted

EIR-MAPPO's core mechanism is the **belief network**: it has to figure out *which teammate
is the adversary* from observed behavior, trained as a supervised classifier whose label
is `ground_truth_type` (a one-hot pointer to the adversary's index).

Under `attack_prob = 0.2`, the adversary takes the **same normal action as its teammates**
on 80% of steps (because `share_param=True` — when the adversary is not firing, it draws
from the same actor as everyone else). So the belief network sees 80% of "adversary label
+ normal behavior" training samples — that is severe label noise, with a noise rate of
80%.

Information-theoretically, the Bayes-optimal accuracy of a binary classifier under noise
rate `1 - p` is bounded around `0.5 + 0.5·p`. At `p = 0.2` the upper bound is ~0.6, barely
above random guessing.

**Why this matters in practice**: the belief output is concatenated back into the actor's
and critic's observations. If belief collapses to noise, the entire defense mechanism
collapses with it. A "lower `attack_prob` → worse defense" curve coming out of this setup
is **not necessarily evidence that intermittent attacks are harder to defend against** — it
might just be evidence that the belief network's training data is broken. These two
interpretations are radically different research-wise.

**Fix sketch (≈10 lines)**: only compute belief loss on steps where `step_attack = True`.
Optionally enrich the belief input with the critic's per-agent advantage as a denser
signal to compensate for sparser supervision.

### 3.2 Limitation B — adversary's training objective is misaligned

The adversary is trained with PPO to maximize damage to the team. The problem:

- Its PPO loss computes gradients across all 200 steps of an episode.
- But the environment actually received the adversary's action only on roughly
  `attack_prob × 200` steps.
- On the remaining steps, the adversary's chosen action got dropped — the environment ran
  the normal actor's action instead.

So the advantage `A_adv(s_t, a_adv_t)` on the non-fired steps **does not reflect the
consequences of the adversary's action** — it reflects what happened when the actor's
normal action was taken. PPO still uses this advantage to update the adversary, which is
effectively saying:

> "You wanted to do this action, we ignored you, but here's a reward update as if we'd
> taken it."

This breaks PPO's on-policy assumption. The adversary **never learns timing** — it can't
distinguish "I attacked successfully" from "I didn't attack but got credited for the
actor's action". The optimal policy degenerates to "output the action that's worst on
average across all states" instead of "pick the right state to strike at".

**Fix sketch (1 line)**: in the adversary's PPO loss, mask out the non-fired steps:
`adv_active_masks[~step_attack] = 0`. This is half-correct: it stops the false credit
assignment, but the adversary still cannot proactively learn *when* to attack (that would
need a hierarchical adversary with its own attack/no-attack head).

### 3.3 Limitation C — train and eval probabilities are coupled

My 9 runs all use `attack_prob_train == attack_prob_eval` — each model is only ever
evaluated at the same attack rate it was trained at.

**Issue**: this measures **task difficulty** (more attacks → harder, obviously), not
**robustness** (whether the defender generalizes to attack rates it has not seen during
training). To attribute findings to "robustness under intermittent attack" rather than
just "tasks are harder when there is more attack", we need cross-evaluation.

**Note on responsibility**: this limitation is *partially* something I introduced. The
original EIR-MAPPO had a related blind spot — its training uses `adv_prob = 0.5` (some
episodes have an adversary, some don't) but its evaluation has the adversary fire on
every step (`adv_prob = 1.0` effectively, hardcoded). That is one off-diagonal point, not
a robustness curve. By adding a single `attack_prob` parameter that is read by both
`runner.run()` and `_eval_adv`, my version made the coupling stricter — train and eval
now use exactly the same probability. The fix is the same either way.

For reference, robust-RL papers like Pinto et al. 2017 (*Robust Adversarial RL*),
Tessler et al. 2019 (*Action Robust RL*), and the original EIR-MAPPO paper's Table 2 all
use cross-evaluation matrices when arguing about robustness.

**Fix sketch (≈30 lines)**: split `attack_prob` into `attack_prob_train` and
`attack_prob_eval` as independent parameters. Add an eval-only script that, for each
trained model, runs evaluation at multiple `attack_prob_eval` values (e.g.,
`{0.0, 0.2, 0.5, 0.8, 1.0}`) without further training. This produces a 5×5 matrix instead
of a 3-point diagonal.

### 3.4 How the three interact

```
A (belief noise) ────────► defender decisions degrade ─┐
                                                       │
B (adversary misaligned) ► attack difficulty understated ─┼──► eval return becomes hard to interpret;
                                                       │       A and B confound any apparent C result
C (train=eval coupled) ──► no robustness curve visible ─┘
```

Fixing C alone gives a clean robustness matrix, but every cell is still polluted by A and
B. Fixing A or B alone improves one piece of the pipeline but the experiment still measures
"task difficulty", not robustness. The three need to be addressed together (or at least in
the order C → A → B) for results to support a defensible claim.

### 3.5 Priority

| | Fix | Cost | Why first/next/last |
|--|------|------|---------------------|
| 1 | **C** — split train/eval `attack_prob`, add cross-eval script | ~30 lines | Pure infrastructure; doesn't touch the algorithm; necessary for any subsequent fix to be measurable. |
| 2 | **A** — belief loss only on fired steps (+ optional advantage input) | ~10 lines | Restores supervised-signal validity. Required for the cross-eval matrix to mean anything. |
| 3 | **B** — PPO mask on non-fired steps for adversary | 1 line | Theoretical clean-up. Improves adversary quality but improvements only show up clearly once A and C are in place. |

---

## 4. Design notes

- **Identity vs behavior.** Upstream `adv_prob` controls **who is adversary** per episode;
  new `attack_prob` controls **whether the adversary actually attacks** at each step.
  They are orthogonal.
- **Belief network.** `ground_truth_type` is unchanged in the current diff: belief is still
  trained to label the adversary's identity. Limitation A above is precisely about this
  choice and how it interacts with the new sparsity.
- **Reproducibility.** `save_config` already serializes the active `algo_args` to
  `run_dir/config.json`, including `attack_prob`. The aggregation script reads
  `config.json` as the source of truth for each run's configured probability, never
  trusting the directory name alone.

---

## 5. Citation

If you use this code, please cite the original EIR-MAPPO paper:

```
@inproceedings{yuan2024byzantine,
  title={Byzantine Robust Cooperative Multi-Agent Reinforcement Learning as a Bayesian Game},
  author={Yuan, Yi and Zhang, Yunbo and ...},
  booktitle={ICLR},
  year={2024}
}
```

## 6. License

Inherits the upstream EIR-MAPPO license.
