# COMP579 Project — Probabilistic Adversary in Cooperative MARL

This repository extends [EIR-MAPPO](https://github.com/DIG-Beihang/EIR-MAPPO) (ICLR 2024:
*Byzantine Robust Cooperative Multi-Agent Reinforcement Learning as a Bayesian Game*)
with a **step-level probabilistic adversary** for the COMP579 course project.

## TL;DR

Original EIR-MAPPO models adversaries with a **per-episode** probability `adv_prob`:
either an agent is the adversary for the entire episode, or it is not. We add a
**per-step** probability `attack_prob`: the adversary's *identity* is fixed (or
sampled per episode as before), but at every timestep it independently decides
whether to actually fire the adversarial action with probability `attack_prob`.

The COMP579 experiment trains a defender at three step-level attack rates and
compares robustness:

| Setting | `attack_prob` | Behavior |
|---------|---------------|----------|
| Sparse  | 0.2 | Adversary attacks only ~20% of steps; very stealthy. |
| Medium  | 0.5 | Coin-flip every step. |
| Aggressive | 0.8 | Attacks ~80% of steps; close to legacy `attack_prob=1.0`. |

Each setting × 3 seeds = 9 training runs, then we aggregate evaluation return
into a single `attack_prob` vs return curve.

## What changed vs. upstream EIR-MAPPO

| File | Change |
|------|--------|
| [`eir_mappo/configs/algo/mappo_advt_belief.yaml`](eir_mappo/configs/algo/mappo_advt_belief.yaml) | New key `attack_prob: 1.0` (default = legacy behavior). |
| [`eir_mappo/configs/algo/mappo_traitor_belief.yaml`](eir_mappo/configs/algo/mappo_traitor_belief.yaml) | New key `attack_prob: 1.0`. |
| [`eir_mappo/runner/on_policy_ma_runner_advt_with_belief.py`](eir_mappo/runner/on_policy_ma_runner_advt_with_belief.py) | (a) `__init__` reads `attack_prob` and asserts it ∈ [0,1]; (b) training rollout combines `episode_adversary` with a per-step `step_attack` mask; (c) `_eval_adv` does the same per-step masking and calls a new logger hook. |
| [`eir_mappo/common/base_logger.py`](eir_mappo/common/base_logger.py) | New method `log_attack_prob(adv_id, attack_prob, mean_return)` writes a `[attack_prob] ...` line to `progress.txt` and adds TensorBoard scalars. |
| [`scripts/run_attack_prob_training.sh`](scripts/run_attack_prob_training.sh) | New: batch trainer for `{0.2, 0.5, 0.8} × {seed1,2,3}`. |
| [`scripts/aggregate_attack_prob_results.py`](scripts/aggregate_attack_prob_results.py) | New: walks `eir_mappo/results/.../attack_prob_*/`, reads `config.json` + TensorBoard events, emits CSV + matplotlib curve. |
| [`tests/test_attack_prob_logic.py`](tests/test_attack_prob_logic.py) | New: pure-numpy unit tests of the mask math. |
| [`docs/experiment-instructions.md`](docs/experiment-instructions.md) | Step-by-step run instructions. |
| `README_EIR_MAPPO.md` | The original EIR-MAPPO README is preserved verbatim. |

### Mask math

The runner combines the existing per-episode mask with the new per-step mask:

```python
# In runner.run() training rollout (each step, n_threads independent samples):
step_attack = (np.random.rand(self.n_rollout_threads) < self.attack_prob)
attack_mask = self.episode_adversary & step_attack
input_actions[attack_mask, self.agent_adversary] = adv_actions[attack_mask, self.agent_adversary]

# In runner._eval_adv() per evaluation step:
step_attack = (np.random.rand(n_eval_threads) < self.attack_prob)
eval_actions[step_attack, adv_id] = eval_adv_actions[step_attack, adv_id]
```

`attack_prob = 1.0` exactly reproduces upstream behavior, so all original
EIR-MAPPO experiments still work unchanged.

## How to run

> Setup: follow upstream env setup (PyTorch, SC2, etc.) per
> [README_EIR_MAPPO.md](README_EIR_MAPPO.md) and the parent
> [HARL repo](https://github.com/PKU-MARL/HARL). The new code adds **no new
> Python deps beyond NumPy + TensorBoard + matplotlib**.

### 1) (Optional) Pretrain a victim defender

If you don't already have an `mappo_advt_belief` checkpoint:

```bash
python -u train.py --alg mappo_advt_belief --env smac \
  --map_name 4m_vs_3m --exp_name baseline --seed 1
```

### 2) Run the COMP579 batch experiment

```bash
bash scripts/run_attack_prob_training.sh smac 4m_vs_3m 5000000
```

This trains 9 models — 3 attack probabilities × 3 seeds — under
`eir_mappo/results/smac/4m_vs_3m/mappo_advt_belief/attack_prob_{0.2,0.5,0.8}/{1,2,3}/run1/`.

For toy:

```bash
bash scripts/run_attack_prob_training.sh toy '' 2000000
```

For LBF:

```bash
bash scripts/run_attack_prob_training.sh lbforaging '' 5000000
```

### 3) Aggregate results

```bash
python scripts/aggregate_attack_prob_results.py \
  --env smac --map 4m_vs_3m --out-dir analysis/attack_prob_smac
```

Outputs:
- `analysis/attack_prob_smac/attack_prob_summary.csv` — one row per run
- `analysis/attack_prob_smac/attack_prob_curve.png` — mean ± std return vs `attack_prob`

If the script reports "none of the candidate tags matched", inspect
TensorBoard tags via the diagnostic command it prints, then rerun with
`--tag <correct_tag>`.

### 4) Sanity-check the mask logic

```bash
pytest tests/test_attack_prob_logic.py -v
```

Should report 8 tests passing. No GPU / RL env required.

## Design notes

- **Identity vs behavior.** Upstream `adv_prob` controls **who is adversary**
  per episode; new `attack_prob` controls **whether the adversary actually
  attacks** at each step. They are orthogonal. Setting `adv_prob=1` and
  `attack_prob=p` gives "always-present, intermittently-attacking adversary",
  which matches the COMP579 threat model (adversary present but stealthy).
- **Belief network.** `ground_truth_type` is unchanged: belief is still trained
  to label the adversary's identity correctly. With `attack_prob<1`, the belief
  signal is sparser (the adversary is "betraying" only some steps), so this
  experiment tests how well the Bayesian-game belief module generalizes under
  intermittent betrayal.
- **Reproducibility.** `save_config` already serializes the active
  `algo_args` to `run_dir/config.json`, including `attack_prob`. The
  aggregation script reads `config.json` as the source of truth for each run's
  configured probability, never trusting the directory name alone.

## Citation

If you use this code, please cite the original EIR-MAPPO paper:

```
@inproceedings{yuan2024byzantine,
  title={Byzantine Robust Cooperative Multi-Agent Reinforcement Learning as a Bayesian Game},
  author={Yuan, Yi and Zhang, Yunbo and ...},
  booktitle={ICLR},
  year={2024}
}
```

## License

Inherits the upstream EIR-MAPPO license.
