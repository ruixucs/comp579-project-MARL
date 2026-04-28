# Experiment Instructions — Probabilistic Adversary

Detailed run-book for the COMP579 step-level probabilistic adversary
experiment. The 9-cell grid is `attack_prob ∈ {0.2, 0.5, 0.8}` × `seed ∈ {1, 2, 3}`.

## Prerequisites

1. CUDA-capable GPU (training runs use the original EIR-MAPPO settings:
   ~5M env steps for SMAC).
2. Upstream EIR-MAPPO env setup (PyTorch, StarCraft II for SMAC, LBF, etc.).
   See [README_EIR_MAPPO.md](../README_EIR_MAPPO.md) and the parent
   [HARL README](https://github.com/PKU-MARL/HARL).
3. `pip install tensorboardX tensorboard matplotlib pytest`

## Sanity check first (no GPU needed)

```bash
pytest tests/test_attack_prob_logic.py -v
```

Should report 8 tests passing. This validates the per-step mask math used
inside `runner.run()` and `runner._eval_adv()`.

## Train the 9 models

### SMAC (4m vs 3m, ~5M steps)

```bash
nohup bash scripts/run_attack_prob_training.sh smac 4m_vs_3m 5000000 > train.log 2>&1 &
tail -f train.log
```

Expect ~1-2 days on a single GPU. To run a faster preview, drop steps and
seeds:

```bash
# inside scripts/run_attack_prob_training.sh, set SEEDS=(1) for one-seed dry-run
bash scripts/run_attack_prob_training.sh smac 4m_vs_3m 1000000
```

### LBF (12x12-4p-3f)

```bash
bash scripts/run_attack_prob_training.sh lbforaging '' 5000000
```

### Toy (fast, CPU-friendly)

```bash
bash scripts/run_attack_prob_training.sh toy '' 2000000
```

## Verify each run captured `attack_prob`

After any single run completes, confirm `config.json` has the right value:

```bash
python -c "
import json
for p in [0.2, 0.5, 0.8]:
    cfg = json.load(open(f'eir_mappo/results/smac/4m_vs_3m/mappo_advt_belief/attack_prob_{p}/1/run1/config.json'))
    print(f'attack_prob_{p}:', cfg['algo_args']['algo']['attack_prob'])
"
```

Expected:

```
attack_prob_0.2: 0.2
attack_prob_0.5: 0.5
attack_prob_0.8: 0.8
```

If the printed value is `1.0` instead, the `--attack_prob` CLI override did
not propagate. Check
[`eir_mappo/util/args_util.py:39`](../eir_mappo/util/args_util.py#L39) — its
`update_dict` only updates keys that already exist, so the YAML must contain
`attack_prob` (we added it in
[`mappo_advt_belief.yaml:75`](../eir_mappo/configs/algo/mappo_advt_belief.yaml#L75)).

Also check that `progress.txt` has at least one `[attack_prob]` line:

```bash
grep attack_prob eir_mappo/results/smac/4m_vs_3m/mappo_advt_belief/attack_prob_0.5/1/run1/progress.txt | head -3
```

Expected: lines like `[attack_prob] step=... adv_id=... attack_prob=0.500 mean_return=...`.

## Aggregate

```bash
python scripts/aggregate_attack_prob_results.py \
  --env smac --map 4m_vs_3m \
  --out-dir analysis/attack_prob_smac_4m_vs_3m
```

Produces:

| File | Contents |
|------|----------|
| `analysis/attack_prob_smac_4m_vs_3m/attack_prob_summary.csv` | One row per run: `run_dir, seed, attack_prob, metric_tag, metric_value`. |
| `analysis/attack_prob_smac_4m_vs_3m/attack_prob_curve.png` | mean ± std final-eval return vs `attack_prob`, with `n=k` annotations. |

### TensorBoard tag fallback

The aggregation script tries these tags in order:

1. `env/eval_return_mean`
2. `env/eval_average_episode_rewards`
3. `env/eval_episode_reward`
4. `env/return_mean`
5. `env/episode_reward`

If none match, it prints the diagnostic command to list the actual tags. Use
`--tag <name>` to override.

## Interpretation

The training-side change makes the defender see attacks at random intervals
during training rollouts. Expected qualitative outcomes:

- `attack_prob=0.2`: very sparse betrayals → belief network has weak signal →
  defender may underfit attacks → eval return under attack lower.
- `attack_prob=0.5`: balanced.
- `attack_prob=0.8`: nearly continuous attacks → defender becomes very
  defensive → may sacrifice cooperation when no attack happens.

The shape of the curve depends on the environment. Non-monotonic curves are
the interesting finding — they would show that intermittent attackers are
genuinely harder than always-attacking ones, supporting the Bayesian-game
framing of the original paper.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| All `attack_prob` values in CSV are `1.0`. | YAML key missing or CLI not propagating. | Check yaml has `attack_prob:` line. |
| `[attack_prob]` line never appears in `progress.txt`. | `_eval_adv` early-returned (e.g., `state_adversary=True`). | Inspect `runner.run()` eval branch. |
| `metric_value` column is all `NaN`. | None of the candidate TB tags matched. | Use `--tag` flag or print TB tags via the diagnostic command. |
| Plot has only one or two points. | Some runs crashed mid-training. | Check `train.log` for stack traces; rerun the failing combinations. |
| Pytest fails. | Numpy version mismatch. | `pip install -U numpy`. |
