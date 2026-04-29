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
| [`eir_mappo/runner/on_policy_ma_runner_advt_with_belief.py`](eir_mappo/runner/on_policy_ma_runner_advt_with_belief.py) | (a) `__init__` reads `attack_prob` and asserts it ∈ [0,1]; (b) training rollout combines `episode_adversary` with a per-step `step_attack` mask; (c) `insert()` applies the §5.6 mask `adv_active_masks[~step_attack]=0` so the adversary's PPO is on-policy; (d) `step_attack` is forwarded to the buffer for use in the §5.7 belief loss; (e) `_eval_adv` applies the same per-step masking and calls a new logger hook. |
| [`eir_mappo/common/actor_buffer_advt_with_belief.py`](eir_mappo/common/actor_buffer_advt_with_belief.py) | Stores per-step `step_attack` (`(episode_length, n_rollout_threads, 1)`); `recurrent_generator_belief` returns it as the 16th batch field. |
| [`eir_mappo/algo/mappo_advt_with_belief.py`](eir_mappo/algo/mappo_advt_with_belief.py) | §5.7: belief BCE loss is multiplied elementwise by `step_attack_batch` and renormalized (`step_attack_batch.sum() * num_agents`) so non-fired steps stop contributing supervision. The denominator is chosen so `attack_prob = 1.0` is bit-equivalent to upstream `loss.mean()`. |
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

## 3. Critical self-review — limitations

After implementing the per-step override I reviewed my own diff and found three issues
that would have made results hard to interpret. **A and B are now fixed in code; C
remains a limitation acknowledged in this section.**

### 3.1 Limitation A — belief network's supervision signal gets corrupted (FIXED)

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

**Status: fixed.** See §5.7 derivation and the implemented mask in
[`mappo_advt_with_belief.update_belief`](eir_mappo/algo/mappo_advt_with_belief.py): the
BCE loss is now multiplied elementwise by `step_attack_batch` and renormalized by
`step_attack_batch.sum() * num_agents`, so non-fired steps no longer enter the
supervised signal. The denominator is chosen so that `attack_prob = 1.0` recovers
`loss.mean()` bit-equivalently. Belief inputs are unchanged; the optional advantage
enrichment was deferred as a future direction.

### 3.2 Limitation B — adversary's training objective is misaligned (FIXED)

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

**Status: fixed (partial).** Implemented in
[`runner.insert()`](eir_mappo/runner/on_policy_ma_runner_advt_with_belief.py) as
`adv_active_masks[~step_attack] = 0` (see §5.6). This stops the false credit
assignment so PPO is on-policy with respect to the realized actions. It does **not**
let the adversary proactively learn *when* to attack — that would require a
hierarchical adversary with its own attack/no-attack head, which is out of scope for
this fork.

### 3.3 Limitation C — train and eval probabilities are coupled (NOT FIXED)

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
A (belief noise, FIXED) ─────► defender decisions are now trained on informative steps only
B (adversary misaligned, FIXED) ► PPO is on-policy w.r.t. realized actions
C (train=eval coupled, OPEN) ─► no robustness curve visible — measures task difficulty,
                                  not generalization across attack rates
```

With A and B addressed, each cell of the resulting *return-vs-attack-prob* curve is
internally consistent. The remaining caveat is exactly C: each cell still uses
`attack_prob_train == attack_prob_eval`, so the curve still measures task difficulty
under a matched attacker, not robustness to *unseen* attack rates. A cross-evaluation
matrix would be the cleanest way to disentangle these; that work is left to a follow-up.

### 3.5 Priority and status

| | Fix | Cost | Status |
|--|------|------|--------|
| 1 | **A** — belief BCE loss only on fired steps | ~50 lines (buffer plumbing + algo) | **Done** — see §5.7 / `update_belief` |
| 2 | **B** — PPO active-mask on non-fired steps for adversary | 1 line | **Done** — see §5.6 / `runner.insert` |
| 3 | **C** — split `attack_prob_train` / `attack_prob_eval`, add cross-eval script | ~30 lines | **Open** — acknowledged limitation |

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

## 5. Mathematical derivation — extending the BARDec-POMDP equations to step-level attack

This section grounds the proposed fixes for limitations A and B in the actual
equations of the original paper, so that the modifications are not ad-hoc but
follow from re-deriving the policy-gradient and belief-loss expressions under
the new step-level threat model.

### 5.1 Notation from the original paper

Following Li et al. (ICLR 2024), Section 3.2 and Appendix A.5:

| Symbol | Meaning |
|---|---|
| $\theta_i \in \\{0, 1\\}$ | type indicator; $\theta_i = 1$ if agent $i$ is the adversary, $0$ otherwise. **Sampled per episode** by nature. |
| $\pi_\varphi$ | defender policy (parameters $\varphi$) |
| $\hat{\pi}_{\hat{\varphi}}$ | adversary policy (parameters $\hat{\varphi}$) |
| $b^i$ | agent $i$'s belief over types |
| $Q^i(s, a, b^i)$ | belief-conditioned action-value function |
| $H^i$ | observation history of agent $i$ |
| $\rho_\pi(s)$ | state visitation distribution |

### 5.2 Original equations (verbatim from the paper)

**Mixed policy** (paper, Appendix A.5, line 1706):

$$
\pi_{\varphi, \hat{\varphi}}(a \mid H, b, \theta) \;=\; (1 - \theta) \cdot \pi_\varphi(a \mid H, b) \;+\; \theta \cdot \hat{\pi}_{\hat{\varphi}}(\hat{a} \mid H, \theta)
$$

**Eq. (8) — defender policy gradient**:

$$
\nabla_{\varphi^i} J^i(\varphi^i) \;=\; \mathbb{E}_{s \sim \rho_\pi,\, a \sim \pi}\!\left[ (1 - \theta^i)\,\nabla \log \pi_\varphi^i(a^i \mid H^i, b^i)\,Q^i(s, a, b^i) \right]
$$

**Eq. (9) — adversary policy gradient**:

$$
\nabla_{\hat{\varphi}^i} J^i(\hat{\varphi}^i) \;=\; \mathbb{E}_{s \sim \rho_\pi,\, a \sim \pi}\!\left[ -\,\theta^i\,\nabla \log \hat{\pi}_{\hat{\varphi}}^i(\hat{a}^i \mid H^i, \theta)\,Q^i(s, a, b^i) \right]
$$

**Eq. (11) — belief network BCE loss**:

$$
\min_\xi \;\; -\,\theta\,\log p_\xi(\theta \mid H^i) \;-\; (1 - \theta)\,\log\!\bigl(1 - p_\xi(\theta \mid H^i)\bigr)
$$

In the original per-episode formulation $\theta_i$ plays three coupled roles
simultaneously: (i) it determines who the adversary is, (ii) it selects which
action distribution mixes into the trajectory, and (iii) it acts as a gradient
mask in (8) and (9). All three coincide because identity is constant within an
episode.

### 5.3 Adding step-level attack: the new variable $\zeta_t$

We introduce a fresh i.i.d. Bernoulli variable that decides, **at each timestep**,
whether the adversary actually fires:

$$
\zeta_t \in \\{0, 1\\}, \qquad \zeta_t \;\sim\; \text{Bernoulli}(p_{\text{attack}}), \qquad \zeta_t \perp\!\!\!\perp \zeta_{t'} \text{ for } t \neq t'.
$$

The identity variable $\theta_i$ keeps its original meaning (per-episode). What
changes is that the **effective adversarial-action mask** at timestep $t$ is now
the product

$$
\theta_i \cdot \zeta_t \;=\; \begin{cases} 1 & \text{agent } i \text{ is the adversary } \mathbf{and} \text{ it fires at step } t, \\\\ 0 & \text{otherwise.}\end{cases}
$$

This is exactly the boolean mask `attack_mask = self.episode_adversary & step_attack`
in [`runner.py`](eir_mappo/runner/on_policy_ma_runner_advt_with_belief.py).

### 5.4 Modified mixed policy

The action-distribution mixture becomes:

$$
\boxed{\;\pi_{\varphi, \hat{\varphi}}(a \mid H, b, \theta, \zeta_t) \;=\; (1 - \theta_i \zeta_t)\,\pi_\varphi(a \mid H, b) \;+\; \theta_i \zeta_t\,\hat{\pi}_{\hat{\varphi}}(\hat{a} \mid H, \theta)\;}
$$

This is the per-step generalization of paper line 1706 with $\theta \mapsto \theta_i \zeta_t$.

### 5.5 Modified Eq. (8) — defender gradient is **unchanged**

$$
\nabla_{\varphi^i} J^i(\varphi^i) \;=\; \mathbb{E}\!\left[ (1 - \theta^i)\,\nabla \log \pi_\varphi^i(a^i \mid H^i, b^i)\,Q^i(s, a, b^i) \right]
$$

The mask remains $(1 - \theta^i)$. Reasoning:

- If $\theta^i = 0$ (agent is a defender) — the defender always acts via $\pi_\varphi$, every timestep, regardless of $\zeta_t$. So gradient updates apply to all steps.
- If $\theta^i = 1$ (agent is the adversary) — we never want to update its defender head, regardless of $\zeta_t$.

So the mask depends only on identity, not on $\zeta_t$. **No code change needed
for the defender side.**

### 5.6 Modified Eq. (9) — adversary gradient gains a $\zeta_t$ factor

$$
\boxed{\;\nabla_{\hat{\varphi}^i} J^i(\hat{\varphi}^i) \;=\; \mathbb{E}\!\left[ -\,\theta^i\,\zeta_t\,\nabla \log \hat{\pi}_{\hat{\varphi}}^i(\hat{a}^i \mid H^i, \theta)\,Q^i(s, a, b^i) \right]\;}
$$

**Derivation sketch** (mirrors the paper's Appendix A.5, lines 1697–1769, with
$\theta \mapsto \theta_i \zeta_t$ in the mixture):

$$
\nabla_{\hat{\varphi}^i}\!\sum_{a \in A}\!\bigl[(1 - \theta_i \zeta_t)\pi_\varphi(a) + \theta_i \zeta_t\,\hat{\pi}_{\hat{\varphi}}(\hat{a})\bigr] Q^i(s, a, b^i)
\;=\; \sum_{a \in A} \theta_i \zeta_t \,\nabla_{\hat{\varphi}^i}\hat{\pi}_{\hat{\varphi}}(\hat{a})\,Q^i + (\text{recursive term})
$$

The first $(1 - \theta_i \zeta_t)\pi_\varphi$ term has zero gradient with respect to
$\hat{\varphi}^i$. Following the same log-derivative trick as in the paper
(line 1758), this becomes an expectation under $\rho_\pi$ with score function
$\nabla \log \hat{\pi}$ multiplied by $\theta_i \zeta_t$.

**Why this is the *correct* fix and not just a heuristic**: the policy-gradient
estimator is unbiased only when $a$ is actually drawn from $\hat{\pi}$. When
$\zeta_t = 0$ the realized action $a$ comes from $\pi_\varphi$, not from
$\hat{\pi}$, so including those samples breaks the importance-sampling identity.
Masking by $\zeta_t$ restricts the sum to events where $\hat{\pi}$ truly
generated the action, restoring unbiasedness.

**Code mapping**: now landed in [`runner.insert()`](eir_mappo/runner/on_policy_ma_runner_advt_with_belief.py).
`run()` forwards the per-step `step_attack` (a `(n_threads,)` boolean) to `insert()` via
the data tuple. `insert()` then composes the three masks:

```python
adv_active_masks[~self.episode_adversary] = 0
adv_active_masks[:, np.arange(self.num_agents) != self.agent_adversary] = 0
adv_active_masks[~step_attack] = 0          # the ζ_t mask
```

When `attack_prob = 1.0` the third line is a no-op (`step_attack` is all-True), so
upstream behavior is preserved exactly.

### 5.7 Modified Eq. (11) — belief BCE loss gains a $\zeta_t$ factor

$$
\boxed{\;\min_\xi \;\; \zeta_t \cdot \Bigl[\,-\,\theta\,\log p_\xi(\theta \mid H^i) \;-\; (1 - \theta)\,\log\!\bigl(1 - p_\xi(\theta \mid H^i)\bigr)\Bigr]\;}
$$

**Why**: the BCE objective treats observations $H^i$ as samples from a
type-conditional distribution $p(H^i \mid \theta)$. But on a step where
$\zeta_t = 0$, the action visible in $H^i$ is drawn from $\pi_\varphi$
**regardless of $\theta$**, so

$$
p(\text{observed action} \mid \theta_i = 1, \zeta_t = 0) \;=\; p(\text{observed action} \mid \theta_i = 0) \;=\; \pi_\varphi(a)
$$

The likelihood ratio is $1$ — the observation is *uninformative* about $\theta$.
Including it in the BCE loss is equivalent to teaching the network that "normal
behavior implies adversary identity," which corrupts the classifier under high
$1 - p_{\text{attack}}$. Masking by $\zeta_t$ removes uninformative samples and
restores valid posterior estimation.

**Code mapping** — landed in [`mappo_advt_with_belief.py:update_belief`](eir_mappo/algo/mappo_advt_with_belief.py).
The implemented form is the *backward-compatible* variant: when `attack_prob = 1.0`
(`ζ_t ≡ 1`) the masked loss is **bit-equivalent** to the upstream `loss.mean()`. This is
slightly more elaborate than the cartoon two-liner above because we must compensate the
denominator by the per-agent dimension to keep loss-per-sample units consistent:

```python
loss_raw = F.binary_cross_entropy(belief, ground_truth_type_batch, reduction='none')
step_attack_mask = step_attack_batch  # shape (B, 1)
masked_loss = loss_raw * step_attack_mask
last_dim = loss_raw.shape[-1]                       # = num_agents
denom = step_attack_mask.sum() * last_dim           # = numel when ζ_t ≡ 1
loss = masked_loss.sum() / (denom + 1e-8)
```

When `attack_prob = 1.0` the formula above is numerically identical to the upstream
`loss.mean()` (since the denominator becomes `numel`), so the legacy code path is
preserved. At `attack_prob < 1.0` the loss reduces to "average BCE over fired
(ζ_t = 1) steps", which is what eq. (5.7) prescribes.

`actor_buffer_advt_with_belief.py` now stores `step_attack` per step
(`(episode_length, n_rollout_threads, 1)`) and `recurrent_generator_belief` returns it
as the 16th batch field so the algo can multiply it into the BCE loss.

### 5.8 Backward compatibility

When $p_{\text{attack}} = 1$ we have $\zeta_t \equiv 1$ and every modified
equation reduces to its original form:

| Modified | $\zeta_t \equiv 1$ reduction | Original |
|---|---|---|
| $\theta_i \zeta_t$ | $\theta_i$ | original $\theta_i$ |
| $(1 - \theta_i \zeta_t)\pi_\varphi + \theta_i \zeta_t \hat{\pi}$ | $(1 - \theta_i)\pi_\varphi + \theta_i \hat{\pi}$ | line 1706 |
| Eq. (9) with $\zeta_t$ mask | Eq. (9) original | Eq. (9) |
| Eq. (11) with $\zeta_t$ mask | Eq. (11) original | Eq. (11) |

So the code path with `attack_prob = 1.0` is bit-equivalent to the upstream
EIR-MAPPO algorithm.

### 5.9 Caveat on convergence

The original paper's Theorem 3.1 (almost-sure convergence to an ex-interim
RMPBE) is stated for the per-episode formulation. Strictly speaking the
$\zeta_t$-masked variant requires a separate convergence argument. Intuitively,
masking by an i.i.d. Bernoulli does not break the stochastic-approximation
conditions used in the paper's proof (Borkar 1997, 2009): the masked gradients
still form an unbiased estimator with bounded variance, only the effective
sample rate per parameter update is reduced from $1$ to $p_{\text{attack}}$.
A formal extension of the proof is left to future work and acknowledged as a
limitation.

### 5.10 Summary table — equations to code

| Quantity | Paper location | Modification | Code site |
|---|---|---|---|
| Mixed policy | line 1706 | $\theta \to \theta_i \zeta_t$ | `runner.py` action-override (landed) |
| Eq. (8) defender gradient | line 386 | **unchanged** | n/a |
| Eq. (9) adversary gradient | line 390 | mask by $\zeta_t$ | `runner.insert()` `adv_active_masks[~step_attack]=0` (landed) |
| Eq. (11) belief BCE loss | line 423 | multiply by $\zeta_t$ | `mappo_advt_with_belief.update_belief` (landed, bit-equivalent at `attack_prob=1.0`) |

---

## 6. Citation

If you use this code, please cite the original EIR-MAPPO paper:

```bibtex
@inproceedings{li2024byzantine,
  title     = {Byzantine Robust Cooperative Multi-Agent Reinforcement Learning as a {B}ayesian Game},
  author    = {Li, Simin and Guo, Jun and Xiu, Jingqiao and Xu, Ruixiao and Yu, Xin and Wang, Jiakai and Liu, Aishan and Yang, Yaodong and Liu, Xianglong},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024}
}
```

## 7. License

Inherits the upstream EIR-MAPPO license.
