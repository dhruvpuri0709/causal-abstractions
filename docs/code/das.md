# The DAS implementation (`src/cas/das.py`)

## Interchange intervention

For a hidden layer with post-ReLU activations h ∈ R^w and an orthogonal R (w × w):

```
r_b = h(base) R            r_i = h(source_i) R          (rotated coordinates)
r'  = r_b, but r'[i k : (i+1) k] = r_i[i k : (i+1) k]   for every intervened variable i
h'  = r' Rᵀ                                             (Rᵀ = R⁻¹)
logits = network layers above, applied to h'
```

Variable V1 is aligned with rotated dims [0, k), V2 with [k, 2k); k is the per-variable
subspace size (the paper's "intervention size"), so 2k ≤ w. With R = I this is an interchange of
neurons. This is the same computation as pyvene's `RotatedSpaceIntervention` (`rotated_base =
base @ W`, swap the selected columns from `source @ W`, `output = rotated_base @ W.T`) and as
Sutter et al.'s `Distributed_Alignment_Search_MLP` hook. A "both" intervention takes V1's dims
from source 1 and V2's dims from source 2, as in the pyvene tutorial with two linked
interventions. In code this is one masked expression, `das.interchange`.

**IIA** for a split is the share of its pairs where `argmax(logits) ==` the algorithm's
counterfactual label.

## Fitting

* `Rotation`: `nn.Linear(w, w, bias=False)` with `torch.nn.utils.parametrizations.orthogonal`
  (matrix-exponential map), initialised with `nn.init.orthogonal_` from a fixed seed.
* The network is frozen. Activations of every base and source input at the target layer are
  computed once (`pair_activations`); each step only evaluates R and the layers above it.
* Loss: cross-entropy between intervened logits and counterfactual labels; Adam, lr 0.01,
  minibatch 128 drawn by cycling through random permutations of the fitting pairs.
* **Convergence rule** (stored with every fit): every `window` = 50 steps the mean training loss
  of the window is recorded. The fit stops as *converged* when the window loss has not improved
  by more than `tol` = 1e-3 for `patience` = 5 windows, or when it falls below `min_loss` =
  1e-3. Reaching `max_steps` = 4000 first, or a non-finite loss, means *not converged*, and the
  fit is excluded from every procedure and counted in `tables/excluded_fits.csv`. The final R
  is used (not the best window).
* The step budget does not depend on the number of fitting pairs. (A first version counted
  epochs, which gave 256-pair fits only a handful of updates; it is noted in
  [deviations_from_proposal.md](deviations_from_proposal.md).)

Hyperparameters live under `das.fit` in the config; `reproduce.yaml` uses the same defaults
(batch 512 for wide random networks).

## Correctness checks

| Check | Where | Result |
|---|---|---|
| Handcrafted pyvene-tutorial network solves the task | `tests/test_das.py::test_handcrafted_solves_task` | accuracy 1.0 |
| Its known alignment (neurons 0–3 = WX, 4–7 = YZ of layer 0) gives IIA = 1, with R = None and R = I | `test_handcrafted_neuron_alignment_iia_is_one`; also section 1 of `cas reproduce` | IIA = 1.0000 |
| Swapping the two neuron blocks breaks it | `test_handcrafted_wrong_alignment_is_not_perfect` | IIA < 0.9 |
| Interchange algebra (no mask = identity; the right rotated dims come from the right source; R orthogonal) | `test_interchange_algebra` | exact to 1e-5 |
| Tiny random network: DAS IIA near chance | `test_random_network_iia_near_chance` | 0.4 < IIA < 0.6 |
| Counterfactual labels on hand-worked examples | `tests/test_tasks.py` | pass |
| Same predictions as pyvene's `RotatedSpaceIntervention` with our fitted R | `tests/test_pyvene_crosscheck.py` (`-m slow`) | see [running.md](running.md) |

Note: DAS started from a random rotation does *not* reliably rediscover the handcrafted
network's alignment, because that network's logit margins are about 1e-6 (the head computes
||a−b|−|c−d|| − 0.999999(|a−b|+|c−d|)), so gradients are tiny. The check therefore uses the
known alignment, as the tutorial does.
