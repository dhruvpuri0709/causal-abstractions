# Tasks, algorithms and networks

## Inputs

An input is a concatenation of `n_objects` vectors in [-0.5, 0.5]^4, grouped into consecutive
pairs (x0, x1), (x2, x3), .... Each pair is either *equal* (x_{2j+1} is a copy of x_{2j}) or
*different* (two independent uniform draws). An algorithm only needs the equality bits
e_j = [x_{2j} == x_{2j+1}], so `tasks.py` samples bits first and then vectors that realise them.
Bits are sampled so that the task label is balanced, uniformly over bit patterns within each
label (for hierarchical equality this is uniform over the four patterns, as in the pyvene
tutorial; for the distributive law it matches Sutter et al.'s rejection sampler). Every input is
a fresh continuous draw, so inputs never repeat across training, calibration and DAS splits.

## Tasks and algorithms

Every algorithm has two intermediate variables V1, V2 and an output function.

| Task | Input | Label | Algorithm | V1 | V2 | Output |
|---|---|---|---|---|---|---|
| `hierarchical_equality` (Geiger et al. 2024 §4) | 4 objects, 16 dims | (e0 == e1) | `both_equality` | e0 = [w = x] | e1 = [y = z] | V1 == V2 |
| `distributive_law` (Sutter et al. 2025 App. I.3) | 6 objects, 24 dims | (e0 ∧ e1) ∨ (e1 ∧ e2) | `and_or_and` | e0 ∧ e1 | e1 ∧ e2 | V1 ∨ V2 |
| | | | `and_or` | e1 | e0 ∨ e2 | V1 ∧ V2 |

`and_or_and` and `and_or` compute the same function (the distributive law) with different
intermediate variables, which is what makes a clean wrong-algorithm negative possible.
These definitions were checked against `external/non-linear-representation-dilemma/das/Dataset_Generation.py`
(`make_dataset_sample_intervention_AndOrAnd` uses V1 = s0 ∧ s1, V2 = s1 ∧ s2, output V1 ∨ V2;
`..._AndOr` uses V1 = s1, V2 = s0 ∨ s2, output V1 ∧ V2).

**Counterfactual label.** For a base input b and sources s1, s2 with an intervention mask
(m1, m2): V_i' = V_i(s_i) if m_i else V_i(b); label = output(V1', V2'). This is
`Algorithm.counterfactual`; hand-worked examples are in `tests/test_tasks.py`.

## Networks

All networks are ReLU MLPs `input → [width] × 3 → 2` (`models.MLP`). "Layer l" is the post-ReLU
output of hidden layer l, l = 0, 1, 2 (pyvene's `block_output`; the paper's "Layer 1" is our
layer 0). Class 1 means True.

| Kind | How it is built | Truth for hypothesis H |
|---|---|---|
| `random` | PyTorch default init, never trained (Geiger et al. 2024 §4.4) | negative (`negative_random`) |
| `standard` | Adam on the task label only, until held-out accuracy ≥ 0.995 | unknown — descriptive only, never in α or π |
| `iit` toward A | Interchange intervention training (Geiger et al. 2022), identity alignment at layer L* with V1 in neurons [0, k*) and V2 in [k*, 2k*) | positive if H = A; `negative_wrong_algorithm` if H ≠ A |

Random networks are at chance on the task (about 0.5 accuracy); that is expected and recorded in
the network metadata (`task_acc`).

**IIT training** (`train.train_iit`). Each batch of 1024 mixes factual examples and interchange
examples on V1, V2 or both (1/4 each, as described in Sutter et al. App. I.3), with fresh random
inputs. The interchange uses the identity rotation at (L*, k*); the network is trained on
cross-entropy against the algorithm's counterfactual labels (the factual label when nothing is
intervened). Training stops when task accuracy ≥ `target_acc` and neuron-aligned interchange
accuracy ≥ `target_iit_acc` on 4000 fresh examples, or after `max_steps`. Defaults: L* = 1,
k* = width / 4.

**Pass/fail.** After training, task accuracy and the neuron-aligned IIA at (L*, k*) for every
algorithm of the task are measured on 10 000 fresh examples. A network is excluded (and counted
in `tables/excluded_networks.csv`) if: standard and task accuracy < `pass_acc` (0.99); IIT and
task accuracy < `pass_acc` or neuron-aligned IIA for its own algorithm < `pass_iit_acc` (0.98).
Random networks always pass.

## What "truth" means here

T is defined for a *linear (rotation) abstraction at a specific site*: a positive has the
algorithm's variables, by construction, in a known linear subspace (the first 2k* neurons) of a
known layer L*. A wrong-algorithm network has, by construction, the *other* algorithm's variables
at that site, and solves the task. Two caveats:

* A wrong-algorithm network might still implement the hypothesis partially, or at another layer.
  We record, per unit, the agreement between the hypothesised and the trained algorithm's
  counterfactual labels on the test pairs (`label_agreement` in `unit_statistics.csv`); in the
  pilot it was 0.639 on average (range 0.61–0.68) for the distributive law, because the two
  algorithms agree on many interventions.
  At layers other than L* a wrong-algorithm network can be a *true* abstraction of the other
  algorithm. Example: if layer 0 still keeps the three object pairs in separate subspaces, then
  interchanging the subspaces of pairs 0 and 2 together implements And-Or's V2 = e0 ∨ e2 (and
  the subspace of pair 1 implements V1 = e1), whatever the network was IIT-trained to do at L*.
  Max-over-layers procedures can therefore certify the "wrong" algorithm correctly at another
  layer. The analysis reports both scope `all` (counted as a false positive, since truth is
  defined at L*) and scope `site` (procedures restricted to L*), which is the cleaner type I
  error. The selected configuration is recorded (`P1_config`, `P2_config`, `P4_config`).
* For positives, DAS searches with a random-orthogonal starting rotation, so finding the planted
  neuron basis is not guaranteed; low power at some (layer, k) is a property of DAS, not of the
  ground truth.

## Analysis units and nulls

For every network and every algorithm H of its task, and every fitting-set size n_fit, there is
one analysis unit. `Config.units` assigns its truth class:

* evaluation IIT network trained to A, H = A → positive; H ≠ A → `negative_wrong_algorithm`;
* evaluation random network → `negative_random`; evaluation standard network → `unknown_standard`;
* calibration random networks (any H) → random null; calibration IIT networks trained to A with
  H ≠ A → wrong-algorithm null. Calibration networks have their own ids and seeds and are never
  evaluated. For hierarchical equality there is only one algorithm, so there are no
  wrong-algorithm networks or nulls (a clean wrong algorithm for that task is not implemented).
