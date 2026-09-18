# Uncertainty-Aware Warehouse Robot Brain (`warehouse-brain`)

An integrated multi-paradigm autonomous robot brain for e-commerce warehouses. It combines classical symbolic planning, multi-framework uncertainty reasoning, adversarial search, connectionist neural memory/anomaly forecasting, and a forward-chaining diagnostic expert system.

---

## Quickstart & Environment Setup

```bash
# Clone or navigate to the repository
cd warehouse-brain

# Install dependencies (Python 3.10+)
pip install numpy matplotlib torch scikit-fuzzy pgmpy scipy scikit-learn pytest
```

---

## How to Run Each Module

| Module | CLI Command | Description |
|---|---|---|
| **Module A** (Planning) | `python plan.py data/start_pallet.json data/goal_pallet.json` | Runs Goal Stack Planning, POP constraint posting, HTN decomposition, and reactive disturbance recovery. |
| **Module B** (Uncertainty) | `python uncertainty_reasoning.py` | Runs the nonmonotonic vs monotonic demo and outputs the 10-reading comparison table across all 6 paradigms. |
| **Module C** (Game Playing) | `python dock_game.py --depth 5 --mode ai_vs_random`<br>`python dock_game.py --chart` | Interactive 5x5 dock contention game, with `--chart` generating the search node exploration benchmark. |
| **Module D** (Connectionist) | `python anomaly_recall.py` | Runs from-scratch Hopfield 8x8 label recall, trains the PyTorch LSTM anomaly forecaster, and displays comparison. |
| **Module E** (Expert System)| `python expert_advisor.py --symptoms wheel_rpm_high,ground_speed_zero`<br>`python expert_advisor.py --add-rule` | Diagnoses hardware faults with a full recursive "WHY" deduction trace; `--add-rule` opens dynamic rule authoring. |
| **Integration** (Full Loop) | `python main.py` | Runs the integrated shift controller processing 100 simulated warehouse events across all 5 modules. |
| **Automated Tests** | `pytest test_warehouse_brain.py -v` | Executes the comprehensive test suite validating all 5 modules and integration. |

---

## Module Demonstrations (1 Paragraph Per Module)

### Module A — Planning: Stack the Pallet (Blocks World)
Module A demonstrates deliberative and reactive action synthesis over a Blocks World warehouse pallet (boxes A–G). It generates end-to-end stacking schedules using classical Goal Stack Planning and shows how Partial Order Planning (POP) uses least-commitment constraint posting to avoid premature ordering decisions on the Sussman Anomaly, producing an optimal 6-step plan without redundant undo-cycles. In addition, Hierarchical Task Network (HTN) decomposition expands high-level operators (`LoadPallet`) into executable primitive pick-and-place actions. Crucially, when an unexpected physical disturbance occurs mid-execution (e.g., an arm collision or dropped box), a dedicated reactive monitoring layer synthesizes local precondition repairs rather than discarding the schedule and replanning from scratch.

### Module B — Reasoning Under Uncertainty: "Is This Box Damaged?"
Module B demonstrates how differing epistemic paradigms handle noisy, conflicting warehouse sensory data (weight scales, optical crack detectors, and tilt switches). It contrasts classical monotonic logic—where proved theorems can never be invalidated—with Nonmonotonic Default Logic, where asserting a new sensor anomaly dynamically retracts the default assumption of box integrity. It benchmarks Bayesian updating against MYCIN Certainty Factors, exact Bayesian Network inference via variable elimination in `pgmpy`, Dempster-Shafer belief-plausibility intervals, and Mamdani fuzzy centroid defuzzification across 10 synthetic test cases. The module illustrates key divergence phenomena, such as how Bayesian Networks use causal structure (`SensorNoise` $\rightarrow$ `WeightReading`) to "explain away" false alarms, whereas naive Bayes and rigid rule systems cannot.

### Module C — Game Playing: Dock Contention
Module C demonstrates adversarial decision-making under resource scarcity as two warehouse robots compete for the last available charging dock on an obstacle-strewn 5x5 grid with a branching factor $\ge 3$. It implements plain MiniMax alongside Alpha-Beta pruning, experimentally proving that pruning eliminates over 80% of visited states at search depths 5–6 while preserving identical minimax values. To enable real-time anytime responsiveness, it adds Iterative Deepening Search (IDS) coupled with a domain move-ordering heuristic that evaluates dock-proximate moves first, maximizing alpha-beta cutoffs. The accompanying Matplotlib chart (`dock_game_nodes_explored.png`) empirically charts the logarithmic divergence in nodes explored between MiniMax, Alpha-Beta, and ordered IDS.

### Module D — Connectionist Models: Anomaly Recall
Module D explores distributed neural representations for associative pattern restoration and sequential anomaly forecasting. It implements a discrete Hopfield Network from scratch using NumPy and Hebbian outer-product learning, storing 64-bit ($8 \times 8$) bipolar box barcodes and proving associative recall convergence under 10–15% random bit-flip sensor corruption. In parallel, a PyTorch LSTM recurrent neural network is trained on multivariate time-series (vibration, motor temperature, current, and speed drift) to forecast imminent mechanical failures over a 10-step horizon with $>95\%$ test accuracy. An embedded comparative analysis articulates the trade-offs between Hopfield energy-attractor memory, temporal recurrent sequence models, and symbolic rule systems.

### Module E — Expert System: The Maintenance Advisor
Module E demonstrates symbolic domain knowledge representation and automated fault diagnosis through a forward-chaining expert system engine operating on 18 industrial maintenance rules. When presented with anomalous hardware symptoms (e.g., elevated wheel RPM and zero ground speed), the engine resolves conflicts via condition specificity and recursively builds a complete explanatory proof tree ("WHY" trace) detailing every fired rule, satisfied antecedent, and asserted conclusion. Furthermore, it incorporates a dynamic knowledge acquisition interface that validates and appends new technician-authored diagnostic rules to `rules.json` without modifying code or restarting the controller.

---

## Integration Summary (`main.py`)
The top-level controller executes an event-driven control loop over a 100-event simulated shift log. Physical telemetry routes through Module B for uncertainty evaluation; confirmed package anomalies trigger Module E for root-cause diagnosis; pallet movements are dispatched to Module A with active disturbance monitoring; low battery thresholds trigger Module C to contend for dock access; and incoming high-frequency streams are continuously monitored by Module D's RNN alongside Hopfield barcode cleanup.
