"""
Automated PyTest Suite for Uncertainty-Aware Warehouse Robot Brain.
Validates all 5 modules and the integrated control loop.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Module A
from module_a_planning.blocks_world import WorldState, Action, get_all_actions
from module_a_planning.goal_stack_planner import GoalStackPlanner
from module_a_planning.partial_order_planner import PartialOrderPlanner
from module_a_planning.htn_planner import HTNPlanner
from module_a_planning.reactive_layer import ReactiveExecutionMonitor, DisturbanceEvent

# Module B
from module_b_uncertainty.nonmonotonic import MonotonicKB, NonmonotonicDefaultKB
from module_b_uncertainty.bayes_filter import BayesDamageFilter
from module_b_uncertainty.certainty_factors import CertaintyFactorReasoner
from module_b_uncertainty.bayes_network import WarehouseBayesNet
from module_b_uncertainty.dempster_shafer import DempsterShaferReasoner
from module_b_uncertainty.fuzzy_reasoner import FuzzyDamageReasoner

# Module C
from module_c_games.dock_game_env import DockGameState
from module_c_games.minimax import MiniMaxSolver
from module_c_games.alpha_beta import AlphaBetaSolver
from module_c_games.iterative_deepening import IterativeDeepeningSolver

# Module D
from module_d_connectionist.hopfield_network import HopfieldNetwork, generate_warehouse_label_patterns
from module_d_connectionist.rnn_anomaly_detector import RNNAnomalyTrainer

# Module E
from module_e_expert_system.engine import MaintenanceExpertEngine
from module_e_expert_system.knowledge_acquisition import add_diagnostic_rule

# Integration
from main import IntegratedWarehouseBrain


# =====================================================================
# MODULE A TESTS
# =====================================================================
def test_module_a_goal_stack_planner():
    blocks = ["A", "B", "C"]
    init_state = WorldState({
        ("OnTable", "A"), ("OnTable", "B"), ("OnTable", "C"),
        ("Clear", "A"), ("Clear", "B"), ("Clear", "C"), ("ArmEmpty",)
    })
    goal = {("On", "A", "B"), ("On", "B", "C")}
    planner = GoalStackPlanner(blocks)
    plan = planner.plan(init_state, goal)
    assert len(plan) > 0

    curr = init_state
    for act in plan:
        assert act.is_applicable(curr)
        curr = act.apply(curr)
    assert curr.satisfies(goal)


def test_module_a_partial_order_planner():
    sussman_init = WorldState({
        ("On", "C", "A"), ("OnTable", "A"), ("OnTable", "B"),
        ("Clear", "C"), ("Clear", "B"), ("ArmEmpty",)
    })
    sussman_goal = {("On", "A", "B"), ("On", "B", "C")}
    pop = PartialOrderPlanner(["A", "B", "C"])
    plan = pop.plan(sussman_init, sussman_goal)
    assert plan is not None
    linear = plan.topological_sort()
    assert len(linear) == 6
    assert len(plan.causal_links) > 0


def test_module_a_htn_and_reactive():
    blocks = ["A", "B", "C"]
    htn = HTNPlanner(blocks)
    init_state = WorldState({
        ("OnTable", "A"), ("OnTable", "B"), ("OnTable", "C"),
        ("Clear", "A"), ("Clear", "B"), ("Clear", "C"), ("ArmEmpty",)
    })
    plan = htn.plan_load_pallet(init_state, [["A", "B", "C"]])
    assert len(plan) > 0

    def dist_fn(st):
        st.remove(("ArmEmpty",))
        st.add(("Holding", "B"))

    dist = DisturbanceEvent(step_trigger=2, description="Test Disturbance", state_mutation=dist_fn)
    monitor = ReactiveExecutionMonitor(blocks)
    final_st, trace, interventions = monitor.execute_with_monitoring(init_state, plan, [dist])
    assert len(interventions) == 1
    assert final_st.satisfies({("On", "B", "A"), ("On", "C", "B")})


# =====================================================================
# MODULE B TESTS
# =====================================================================
def test_module_b_nonmonotonic():
    mono = MonotonicKB()
    mono.tell("Box(B1)")
    assert mono.query_undamaged("B1") is True
    mono.tell("TiltSensorTriggered(B1)")
    assert mono.query_undamaged("B1") is True

    default_kb = NonmonotonicDefaultKB()
    default_kb.tell("Box(B1)")
    r1 = default_kb.query("B1")
    assert r1["status"] == "UNDAMAGED"
    assert r1["retraction_occurred"] is False

    default_kb.tell("TiltSensorTriggered(B1)")
    r2 = default_kb.query("B1")
    assert r2["status"] == "DAMAGED"
    assert r2["retraction_occurred"] is True


def test_module_b_bayes_and_cf():
    bayes = BayesDamageFilter()
    all_clear = {"weight_anomaly": False, "crack_detected": False, "tilt_triggered": False}
    all_positive = {"weight_anomaly": True, "crack_detected": True, "tilt_triggered": True}
    
    p_clear = bayes.compute_posterior(all_clear)
    p_pos = bayes.compute_posterior(all_positive)
    assert p_clear < 0.05
    assert p_pos > 0.95

    cf = CertaintyFactorReasoner()
    cf_clear = cf.combine_evidence(all_clear)
    cf_pos = cf.combine_evidence(all_positive)
    assert cf_clear < -0.8
    assert cf_pos > 0.9


def test_module_b_bayes_net_and_ds_and_fuzzy():
    bn = WarehouseBayesNet()
    p_dmg_low_noise = bn.query_damaged_posterior({'CrackVisible': 'No', 'Tilt': 'No', 'WeightReading': 'Normal', 'SensorNoise': 'Low'})
    assert p_dmg_low_noise < 0.05

    ds = DempsterShaferReasoner()
    (bel, pl), _ = ds.evaluate({"weight_anomaly": True, "crack_detected": True, "tilt_triggered": True})
    assert bel > 0.9
    assert pl >= bel

    fuzzy = FuzzyDamageReasoner()
    score_clean = fuzzy.evaluate(0.2, 0.1)
    score_damaged = fuzzy.evaluate(8.5, 9.2)
    assert score_clean < 25.0
    assert score_damaged > 75.0


# =====================================================================
# MODULE C TESTS
# =====================================================================
def test_module_c_minimax_vs_alphabeta():
    state = DockGameState()
    depth = 4

    mm = MiniMaxSolver()
    mm_move, mm_val = mm.get_best_move(state, depth)

    ab = AlphaBetaSolver()
    ab_move, ab_val = ab.get_best_move(state, depth)

    assert abs(mm_val - ab_val) < 1e-4
    assert ab.nodes_explored < mm.nodes_explored
    assert ab.pruning_cutoffs > 0


def test_module_c_iterative_deepening():
    state = DockGameState()
    ids = IterativeDeepeningSolver()
    move, val, nodes, elapsed = ids.search(state, max_depth=3)
    assert move is not None
    assert nodes > 0
    assert len(ids.depth_profile) == 3


# =====================================================================
# MODULE D TESTS
# =====================================================================
def test_module_d_hopfield_recall():
    patterns = generate_warehouse_label_patterns()
    hopfield = HopfieldNetwork(size=64)
    hopfield.train(list(patterns.values()))

    target = patterns["HAZARD_CROSS"]
    corrupted, flips = HopfieldNetwork.corrupt_pattern(target, flip_ratio=0.12)
    recalled, energy_hist, iters = hopfield.recall(corrupted)

    assert iters <= 5
    assert np.array_equal(recalled, target)
    assert energy_hist[-1] <= energy_hist[0]


def test_module_d_rnn_classifier():
    trainer = RNNAnomalyTrainer(hidden_dim=16)
    metrics = trainer.train_and_evaluate(epochs=10, batch_size=32)
    assert metrics["test_accuracy"] >= 0.90

    # Test distinct clean normal vs deteriorating sequence
    norm_seq = np.zeros((10, 4), dtype=np.float32)
    norm_seq[:, 0] = np.random.normal(0.2, 0.02, 10)  # low vibration
    norm_seq[:, 1] = 0.0  # normalized nominal temp
    norm_seq[:, 2] = 0.0  # normalized nominal current
    norm_seq[:, 3] = 0.0  # zero drift

    is_anom, prob = trainer.predict_imminent_anomaly(norm_seq)
    assert prob < 0.5
    assert is_anom is False

    anom_seq = norm_seq.copy()
    anom_seq[5:, 0] += np.linspace(0.8, 1.5, 5).astype(np.float32)
    anom_seq[5:, 1] += np.linspace(1.5, 2.5, 5).astype(np.float32)
    anom_seq[5:, 2] += np.linspace(1.2, 2.0, 5).astype(np.float32)
    is_anom_escalated, prob_escalated = trainer.predict_imminent_anomaly(anom_seq)
    assert prob_escalated > 0.5
    assert is_anom_escalated is True


# =====================================================================
# MODULE E TESTS
# =====================================================================
def test_module_e_expert_system_and_explanation():
    engine = MaintenanceExpertEngine("module_e_expert_system/rules.json")
    engine.assert_symptom("wheel_rpm_high")
    engine.assert_symptom("ground_speed_zero")
    engine.assert_symptom("accelerometer_tilt_normal")

    engine.infer()
    diagnoses = engine.get_diagnoses()
    assert "wheel_slipping" in diagnoses
    assert "traction_loss_hazard" in diagnoses
    assert "inspect_floor_surface" in diagnoses

    why_trace = engine.explain_why("inspect_floor_surface")
    assert len(why_trace) > 1
    assert any("RULE_02" in line for line in why_trace)


def test_module_e_knowledge_acquisition(tmp_path):
    temp_rules_file = str(tmp_path / "temp_rules.json")
    with open(temp_rules_file, "w") as f:
        f.write("[]")

    added = add_diagnostic_rule(
        rules_path=temp_rules_file,
        name="Test Sensor Fault",
        if_all=["voltage_spike", "fuse_blown"],
        then_assert=["replace_fuse"],
        explanation="Overvoltage blew primary protection fuse.",
        rule_id="RULE_TEST_99"
    )
    assert added["id"] == "RULE_TEST_99"

    eng = MaintenanceExpertEngine(temp_rules_file)
    assert len(eng.rules) == 1
    eng.assert_symptom("voltage_spike")
    eng.assert_symptom("fuse_blown")
    eng.infer()
    assert "replace_fuse" in eng.get_diagnoses()


# =====================================================================
# INTEGRATION CONTROL LOOP TEST
# =====================================================================
def test_integration_control_loop():
    brain = IntegratedWarehouseBrain()
    brain.run_shift_simulation(shift_log_path="data/shift_stream_100.json", max_events=10)
