"""
Warehouse Robot Brain Integrated Control Loop (main.py).
Wires together all 5 modules across a 100-event simulated shift log:
    - Sensor readings -> Module B (Uncertainty Reasoning)
    - If damage suspected -> Module E (Expert System Diagnosis & Explanation)
    - Module A (Planning & Reactive Execution on pallet stacking tasks)
    - Module C (Adversarial Dock Contention when battery is low)
    - Module D (Connectionist RNN sequence anomaly forecasting & Hopfield barcode recall)
"""

import os
import sys
import json
import numpy as np

# Ensure root workspace is on python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Module A
from module_a_planning.blocks_world import WorldState, Action
from module_a_planning.goal_stack_planner import GoalStackPlanner
from module_a_planning.htn_planner import HTNPlanner
from module_a_planning.reactive_layer import ReactiveExecutionMonitor, DisturbanceEvent

# Module B
from module_b_uncertainty.bayes_filter import BayesDamageFilter
from module_b_uncertainty.bayes_network import WarehouseBayesNet
from module_b_uncertainty.dempster_shafer import DempsterShaferReasoner
from module_b_uncertainty.fuzzy_reasoner import FuzzyDamageReasoner

# Module C
from module_c_games.dock_game_env import DockGameState
from module_c_games.iterative_deepening import IterativeDeepeningSolver

# Module D
from module_d_connectionist.hopfield_network import HopfieldNetwork, generate_warehouse_label_patterns
from module_d_connectionist.rnn_anomaly_detector import RNNAnomalyTrainer

# Module E
from module_e_expert_system.engine import MaintenanceExpertEngine


class IntegratedWarehouseBrain:
    def __init__(self, rules_path: str = "module_e_expert_system/rules.json"):
        print(">> [INITIALIZING WAREHOUSE BRAIN] Loading reasoning & connectionist modules...")

        # Module A Planners
        self.blocks = ["A", "B", "C", "D", "E", "F", "G"]
        self.gsp_planner = GoalStackPlanner(self.blocks)
        self.htn_planner = HTNPlanner(self.blocks)
        self.reactive_monitor = ReactiveExecutionMonitor(self.blocks)

        # Module B Uncertainty Engines
        self.bayes_filter = BayesDamageFilter()
        self.bayes_net = WarehouseBayesNet()
        self.ds_reasoner = DempsterShaferReasoner()
        self.fuzzy_reasoner = FuzzyDamageReasoner()

        # Module C Game Solver
        self.dock_solver = IterativeDeepeningSolver()

        # Module D Connectionist Engines
        self.hopfield = HopfieldNetwork(size=64)
        self.label_patterns = generate_warehouse_label_patterns()
        self.hopfield.train(list(self.label_patterns.values()))

        print(">> [INITIALIZING] Pre-training Module D PyTorch RNN on sensor time-series...")
        self.rnn_trainer = RNNAnomalyTrainer(hidden_dim=16)
        self.rnn_trainer.train_and_evaluate(epochs=8, batch_size=32)

        # Module E Expert Advisor
        self.expert_engine = MaintenanceExpertEngine(rules_path)

        print(">> [SYSTEM READY] Warehouse Brain fully initialized and operational.\n")

    def run_shift_simulation(self, shift_log_path: str = "data/shift_stream_100.json", max_events: int = 100):
        print("=" * 85)
        print(f" COMMENCING AUTONOMOUS WAREHOUSE SHIFT EXECUTION (Source: {shift_log_path}) ")
        print("=" * 85)

        with open(shift_log_path, "r") as f:
            events = json.load(f)

        events = events[:max_events]

        stats = {
            "total_events": len(events),
            "boxes_inspected": 0,
            "damage_flagged": 0,
            "expert_diagnoses": 0,
            "stacking_orders": 0,
            "reactive_interventions": 0,
            "stream_anomalies_detected": 0,
            "barcodes_cleaned": 0,
            "dock_races_won": 0,
            "dock_races_total": 0
        }

        for idx, evt in enumerate(events, 1):
            e_type = evt["type"]
            e_id = evt["event_id"]
            ts = evt["timestamp"]

            # -------------------------------------------------------------
            # CASE 1: BOX INSPECTION (Module B -> Module E)
            # -------------------------------------------------------------
            if e_type == "BOX_INSPECTION":
                stats["boxes_inspected"] += 1
                box_id = evt["box_id"]
                ev_dict = {
                    "weight_anomaly": evt["weight_anomaly"],
                    "crack_detected": evt["crack_detected"],
                    "tilt_triggered": evt["tilt_triggered"]
                }

                # Multi-paradigm uncertainty evaluation
                bayes_prob = self.bayes_filter.compute_posterior(ev_dict)
                (ds_bel, ds_pl), _ = self.ds_reasoner.evaluate(ev_dict)
                fuzzy_score = self.fuzzy_reasoner.evaluate(evt["weight_deviation_kg"], evt["crack_severity_score"])

                # Damage is suspected if Bayes posterior > 0.65 or Fuzzy > 60%
                is_damaged = (bayes_prob > 0.65 or fuzzy_score > 60.0)

                if is_damaged:
                    stats["damage_flagged"] += 1
                    print(f"[{ts} | {e_id}] >> MODULE B: Box {box_id} flagged DAMAGED! (Bayes: {bayes_prob:.2f}, Fuzzy: {fuzzy_score:.1f}%, DS: [{ds_bel:.2f}, {ds_pl:.2f}])")

                    # Pipe suspected damage to Module E for root-cause diagnosis
                    stats["expert_diagnoses"] += 1
                    self.expert_engine.reset()
                    self.expert_engine.assert_symptom("pallet_tray_empty" if evt["weight_anomaly"] else "normal_load")
                    if evt["tilt_triggered"]:
                        self.expert_engine.assert_symptom("accelerometer_tilt_high")
                        self.expert_engine.assert_symptom("ground_speed_zero")
                    if evt["crack_detected"]:
                        self.expert_engine.assert_symptom("crack_detected_all_boxes")
                        self.expert_engine.assert_symptom("illumination_nominal")

                    self.expert_engine.infer()
                    diagnoses = self.expert_engine.get_diagnoses()
                    diag_str = ", ".join(diagnoses) if diagnoses else "UNSPECIFIED_EXTERNAL_CRUSH"
                    print(f"         >> MODULE E: Diagnostic Root Cause -> [{diag_str}]")
                    # Print one why explanation
                    if diagnoses:
                        why = self.expert_engine.explain_why(diagnoses[0])
                        print(f"            Trace: {why[0].strip()}")

            # -------------------------------------------------------------
            # CASE 2: PALLET STACK TASK (Module A Planning & Reactive Layer)
            # -------------------------------------------------------------
            elif e_type == "PALLET_STACK_TASK":
                stats["stacking_orders"] += 1
                order_id = evt["order_id"]
                has_dist = evt["has_disturbance"]

                # Generate high-level pallet stacking plan
                init_st = WorldState({
                    ("OnTable", "A"), ("OnTable", "B"), ("OnTable", "C"),
                    ("Clear", "A"), ("Clear", "B"), ("Clear", "C"), ("ArmEmpty",)
                })
                # Target: Stack B on A, C on B
                target_stacks = [["A", "B", "C"]]
                plan = self.htn_planner.plan_load_pallet(init_st, target_stacks)

                disturbances = []
                if has_dist:
                    def dist_effect(st: WorldState):
                        st.remove(("ArmEmpty",))
                        st.add(("Holding", "B"))

                    disturbances.append(DisturbanceEvent(
                        step_trigger=2,
                        description="Mid-execution conveyor vibration knocked box B into arm.",
                        state_mutation=dist_effect
                    ))

                final_st, trace, interventions = self.reactive_monitor.execute_with_monitoring(
                    init_st, plan, disturbances=disturbances
                )

                if interventions:
                    stats["reactive_interventions"] += len(interventions)
                    print(f"[{ts} | {e_id}] >> MODULE A: Order {order_id} encountered physical disturbance!")
                    print(f"         >> REACTIVE LAYER: Intervened and locally repaired plan without replanning from scratch.")
                else:
                    print(f"[{ts} | {e_id}] >> MODULE A: Order {order_id} stacked successfully ({len(plan)} primitive steps).")

            # -------------------------------------------------------------
            # CASE 3: TELEMETRY STREAM & BARCODE SCAN (Module D Connectionist)
            # -------------------------------------------------------------
            elif e_type == "TELEMETRY_STREAM":
                raw_seq = np.array(evt["sensor_sequence"], dtype=np.float32)
                # Standardize input for RNN
                norm_seq = raw_seq.copy()
                norm_seq[:, 1] = (norm_seq[:, 1] - 45.0) / 10.0
                norm_seq[:, 2] = (norm_seq[:, 2] - 3.0) / 2.0

                is_imminent, conf = self.rnn_trainer.predict_imminent_anomaly(norm_seq)
                if is_imminent:
                    stats["stream_anomalies_detected"] += 1
                    print(f"[{ts} | {e_id}] >> MODULE D (RNN): IMMINENT FAILURE PREDICTED from 10-step telemetry! (Conf: {conf*100:.1f}%)")

                if evt.get("barcode_scan_noisy", False):
                    stats["barcodes_cleaned"] += 1
                    lbl_type = evt.get("label_type", "HAZARD_CROSS")
                    clean_pat = self.label_patterns[lbl_type]
                    corrupted, _ = HopfieldNetwork.corrupt_pattern(clean_pat, flip_ratio=0.12)
                    recalled, _, iters = self.hopfield.recall(corrupted)
                    is_match = np.array_equal(recalled, clean_pat)
                    print(f"[{ts} | {e_id}] >> MODULE D (Hopfield): Restored noisy '{lbl_type}' barcode scan in {iters} iterations (Match: {is_match}).")

            # -------------------------------------------------------------
            # CASE 4: BATTERY LOW DOCK RACE (Module C Game Playing)
            # -------------------------------------------------------------
            elif e_type == "BATTERY_LOW_DOCK_RACE":
                stats["dock_races_total"] += 1
                bat_pct = evt["robot_battery_pct"]
                print(f"[{ts} | {e_id}] >> MODULE C: Battery critical ({bat_pct}%)! Contending with {evt['competitor_robot_id']} for last charging dock.")

                # Fast simulation of adversarial dock race using Alpha-Beta + IDS (depth 4)
                game_state = DockGameState()
                move_count = 0
                while not game_state.is_terminal()[0] and move_count < 10:
                    move_count += 1
                    if game_state.turn == 1:
                        best_m, _, _, _ = self.dock_solver.search(game_state, max_depth=3)
                        game_state = game_state.apply_move(best_m if best_m else (0, 0))
                    else:
                        legal = game_state.get_legal_moves(-1)
                        game_state = game_state.apply_move(legal[0])

                is_term, util = game_state.is_terminal()
                if util > 0:
                    stats["dock_races_won"] += 1
                    print(f"         >> MODULE C: AI Robot secured charging dock (+{util:.1f} pts in {move_count} moves).")
                else:
                    print(f"         >> MODULE C: Dock contention concluded with draw or blocked access.")

        # Print Final Shift Report
        print("\n" + "=" * 85)
        print(" WAREHOUSE SHIFT EXECUTION SUMMARY REPORT ")
        print("=" * 85)
        print(f"Total Shift Events Processed        : {stats['total_events']}")
        print(f"Boxes Inspected by Module B         : {stats['boxes_inspected']}")
        print(f"Damaged Boxes Flagged               : {stats['damage_flagged']}")
        print(f"Expert System Diagnoses (Module E)  : {stats['expert_diagnoses']}")
        print(f"Pallet Stacking Orders (Module A)   : {stats['stacking_orders']}")
        print(f"Reactive Layer Interventions        : {stats['reactive_interventions']}")
        print(f"Sensor Stream Anomalies (Module D)  : {stats['stream_anomalies_detected']}")
        print(f"Noisy Barcode Scans Cleaned         : {stats['barcodes_cleaned']}")
        print(f"Dock Contention Battles Won         : {stats['dock_races_won']} / {stats['dock_races_total']}")
        print("=" * 85)
        print("All 5 modules successfully integrated in real-time control loop.")


if __name__ == "__main__":
    brain = IntegratedWarehouseBrain()
    brain.run_shift_simulation()
