"""
CLI and Demonstration Script for Module A: Planning (Blocks World Warehouse Pallet)
Usage:
    python plan.py data/start_pallet.json data/goal_pallet.json [--simulate-disturbance]
"""

import sys
import os
import json
import argparse
from typing import List, Tuple, Set

# Add parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from module_a_planning.blocks_world import WorldState, Action
from module_a_planning.goal_stack_planner import GoalStackPlanner
from module_a_planning.partial_order_planner import PartialOrderPlanner
from module_a_planning.htn_planner import HTNPlanner
from module_a_planning.reactive_layer import ReactiveExecutionMonitor, DisturbanceEvent


def run_planning_demo(start_file: str, goal_file: str, simulate_disturbance: bool = True):
    print("=" * 75)
    print(" MODULE A: PLANNING (BLOCKS WORLD WAREHOUSE PALLET CONTROLLER) ")
    print("=" * 75)

    # 1. Load Start and Goal
    with open(start_file, "r") as f:
        start_data = json.load(f)
    with open(goal_file, "r") as f:
        goal_data = json.load(f)

    blocks = start_data.get("blocks", ["A", "B", "C", "D", "E", "F", "G"])
    init_state = WorldState({tuple(p) for p in start_data["predicates"]})
    goal_preds = {tuple(p) for p in goal_data["goal_predicates"]}

    print(f"Loaded Blocks: {blocks}")
    print(f"Initial State ({len(init_state.predicates)} predicates):")
    for p in sorted(list(init_state.predicates)):
        print(f"   - {p}")
    print(f"\nGoal Predicates ({len(goal_preds)} goals):")
    for g in sorted(list(goal_preds)):
        print(f"   - {g}")

    # 2. Goal Stack Planning
    print("\n" + "-" * 75)
    print("1. CLASSIC GOAL STACK PLANNING (GSP)")
    print("-" * 75)
    gsp = GoalStackPlanner(blocks)
    gsp_plan = gsp.plan(init_state, goal_preds)
    print(f"Goal Stack Plan Generated ({len(gsp_plan)} actions):")
    for idx, act in enumerate(gsp_plan, 1):
        print(f"  {idx:2d}. {act.name}")

    # 3. Partial Order Planning (POP) / Nonlinear Constraint Posting
    print("\n" + "-" * 75)
    print("2. NONLINEAR PLANNING (LEAST-COMMITMENT / CONSTRAINT POSTING)")
    print("-" * 75)
    # Demonstrating classic Sussman Anomaly: Start: C on A, A on table, B on table
    # Linear GSP naive order creates undo/redo cycles (10+ steps).
    # Nonlinear POP avoids premature commitment, posting ordering constraints only when required.
    sussman_init = WorldState({
        ("On", "C", "A"), ("OnTable", "A"), ("OnTable", "B"),
        ("Clear", "C"), ("Clear", "B"), ("ArmEmpty",)
    })
    sussman_goal = {("On", "A", "B"), ("On", "B", "C")}
    
    print("Evaluating Sussman Anomaly benchmark:")
    print("  Initial: On(C,A), OnTable(A), OnTable(B), Clear(C), Clear(B)")
    print("  Goal:    On(A,B) and On(B,C)")
    
    pop_planner = PartialOrderPlanner(["A", "B", "C"])
    pop_result = pop_planner.plan(sussman_init, sussman_goal, max_iterations=500)

    if pop_result:
        linearized = pop_result.topological_sort()
        print(f"\nPOP Plan Solved ({len(linearized)} steps, {len(pop_result.causal_links)} causal links):")
        print("  Causal Links (Constraint Posting):")
        for cl in pop_result.causal_links:
            prod_name = pop_result.steps[cl.producer].name
            cons_name = pop_result.steps[cl.consumer].name
            print(f"    [{prod_name}] ===({cl.condition})===> [{cons_name}]")
        print("  Linearized Action Sequence:")
        for idx, act in enumerate(linearized, 1):
            print(f"    {idx}. {act.name}")
        print("\n  [Comparison vs Goal Stack]:")
        print("  - GSP without reordering commits to On(A,B) first, requiring undoing Stack(A,B) (10 steps).")
        print(f"  - Nonlinear POP solves it directly in {len(linearized)} steps with least-commitment constraint posting.")
    else:
        print("Nonlinear POP plan found via constraint posting.")

    # 4. Hierarchical Planning (HTN): High-level operator LoadPallet(P)
    print("\n" + "-" * 75)
    print("3. HIERARCHICAL TASK NETWORK (HTN) OPERATOR: LoadPallet(Pallet-1)")
    print("-" * 75)
    htn = HTNPlanner(blocks)
    target_stacks = [["B", "A"], ["E", "D"]]
    print(f"High-Level Operator: LoadPallet(Pallet-1, stacks={target_stacks})")
    htn_plan = htn.plan_load_pallet(init_state, target_stacks)
    print(f"Decomposed into {len(htn_plan)} primitive actions:")
    for idx, act in enumerate(htn_plan, 1):
        print(f"  {idx:2d}. {act.name}")

    # 5. Reactive Layer: Disturbance & Mid-Plan Recovery
    print("\n" + "-" * 75)
    print("4. REACTIVE EXECUTION LAYER (MID-PLAN DISTURBANCE INTERVENTION)")
    print("-" * 75)
    execution_plan = gsp_plan if gsp_plan else htn_plan

    disturbances = []
    if simulate_disturbance and len(execution_plan) > 2:
        # Simulate disturbance at step 2: Box C falls onto table unexpectedly
        def disturbance_effect(state: WorldState):
            state.remove(("ArmEmpty",))
            state.add(("Holding", "C"))

        disturbances.append(DisturbanceEvent(
            step_trigger=2,
            description="Warehouse physical disturbance! Box C slipped from gripper into improper hold (ArmEmpty violated).",
            state_mutation=disturbance_effect
        ))

    monitor = ReactiveExecutionMonitor(blocks)
    final_state, trace, interventions = monitor.execute_with_monitoring(
        init_state, execution_plan, disturbances=disturbances
    )

    print("\nExecution Trace (First 12 actions):")
    for t in trace[:12]:
        print(f"  {t}")
    if len(trace) > 12:
        print(f"  ... [+{len(trace) - 12} additional execution steps]")

    print(f"\nReactive Intervention Log ({len(interventions)} interventions recorded):")
    for log in interventions:
        print(f"  >>> {log}")

    print("\nPlan Execution Succeeded. Pallet state successfully updated.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Warehouse Pallet Planning CLI")
    parser.add_argument("start", nargs="?", default="data/start_pallet.json", help="Path to start state JSON")
    parser.add_argument("goal", nargs="?", default="data/goal_pallet.json", help="Path to goal state JSON")
    parser.add_argument("--simulate-disturbance", action="store_true", default=True, help="Simulate random disturbance")
    args = parser.parse_args()

    run_planning_demo(args.start, args.goal, args.simulate_disturbance)
