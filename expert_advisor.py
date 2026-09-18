"""
Module E: Maintenance Advisor CLI.
Forward-Chaining Rule-Based Expert System with Full Explanation Trace.
Usage:
    python expert_advisor.py --symptoms wheel_rpm_high,ground_speed_zero
    python expert_advisor.py --symptoms motor_current_high,motor_temp_high
    python expert_advisor.py --symptoms lidar_distance_near,ultrasonic_distance_far
    python expert_advisor.py --add-rule
"""

import os
import sys
import argparse
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from module_e_expert_system.engine import MaintenanceExpertEngine
from module_e_expert_system.knowledge_acquisition import interactive_rule_acquisition, add_diagnostic_rule


def run_expert_advisor(symptoms: List[str], rules_path: str = "module_e_expert_system/rules.json"):
    print("=" * 80)
    print(" MODULE E: WAREHOUSE ROBOT MAINTENANCE ADVISOR (EXPERT SYSTEM) ")
    print("=" * 80)

    engine = MaintenanceExpertEngine(rules_path)
    print(f"Loaded Knowledge Base: {len(engine.rules)} diagnostic rules from '{rules_path}'.")

    cleaned_symptoms = [s.strip().lower() for s in symptoms if s.strip()]
    if not cleaned_symptoms:
        # Default scenario if none specified
        cleaned_symptoms = ["wheel_rpm_high", "ground_speed_zero", "accelerometer_tilt_normal"]
        print(f"No symptoms passed. Running default warehouse test scenario: {cleaned_symptoms}")
    else:
        print(f"Observed Input Symptoms ({len(cleaned_symptoms)}):")
        for s in cleaned_symptoms:
            print(f"  * {s}")

    for s in cleaned_symptoms:
        engine.assert_symptom(s)

    # Run forward chaining inference
    engine.infer()

    print("\n" + "-" * 80)
    print("1. INFERENCE EXECUTION & RULE FIRING SEQUENCE")
    print("-" * 80)
    if not engine.fired_rule_history:
        print("  No diagnostic rules triggered. Robot status is within nominal parameters.")
    else:
        for rec in engine.fired_rule_history:
            print(f"  Step {rec.step:2d} | [{rec.rule.id}] {rec.rule.name}")
            print(f"          Premises Matched : {rec.matched_facts}")
            print(f"          Asserted Facts   : {rec.asserted_facts}")
            print(f"          Rule Explanation : {rec.rule.explanation}")
            print()

    print("-" * 80)
    print("2. FINAL DIAGNOSTIC REPORT & DERIVED CONCLUSIONS")
    print("-" * 80)
    diagnoses = engine.get_diagnoses()
    if not diagnoses:
        print("  [STATUS: NORMAL] No equipment defects or safety warnings inferred.")
    else:
        for idx, d in enumerate(diagnoses, 1):
            print(f"  {idx}. DIAGNOSIS / ACTION: {d.upper()}")

    print("\n" + "-" * 80)
    print("3. EXPLANATION FACILITY: 'WHY' DEDUCTION TRACE")
    print("-" * 80)
    for d in diagnoses:
        print(f"\n--- Explanation for '{d}': ---")
        why_lines = engine.explain_why(d)
        for line in why_lines:
            print(line)

    print("\n" + "=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Warehouse Robot Maintenance Advisor Expert System CLI")
    parser.add_argument("--symptoms", type=str, default="", help="Comma-separated list of symptom flags")
    parser.add_argument("--rules", type=str, default="module_e_expert_system/rules.json", help="Path to rules JSON")
    parser.add_argument("--add-rule", action="store_true", help="Launch interactive knowledge acquisition form")
    args = parser.parse_args()

    if args.add_rule:
        interactive_rule_acquisition(args.rules)
    else:
        symptom_list = [s.strip() for s in args.symptoms.split(",") if s.strip()]
        run_expert_advisor(symptom_list, args.rules)
