"""
Module E: Knowledge Acquisition Facility.
Enables warehouse technicians to add new diagnostic rules dynamically
without editing system source code. Validates syntax and appends to rules.json.
"""

import json
import os
from typing import List, Dict, Any, Optional


def add_diagnostic_rule(
    rules_path: str,
    name: str,
    if_all: List[str],
    then_assert: List[str],
    explanation: str,
    rule_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Validates and appends a new rule to the specified JSON knowledge base.
    """
    if not os.path.exists(rules_path):
        raise FileNotFoundError(f"Rules file not found at {rules_path}")

    with open(rules_path, "r") as f:
        rules = json.load(f)

    # Generate next rule ID if not provided
    if not rule_id:
        existing_ids = [r["id"] for r in rules if "id" in r]
        next_num = len(existing_ids) + 1
        rule_id = f"RULE_{next_num:02d}"

    # Validation
    if not if_all:
        raise ValueError("Rule must have at least one antecedent condition in 'if_all'")
    if not then_assert:
        raise ValueError("Rule must have at least one consequent assertion in 'then_assert'")
    if not explanation.strip():
        raise ValueError("Rule must include an explanation of physical causation")

    new_rule_entry = {
        "id": rule_id,
        "name": name.strip(),
        "if_all": [c.strip().lower() for c in if_all],
        "then_assert": [a.strip().lower() for a in then_assert],
        "explanation": explanation.strip()
    }

    rules.append(new_rule_entry)

    with open(rules_path, "w") as f:
        json.dump(rules, f, indent=2)

    return new_rule_entry


def interactive_rule_acquisition(rules_path: str):
    """Interactive command-line prompt for technician rule entry."""
    print("=" * 60)
    print(" TECHNICIAN KNOWLEDGE ACQUISITION INTERFACE ")
    print("=" * 60)

    name = input("Enter rule title/name: ").strip()
    cond_str = input("Enter IF symptoms/facts (comma-separated): ").strip()
    if_all = [c.strip() for c in cond_str.split(",") if c.strip()]

    conseq_str = input("Enter THEN diagnoses/actions (comma-separated): ").strip()
    then_assert = [a.strip() for a in conseq_str.split(",") if a.strip()]

    explanation = input("Enter rationale / physical explanation: ").strip()

    try:
        added = add_diagnostic_rule(rules_path, name, if_all, then_assert, explanation)
        print("\n[SUCCESS]: Rule successfully validated and added to Knowledge Base:")
        print(json.dumps(added, indent=2))
    except Exception as e:
        print(f"\n[ERROR]: Failed to add rule: {e}")
