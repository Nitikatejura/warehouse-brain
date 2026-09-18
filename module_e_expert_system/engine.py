"""
Module E: Forward-Chaining Expert System Rule Engine with Explanation Facility.
Diagnoses warehouse robot hardware and sensor faults using pattern matching.
Generates full deduction traces explaining WHY each conclusion was reached.
"""

import json
from typing import List, Dict, Set, Any, Tuple, Optional


class DiagnosticRule:
    def __init__(self, rule_id: str, name: str, if_all: List[str], then_assert: List[str], explanation: str):
        self.id = rule_id
        self.name = name
        self.if_all = [c.strip().lower() for c in if_all]
        self.then_assert = [a.strip().lower() for a in then_assert]
        self.explanation = explanation

    def is_triggered(self, known_facts: Set[str]) -> bool:
        return all(cond in known_facts for cond in self.if_all)


class FiredRuleRecord:
    def __init__(self, rule: DiagnosticRule, matched_facts: List[str], asserted_facts: List[str], step: int):
        self.rule = rule
        self.matched_facts = matched_facts
        self.asserted_facts = asserted_facts
        self.step = step

    def __repr__(self):
        return f"Step {self.step}: [{self.rule.id}] {self.rule.name} fired on {self.matched_facts} -> Asserted {self.asserted_facts}"


class MaintenanceExpertEngine:
    def __init__(self, rules_file: Optional[str] = None):
        self.rules: List[DiagnosticRule] = []
        self.working_memory: Set[str] = set()
        self.initial_symptoms: Set[str] = set()
        self.fired_rule_history: List[FiredRuleRecord] = []
        self.justification_tree: Dict[str, Dict[str, Any]] = {}

        if rules_file:
            self.load_rules_from_json(rules_file)

    def load_rules_from_json(self, path: str):
        with open(path, "r") as f:
            data = json.load(f)
        self.rules = [
            DiagnosticRule(
                rule_id=r["id"],
                name=r["name"],
                if_all=r["if_all"],
                then_assert=r["then_assert"],
                explanation=r["explanation"]
            )
            for r in data
        ]

    def reset(self):
        self.working_memory.clear()
        self.initial_symptoms.clear()
        self.fired_rule_history.clear()
        self.justification_tree.clear()

    def assert_symptom(self, symptom: str):
        s = symptom.strip().lower()
        self.working_memory.add(s)
        self.initial_symptoms.add(s)
        self.justification_tree[s] = {
            "type": "INITIAL_SYMPTOM",
            "fact": s,
            "source": "Technician Input / Telemetry"
        }

    def infer(self) -> Set[str]:
        """
        Forward chaining until quiescence (no new facts added).
        Applies conflict resolution: rules with more premises (higher specificity) fire first.
        """
        fired_rule_ids = set()
        step = 0

        while True:
            # Find eligible rules that haven't fired yet
            eligible = [
                r for r in self.rules 
                if r.id not in fired_rule_ids and r.is_triggered(self.working_memory)
            ]
            if not eligible:
                break

            # Conflict resolution: sort by specificity (number of conditions descending)
            eligible.sort(key=lambda r: len(r.if_all), reverse=True)
            chosen_rule = eligible[0]
            fired_rule_ids.add(chosen_rule.id)
            step += 1

            new_facts = []
            for fact in chosen_rule.then_assert:
                if fact not in self.working_memory:
                    self.working_memory.add(fact)
                    new_facts.append(fact)
                    self.justification_tree[fact] = {
                        "type": "DERIVED",
                        "fact": fact,
                        "rule_id": chosen_rule.id,
                        "rule_name": chosen_rule.name,
                        "premises": list(chosen_rule.if_all),
                        "explanation": chosen_rule.explanation
                    }

            record = FiredRuleRecord(
                rule=chosen_rule,
                matched_facts=list(chosen_rule.if_all),
                asserted_facts=new_facts,
                step=step
            )
            self.fired_rule_history.append(record)

        return self.working_memory

    def get_diagnoses(self) -> List[str]:
        """Returns all derived conclusions (excluding initial input symptoms)."""
        return sorted(list(self.working_memory - self.initial_symptoms))

    def explain_why(self, target_fact: str, indent: int = 0) -> List[str]:
        """
        Recursively builds an explanation trace explaining WHY target_fact was deduced.
        """
        fact = target_fact.strip().lower()
        lines = []
        prefix = "  " * indent

        if fact not in self.justification_tree:
            lines.append(f"{prefix}- '{fact}' is unknown / not proven.")
            return lines

        node = self.justification_tree[fact]
        if node["type"] == "INITIAL_SYMPTOM":
            lines.append(f"{prefix}* '{fact}' is an observed initial symptom ({node['source']}).")
        else:
            lines.append(f"{prefix}> Conclusion '{fact}' was deduced by [{node['rule_id']}: {node['rule_name']}]")
            lines.append(f"{prefix}  Rationale: {node['explanation']}")
            lines.append(f"{prefix}  Because premises were satisfied:")
            for p in node["premises"]:
                lines.extend(self.explain_why(p, indent + 2))

        return lines
