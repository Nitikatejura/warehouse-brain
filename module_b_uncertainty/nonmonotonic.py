"""
Module B: Nonmonotonic Reasoning vs Classical Monotonic Logic.
Demonstrates default logic (Closed World / Default Rule) where adding a new fact
retracts a prior conclusion, contrasted with monotonic classical propositional logic.
"""

from typing import Dict, Set, Any, Tuple, Optional


class MonotonicKB:
    """
    Classical monotonic logic knowledge base.
    Once a fact/conclusion is proven, adding new axioms cannot retract it.
    """
    def __init__(self):
        self.facts: Set[str] = set()

    def tell(self, fact: str):
        self.facts.add(fact)

    def query_undamaged(self, box_id: str) -> bool:
        # In monotonic logic, if we have an axiom Box(x) -> Undamaged(x),
        # adding Tilt(x) cannot make Undamaged(x) false without contradiction.
        if f"Box({box_id})" in self.facts:
            return True
        return False


class NonmonotonicDefaultKB:
    """
    Nonmonotonic Default Logic / Closed World Assumption Knowledge Base.
    Default rule:
        Box(x) : ~DamagedEvidence(x)
        ----------------------------
               Undamaged(x)
    Meaning: If x is a box, and it is consistent to assume no damage evidence, infer Undamaged(x).
    If new evidence (e.g. TiltTriggered, HighCrack, WeightAnomaly) is asserted,
    the consistency condition is violated, and Undamaged(x) is retracted!
    """
    def __init__(self):
        self.facts: Set[str] = set()

    def tell(self, fact: str):
        self.facts.add(fact)

    def retract(self, fact: str):
        self.facts.discard(fact)

    def has_damage_evidence(self, box_id: str) -> Tuple[bool, Optional[str]]:
        defect_triggers = [
            f"TiltSensorTriggered({box_id})",
            f"HighCrackVisible({box_id})",
            f"SevereWeightAnomaly({box_id})"
        ]
        for trigger in defect_triggers:
            if trigger in self.facts:
                return True, trigger
        return False, None

    def query(self, box_id: str) -> Dict[str, Any]:
        """
        Evaluates box status under default logic.
        """
        if f"Box({box_id})" not in self.facts:
            return {"status": "Unknown", "reason": "Not recognized as a box"}

        has_defect, trigger = self.has_damage_evidence(box_id)
        if has_defect:
            return {
                "status": "DAMAGED",
                "nonmonotonic_verdict": "DAMAGED",
                "retraction_occurred": True,
                "reason": f"Default Undamaged retracted due to newly asserted fact: {trigger}"
            }
        else:
            # Default rule fires: no defeating evidence found in KB
            return {
                "status": "UNDAMAGED",
                "nonmonotonic_verdict": "UNDAMAGED",
                "retraction_occurred": False,
                "reason": "Default rule: Box is normally undamaged (no defeating evidence present)"
            }
