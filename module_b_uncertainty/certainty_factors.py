"""
Module B: Certainty Factors (MYCIN-Style Evidence Combination).
Implements parallel and serial combination rules for reasoning under uncertainty.
"""

from typing import Dict, List, Any


class CertaintyFactorReasoner:
    def __init__(self):
        # Base rule certainty factors: IF Sensor Positive THEN Box Damaged
        self.rule_cfs = {
            "weight_anomaly": 0.75,
            "crack_detected": 0.85,
            "tilt_triggered": 0.70
        }
        # Evidence if sensor is negative (indicates healthy)
        self.negative_cfs = {
            "weight_anomaly": -0.60,
            "crack_detected": -0.75,
            "tilt_triggered": -0.50
        }

    @staticmethod
    def serial_combination(rule_cf: float, evidence_cf: float) -> float:
        """CF(h, e) = CF(rule) * max(0, CF(e))"""
        return rule_cf * max(0.0, evidence_cf)

    @staticmethod
    def parallel_combine_two(cf1: float, cf2: float) -> float:
        """
        Combines two independent CFs for the same hypothesis:
        - If cf1 >= 0 and cf2 >= 0: cf1 + cf2 * (1 - cf1)
        - If cf1 < 0 and cf2 < 0: cf1 + cf2 * (1 + cf1)
        - If signs differ: (cf1 + cf2) / (1 - min(|cf1|, |cf2|))
        """
        if cf1 >= 0 and cf2 >= 0:
            return cf1 + cf2 * (1.0 - cf1)
        elif cf1 < 0 and cf2 < 0:
            return cf1 + cf2 * (1.0 + cf1)
        else:
            denom = 1.0 - min(abs(cf1), abs(cf2))
            if denom == 0:
                return 0.0
            return (cf1 + cf2) / denom

    def combine_evidence(self, evidence: Dict[str, bool]) -> float:
        """
        Evaluates active sensor readings and computes aggregated CF for Damaged hypothesis.
        """
        individual_cfs = []
        for sensor, is_active in evidence.items():
            if is_active:
                rule_cf = self.rule_cfs.get(sensor, 0.5)
                # Assume sensor confidence is 1.0 when active
                cf = self.serial_combination(rule_cf, 1.0)
                individual_cfs.append(cf)
            else:
                neg_cf = self.negative_cfs.get(sensor, -0.5)
                individual_cfs.append(neg_cf)

        if not individual_cfs:
            return 0.0

        current_cf = individual_cfs[0]
        for next_cf in individual_cfs[1:]:
            current_cf = self.parallel_combine_two(current_cf, next_cf)

        return current_cf
