"""
Module B: Bayesian Updating for Box Damage Assessment.
Calculates posterior P(Damaged | Evidence) given prior P(Damaged) and sensor likelihoods.
"""

from typing import Dict, Any


class BayesDamageFilter:
    def __init__(
        self,
        prior_damaged: float = 0.15,
        p_weight_given_dmg: float = 0.85,
        p_weight_given_healthy: float = 0.10,
        p_crack_given_dmg: float = 0.90,
        p_crack_given_healthy: float = 0.05,
        p_tilt_given_dmg: float = 0.75,
        p_tilt_given_healthy: float = 0.02
    ):
        self.prior_d = prior_damaged
        self.prior_not_d = 1.0 - prior_damaged

        self.likelihoods = {
            "weight_anomaly": (p_weight_given_dmg, p_weight_given_healthy),
            "crack_detected": (p_crack_given_dmg, p_crack_given_healthy),
            "tilt_triggered": (p_tilt_given_dmg, p_tilt_given_healthy)
        }

    def compute_posterior(self, evidence: Dict[str, bool]) -> float:
        """
        Computes P(Damaged | evidence) using Bayes' Rule assuming conditional independence.
        P(D | E) = P(D) * P(E | D) / [ P(D) * P(E | D) + P(~D) * P(E | ~D) ]
        """
        p_e_given_d = 1.0
        p_e_given_not_d = 1.0

        for sensor, is_positive in evidence.items():
            if sensor in self.likelihoods:
                p_pos_d, p_pos_not_d = self.likelihoods[sensor]
                if is_positive:
                    p_e_given_d *= p_pos_d
                    p_e_given_not_d *= p_pos_not_d
                else:
                    p_e_given_d *= (1.0 - p_pos_d)
                    p_e_given_not_d *= (1.0 - p_pos_not_d)

        numerator = self.prior_d * p_e_given_d
        denominator = numerator + (self.prior_not_d * p_e_given_not_d)

        if denominator == 0:
            return 0.5
        return numerator / denominator
