"""
Module B: Fuzzy Logic Reasoning for Box Damage Assessment.
Uses scikit-fuzzy control system to map:
    - weight_deviation: (low, medium, high)
    - crack_severity: (none, minor, major)
to:
    - damage_score: (negligible, moderate, critical)
with centroid defuzzification.
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzyDamageReasoner:
    def __init__(self):
        # Antecedents
        self.weight_dev = ctrl.Antecedent(np.arange(0, 10.1, 0.1), 'weight_deviation')
        self.crack_sev = ctrl.Antecedent(np.arange(0, 10.1, 0.1), 'crack_severity')

        # Consequent
        self.damage_score = ctrl.Consequent(np.arange(0, 100.1, 0.5), 'damage_score')

        # Membership functions
        self.weight_dev['low'] = fuzz.trimf(self.weight_dev.universe, [0, 0, 3.5])
        self.weight_dev['medium'] = fuzz.trimf(self.weight_dev.universe, [2.0, 5.0, 7.5])
        self.weight_dev['high'] = fuzz.trapmf(self.weight_dev.universe, [6.0, 8.0, 10.0, 10.0])

        self.crack_sev['none'] = fuzz.trimf(self.crack_sev.universe, [0, 0, 2.5])
        self.crack_sev['minor'] = fuzz.trimf(self.crack_sev.universe, [1.5, 4.5, 7.5])
        self.crack_sev['major'] = fuzz.trapmf(self.crack_sev.universe, [6.0, 8.0, 10.0, 10.0])

        self.damage_score['negligible'] = fuzz.trimf(self.damage_score.universe, [0, 0, 35])
        self.damage_score['moderate'] = fuzz.trimf(self.damage_score.universe, [25, 50, 75])
        self.damage_score['critical'] = fuzz.trapmf(self.damage_score.universe, [65, 85, 100, 100])

        # Rules
        r1 = ctrl.Rule(self.weight_dev['high'] & self.crack_sev['major'], self.damage_score['critical'])
        r2 = ctrl.Rule(self.crack_sev['major'], self.damage_score['critical'])
        r3 = ctrl.Rule(self.weight_dev['medium'] & self.crack_sev['minor'], self.damage_score['moderate'])
        r4 = ctrl.Rule(self.weight_dev['high'] & self.crack_sev['none'], self.damage_score['moderate'])
        r5 = ctrl.Rule(self.weight_dev['low'] & self.crack_sev['none'], self.damage_score['negligible'])

        self.system = ctrl.ControlSystem([r1, r2, r3, r4, r5])
        self.simulation = ctrl.ControlSystemSimulation(self.system)

    def evaluate(self, weight_deviation: float, crack_severity: float) -> float:
        """
        Computes defuzzified damage score [0 - 100%].
        """
        # Clamp inputs
        w_val = max(0.0, min(10.0, float(weight_deviation)))
        c_val = max(0.0, min(10.0, float(crack_severity)))

        self.simulation.input['weight_deviation'] = w_val
        self.simulation.input['crack_severity'] = c_val

        try:
            self.simulation.compute()
            return float(self.simulation.output['damage_score'])
        except Exception:
            # Fallback if outside rule boundary
            return 50.0
