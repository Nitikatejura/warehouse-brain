"""
Module B: Bayesian Network using pgmpy.
5-node DAG modeling:
    - Damaged (root)
    - SensorNoise (root)
    - CrackVisible (child of Damaged)
    - Tilt (child of Damaged)
    - WeightReading (child of Damaged and SensorNoise)
Performs exact inference via Variable Elimination.
"""

import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


class WarehouseBayesNet:
    def __init__(self):
        # 1. Define DAG structure
        self.model = DiscreteBayesianNetwork([
            ('Damaged', 'CrackVisible'),
            ('Damaged', 'Tilt'),
            ('Damaged', 'WeightReading'),
            ('SensorNoise', 'WeightReading')
        ])

        # 2. Define Conditional Probability Distributions (CPDs)
        # Damaged: 0 = No, 1 = Yes
        cpd_damaged = TabularCPD(
            variable='Damaged',
            variable_card=2,
            values=[[0.85], [0.15]],
            state_names={'Damaged': ['No', 'Yes']}
        )

        # SensorNoise: 0 = Low, 1 = High
        cpd_noise = TabularCPD(
            variable='SensorNoise',
            variable_card=2,
            values=[[0.90], [0.10]],
            state_names={'SensorNoise': ['Low', 'High']}
        )

        # CrackVisible: P(CrackVisible | Damaged)
        cpd_crack = TabularCPD(
            variable='CrackVisible',
            variable_card=2,
            values=[
                [0.95, 0.10],  # No crack
                [0.05, 0.90]   # Crack visible
            ],
            evidence=['Damaged'],
            evidence_card=[2],
            state_names={'CrackVisible': ['No', 'Yes'], 'Damaged': ['No', 'Yes']}
        )

        # Tilt: P(Tilt | Damaged)
        cpd_tilt = TabularCPD(
            variable='Tilt',
            variable_card=2,
            values=[
                [0.98, 0.25],  # No tilt
                [0.02, 0.75]   # Tilt triggered
            ],
            evidence=['Damaged'],
            evidence_card=[2],
            state_names={'Tilt': ['No', 'Yes'], 'Damaged': ['No', 'Yes']}
        )

        # WeightReading: P(WeightReading | Damaged, SensorNoise)
        # Damaged: [No, No, Yes, Yes]
        # SensorNoise: [Low, High, Low, High]
        cpd_weight = TabularCPD(
            variable='WeightReading',
            variable_card=2,
            values=[
                [0.92, 0.40, 0.12, 0.05],  # Normal
                [0.08, 0.60, 0.88, 0.95]   # Anomaly
            ],
            evidence=['Damaged', 'SensorNoise'],
            evidence_card=[2, 2],
            state_names={
                'WeightReading': ['Normal', 'Anomaly'],
                'Damaged': ['No', 'Yes'],
                'SensorNoise': ['Low', 'High']
            }
        )

        self.model.add_cpds(cpd_damaged, cpd_noise, cpd_crack, cpd_tilt, cpd_weight)
        self.infer = VariableElimination(self.model)

    def query_damaged_posterior(self, evidence: Dict[str, str]) -> float:
        """
        Runs exact inference using Variable Elimination.
        evidence example: {'CrackVisible': 'Yes', 'Tilt': 'No', 'WeightReading': 'Anomaly'}
        Returns P(Damaged = 'Yes' | evidence).
        """
        result = self.infer.query(variables=['Damaged'], evidence=evidence, show_progress=False)
        prob_yes = result.values[1]  # index 1 corresponds to 'Yes'
        return float(prob_yes)
