"""
Module B: Dempster-Shafer Evidence Combination.
Frame of Discernment Theta = {Damaged (D), NotDamaged (N)}.
Focal sets: {D}, {N}, and {D, N} (Uncertainty / Epistemic Ignorance).
Computes combined mass function and belief interval [Bel(D), Pl(D)].
"""

from typing import Dict, Tuple, Any, List


class MassFunction:
    def __init__(self, m_d: float, m_n: float, m_u: float):
        total = m_d + m_n + m_u
        if abs(total - 1.0) > 1e-5:
            # normalize
            m_d /= total
            m_n /= total
            m_u /= total
        self.mass = {
            frozenset(['D']): m_d,
            frozenset(['N']): m_n,
            frozenset(['D', 'N']): m_u
        }

    @property
    def m_d(self) -> float:
        return self.mass[frozenset(['D'])]

    @property
    def m_n(self) -> float:
        return self.mass[frozenset(['N'])]

    @property
    def m_u(self) -> float:
        return self.mass[frozenset(['D', 'N'])]

    def combine_with(self, other: 'MassFunction') -> 'MassFunction':
        """Combines two mass functions using Dempster's rule of combination."""
        new_mass = {
            frozenset(['D']): 0.0,
            frozenset(['N']): 0.0,
            frozenset(['D', 'N']): 0.0
        }
        conflict_k = 0.0

        for set1, m1 in self.mass.items():
            for set2, m2 in other.mass.items():
                intersection = set1.intersection(set2)
                product = m1 * m2
                if not intersection:
                    conflict_k += product
                else:
                    new_mass[intersection] += product

        if conflict_k >= 1.0:
            raise ValueError("Total contradiction in Dempster-Shafer combination (K=1.0)")

        # Normalize by 1 - K
        norm = 1.0 - conflict_k
        return MassFunction(
            m_d=new_mass[frozenset(['D'])] / norm,
            m_n=new_mass[frozenset(['N'])] / norm,
            m_u=new_mass[frozenset(['D', 'N'])] / norm
        )

    def belief_interval_damaged(self) -> Tuple[float, float]:
        """
        Bel(D) = m({D})
        Pl(D)  = m({D}) + m({D, N}) = 1 - Bel(N)
        """
        bel = self.m_d
        pl = self.m_d + self.m_u
        return (bel, pl)


class DempsterShaferReasoner:
    def __init__(self):
        # Default sensor models: (m_D, m_N, m_Uncertain) when sensor is positive / negative
        self.sensor_bpa = {
            "weight_anomaly": {
                True: MassFunction(m_d=0.70, m_n=0.05, m_u=0.25),
                False: MassFunction(m_d=0.05, m_n=0.65, m_u=0.30)
            },
            "crack_detected": {
                True: MassFunction(m_d=0.85, m_n=0.02, m_u=0.13),
                False: MassFunction(m_d=0.02, m_n=0.80, m_u=0.18)
            },
            "tilt_triggered": {
                True: MassFunction(m_d=0.65, m_n=0.05, m_u=0.30),
                False: MassFunction(m_d=0.05, m_n=0.70, m_u=0.25)
            }
        }

    def evaluate(self, evidence: Dict[str, bool]) -> Tuple[Tuple[float, float], MassFunction]:
        """
        Combines evidence from all 3 sensors and returns ([Bel, Pl], combined_mass).
        """
        bpas: List[MassFunction] = []
        for sensor, is_active in evidence.items():
            if sensor in self.sensor_bpa:
                bpas.append(self.sensor_bpa[sensor][is_active])

        if not bpas:
            vacuous = MassFunction(m_d=0.0, m_n=0.0, m_u=1.0)
            return vacuous.belief_interval_damaged(), vacuous

        combined = bpas[0]
        for next_bpa in bpas[1:]:
            combined = combined.combine_with(next_bpa)

        return combined.belief_interval_damaged(), combined
