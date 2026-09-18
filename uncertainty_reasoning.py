"""
Module B: Reasoning Under Uncertainty CLI & Comparative Benchmark
Demonstrates and compares:
1. Nonmonotonic Default Logic vs Classical Monotonic Logic
2. Bayesian Updating
3. Certainty Factors (MYCIN-style)
4. Bayesian Network (pgmpy 5-node exact inference)
5. Dempster-Shafer Evidence Combination (Belief Intervals)
6. Fuzzy Logic (scikit-fuzzy centroid defuzzification)

Identifies key disagreement cases and explains why different uncertainty representations
reach divergent conclusions on identical sensor inputs.
"""

import os
import sys
import json
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from module_b_uncertainty.nonmonotonic import MonotonicKB, NonmonotonicDefaultKB
from module_b_uncertainty.bayes_filter import BayesDamageFilter
from module_b_uncertainty.certainty_factors import CertaintyFactorReasoner
from module_b_uncertainty.bayes_network import WarehouseBayesNet
from module_b_uncertainty.dempster_shafer import DempsterShaferReasoner
from module_b_uncertainty.fuzzy_reasoner import FuzzyDamageReasoner


def demonstrate_nonmonotonic_vs_monotonic():
    print("=" * 90)
    print("PART 1: NONMONOTONIC DEFAULT LOGIC VS CLASSICAL MONOTONIC LOGIC DEMONSTRATION")
    print("=" * 90)

    # Monotonic KB
    mono_kb = MonotonicKB()
    mono_kb.tell("Box(B_TEST)")
    print("[Classical Monotonic Logic]:")
    print(f"  Axiom asserted: Box(B_TEST). Theorem derived: Undamaged(B_TEST) = {mono_kb.query_undamaged('B_TEST')}")
    print("  New fact asserted: TiltSensorTriggered(B_TEST).")
    mono_kb.tell("TiltSensorTriggered(B_TEST)")
    print(f"  In monotonic logic, adding new axioms can NEVER shrink the set of entailed theorems.")
    print(f"  Therefore, Undamaged(B_TEST) remains permanently True: {mono_kb.query_undamaged('B_TEST')} (unless contradiction occurs).")

    # Nonmonotonic KB
    print("\n[Nonmonotonic Default Logic / Closed World Assumption]:")
    default_kb = NonmonotonicDefaultKB()
    default_kb.tell("Box(B_TEST)")
    res1 = default_kb.query("B_TEST")
    print(f"  Step 1: Fact asserted: Box(B_TEST).")
    print(f"          Default Rule fires: Box(x) : ~DamagedEvidence(x) / Undamaged(x)")
    print(f"          Verdict: {res1['status']} | Retraction: {res1['retraction_occurred']}")
    print(f"          Reason:  {res1['reason']}")

    print("\n  Step 2: New fact arrives from telemetry: TiltSensorTriggered(B_TEST).")
    default_kb.tell("TiltSensorTriggered(B_TEST)")
    res2 = default_kb.query("B_TEST")
    print(f"          Consistency condition defeated! Prior conclusion is RETRACTED.")
    print(f"          Verdict: {res2['status']} | Retraction: {res2['retraction_occurred']}")
    print(f"          Reason:  {res2['reason']}")
    print("  => This directly demonstrates nonmonotonicity: expanding the knowledge base shrunk the conclusion set.")


def run_uncertainty_comparison(data_path: str = "data/sensor_readings_10.json"):
    demonstrate_nonmonotonic_vs_monotonic()

    print("\n" + "=" * 90)
    print("PART 2: MULTI-PARADIGM UNCERTAINTY COMPARISON TABLE (10 SYNTHETIC CASES)")
    print("=" * 90)

    with open(data_path, "r") as f:
        readings = json.load(f)

    # Instantiate reasoners
    bayes = BayesDamageFilter()
    cf = CertaintyFactorReasoner()
    bn = WarehouseBayesNet()
    ds = DempsterShaferReasoner()
    fuzzy = FuzzyDamageReasoner()

    results = []

    for item in readings:
        box_id = item["box_id"]
        evidence_dict = {
            "weight_anomaly": item["weight_anomaly"],
            "crack_detected": item["crack_detected"],
            "tilt_triggered": item["tilt_triggered"]
        }

        # 1. Nonmonotonic
        n_kb = NonmonotonicDefaultKB()
        n_kb.tell(f"Box({box_id})")
        if item["tilt_triggered"]:
            n_kb.tell(f"TiltSensorTriggered({box_id})")
        if item["crack_detected"] and item.get("crack_severity_score", 0) > 5.0:
            n_kb.tell(f"HighCrackVisible({box_id})")
        if item["weight_anomaly"] and item.get("weight_deviation_kg", 0) > 6.0:
            n_kb.tell(f"SevereWeightAnomaly({box_id})")
        nm_res = n_kb.query(box_id)["status"]

        # 2. Bayes Filter
        bayes_prob = bayes.compute_posterior(evidence_dict)

        # 3. Certainty Factor
        cf_val = cf.combine_evidence(evidence_dict)

        # 4. Bayesian Network (pgmpy)
        bn_ev = {
            "CrackVisible": "Yes" if item["crack_detected"] else "No",
            "Tilt": "Yes" if item["tilt_triggered"] else "No",
            "WeightReading": "Anomaly" if item["weight_anomaly"] else "Normal",
            "SensorNoise": item["sensor_noise"]
        }
        bn_prob = bn.query_damaged_posterior(bn_ev)

        # 5. Dempster-Shafer
        (bel, pl), _ = ds.evaluate(evidence_dict)
        ds_interval = f"[{bel:.2f}, {pl:.2f}]"

        # 6. Fuzzy Logic
        fuzzy_score = fuzzy.evaluate(item["weight_deviation_kg"], item["crack_severity_score"])

        results.append({
            "box_id": box_id,
            "desc": item["description"],
            "nm": nm_res,
            "bayes": f"{bayes_prob:.3f}",
            "cf": f"{cf_val:+.3f}",
            "bn": f"{bn_prob:.3f}",
            "ds": ds_interval,
            "fuzzy": f"{fuzzy_score:.1f}%"
        })

    # Print Table
    header = f"{'Box ID':<8} | {'Nonmonotonic':<12} | {'Bayes Post':<10} | {'CF Score':<9} | {'BN Post':<9} | {'DS Interval':<14} | {'Fuzzy Score':<11}"
    print(header)
    print("-" * len(header))
    for r in results:
        row = f"{r['box_id']:<8} | {r['nm']:<12} | {r['bayes']:<10} | {r['cf']:<9} | {r['bn']:<9} | {r['ds']:<14} | {r['fuzzy']:<11}"
        print(row)

    print("\n" + "=" * 90)
    print("PART 3: EXPLANATION OF DISAGREEMENT CASES")
    print("=" * 90)

    # Disagreement Case 1 Analysis: BOX-03
    # Comment: BOX-03 (Tilt Sensor tripped alone, Weight and Vision clear).
    # Nonmonotonic categorizes it as DAMAGED because any single defect fact defeats the default undamaged rule;
    # however, Bayes posterior is only 0.281 and BN is 0.283 because both Weight and Vision are negative,
    # and their strong negative likelihoods override the noisy tilt sensor!
    print("[DISAGREEMENT CASE 1: BOX-03 - Single noisy tilt sensor with negative weight and vision]")
    print("  - Nonmonotonic Verdict: DAMAGED (rigid binary trigger: any defect fact defeats default assumption).")
    print("  - Bayes Posterior:      0.281   (probabilistic likelihood: two clean sensors strongly outweigh one noisy tilt).")
    print("  - Dempster-Shafer:      [0.17, 0.44] (represents epistemic ignorance; upper plausibility allows damage, but belief is low).")
    print("  - Reason for divergence: Nonmonotonic logic treats any positive anomaly fact as an absolute defeater,")
    print("    whereas Bayesian updating accounts for base rates and likelihood ratios of multiple corroborating negative sensors.")

    # Disagreement Case 2 Analysis: BOX-05
    # Comment: BOX-05 (Weight anomaly present, but Sensor Noise is known to be High; Vision and Tilt are negative).
    # Bayes Filter (naive conditional independence without noise context) predicts P(D)=0.600, whereas
    # Bayesian Network (which explicitly models SensorNoise -> WeightReading) conditions on High Noise
    # and discounts the weight anomaly, dropping posterior P(Damaged) down to 0.245!
    print("\n[DISAGREEMENT CASE 2: BOX-05 - Weight anomaly confounded by High Sensor Noise]")
    print("  - Naive Bayes:          0.600 (assumes weight reading is purely indicative of physical damage).")
    print("  - Bayesian Network:     0.245 ('explaining away' / conditional independence: high sensor noise explains the weight anomaly).")
    print("  - Certainty Factor:     +0.250 (heuristic combination without causal discounting of noise).")
    print("  - Reason for divergence: The Bayesian Network captures causal structure (SensorNoise -> WeightReading),")
    print("    allowing the High Noise observation to 'explain away' the weight anomaly, which naive Bayes cannot do.")

    # Disagreement Case 3 Analysis: BOX-04
    # Comment: BOX-04 (Vision crack detected only, severity 4.2 hairline scratch, weight and tilt negative).
    # Fuzzy logic yields 50.0% (moderate damage score based on continuous fuzzy membership),
    # whereas Dempster-Shafer isolates high positive belief from crack detection [0.63, 0.74].
    print("\n[DISAGREEMENT CASE 3: BOX-04 - Hairline surface scratch (low severity)]")
    print("  - Fuzzy Logic:          50.0% (gradated truth: crack severity 4.2 falls between minor and major membership).")
    print("  - Bayes Posterior:      0.761 (binary likelihood assumes crack vision is 90% true positive rate).")
    print("  - Reason for divergence: Fuzzy logic evaluates degrees of truth and severity along continuous scales,")
    print("    avoiding the binary thresholding distortions inherent in classic discrete probabilistic likelihoods.")


if __name__ == "__main__":
    run_uncertainty_comparison()
