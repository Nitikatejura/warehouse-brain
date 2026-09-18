"""
Module D: Connectionist Models CLI (Hopfield Network & RNN Anomaly Predictor).
Usage:
    python anomaly_recall.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from module_d_connectionist.hopfield_network import HopfieldNetwork, generate_warehouse_label_patterns
from module_d_connectionist.rnn_anomaly_detector import RNNAnomalyTrainer
from module_d_connectionist.comparative_analysis import print_connectionist_symbolic_bridge


def run_hopfield_demo():
    print("=" * 75)
    print("PART 1: FROM-SCRATCH HOPFIELD NETWORK (8x8 = 64-BIT ASSOCIATIVE RECALL)")
    print("=" * 75)

    patterns = generate_warehouse_label_patterns()
    pattern_names = list(patterns.keys())
    print(f"Storing {len(patterns)} orthogonal warehouse label patterns into Hopfield Network:")
    for name in pattern_names:
        print(f"  - Pattern: {name}")

    hopfield = HopfieldNetwork(size=64)
    hopfield.train(list(patterns.values()))
    print("Hebbian Training Complete. Zero-diagonal weight matrix W (64x64) constructed.")

    # Test auto-associative recall on corrupted pattern
    target_name = "HAZARD_CROSS"
    target_pattern = patterns[target_name]

    corrupted_pattern, num_flips = HopfieldNetwork.corrupt_pattern(target_pattern, flip_ratio=0.15)
    print(f"\nTesting Associative Recall on '{target_name}':")
    print(f"Injected Noise: {num_flips} bits flipped out of 64 ({num_flips/64*100:.1f}% corruption).")

    print("\n[Original Clean Label Pattern]:")
    print(HopfieldNetwork.render_ascii_8x8(target_pattern))

    print("\n[Corrupted Noisy Sensor Scan]:")
    print(HopfieldNetwork.render_ascii_8x8(corrupted_pattern))

    recalled_pattern, energy_hist, iters = hopfield.recall(corrupted_pattern)

    print(f"\n[Reconstructed Associative Memory Pattern (Converged in {iters} iterations)]:")
    print(HopfieldNetwork.render_ascii_8x8(recalled_pattern))

    is_identical = np.array_equal(recalled_pattern, target_pattern)
    print(f"\nConvergence Energy Trajectory: {['{:.1f}'.format(e) for e in energy_hist]}")
    print(f"Associative Recall Match: {'PERFECT (100% Correct Restoration)' if is_identical else 'FAILED'}")


def run_rnn_demo():
    print("\n" + "=" * 75)
    print("PART 2: RECURRENT NEURAL NETWORK (LSTM) TIME-SERIES ANOMALY PREDICTOR")
    print("=" * 75)

    trainer = RNNAnomalyTrainer(hidden_dim=16, lr=0.01)
    print("Training PyTorch LSTM on 800 sequences (10 timesteps x 4 sensor channels)...")
    eval_metrics = trainer.train_and_evaluate(epochs=12, batch_size=32)

    print(f"Training Complete across {eval_metrics['epochs']} Epochs.")
    print(f"Held-out Test Evaluation ({eval_metrics['num_test_samples']} test sequences):")
    print(f"  - Test Loss:     {eval_metrics['test_loss']:.4f}")
    print(f"  - Test Accuracy: {eval_metrics['test_accuracy']*100:.2f}%")

    # Sample live prediction test
    print("\nEvaluating Live Sequence Test Case:")
    test_seq = np.random.normal(0.2, 0.05, (10, 4)).astype(np.float32)
    # Inject escalating trend into vibration (col 0) and temperature (col 1)
    test_seq[5:, 0] += np.linspace(0.5, 1.5, 5).astype(np.float32)
    test_seq[5:, 1] += np.linspace(1.0, 2.5, 5).astype(np.float32)

    is_anom, prob = trainer.predict_imminent_anomaly(test_seq)
    print(f"  Input Sensor Trajectory: Escalating vibration & heat trend over 10 cycles.")
    print(f"  Prediction: Imminent Anomaly = {is_anom} (Confidence: {prob*100:.1f}%)")


def run_connectionist_demo():
    run_hopfield_demo()
    run_rnn_demo()
    print_connectionist_symbolic_bridge()


if __name__ == "__main__":
    run_connectionist_demo()
