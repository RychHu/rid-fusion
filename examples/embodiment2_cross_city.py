"""
Embodiment 2: Cross-City Protocol Adaptation (10-Shot Meta-Learning)
=====================================================================

Demonstrates the meta-learning claim of the patent:
  - Source domain: Chengdu (Wi-Fi + BLE, known protocols)
  - Target domain: Shenzhen (4G/5G NR, previously unknown protocol)
  - With only 10 labeled samples of the new protocol, the meta-learner
    adapts the tokenizer and encoder without retraining the fusion engine

Usage:
    python examples/embodiment2_cross_city.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rid_fusion.meta_learner import simulate_meta_learning_demo


def main():
    print("=" * 60)
    print("EMBODIMENT 2: Cross-City Meta-Learning Adaptation")
    print("Chengdu (Wi-Fi+BLE) → Shenzhen (4G/5G NR, 10-Shot)")
    print("=" * 60)

    result = simulate_meta_learning_demo(seed=42)

    print("\n--- Source Domain (Chengdu) ---")
    print("  Wi-Fi tokens:  %d" % result["n_source_tokens"]["wifi"])
    print("  BLE tokens:    %d" % result["n_source_tokens"]["ble"])

    print("\n--- Target Domain (Shenzhen) ---")
    print("  4G/5G NR tokens: %d" % result["n_target_tokens"])
    print("  Adaptation shots: %d" % result["n_adaptation_shots"])

    print("\n--- Adaptation Results ---")
    print("  Random projection loss:  %.6f" % result["random_projection_loss"])
    print("  Adapted loss (10-shot):  %.6f" % result["adapted_loss"])
    print("  Improvement factor:      %.1fx" % result["improvement_factor"])

    if result["improvement_factor"] > 1.5:
        verdict = "✓ Meta-learning significantly improves adaptation"
    else:
        verdict = "△ Adaptation benefit is measurable but modest"
    print("  Verdict: %s" % verdict)

    print("\n--- Meta-Training Convergence ---")
    if result["meta_train_losses"]:
        print("  Initial loss: %.6f" % result["meta_train_losses"][0])
        print("  Final loss:   %.6f" % result["meta_train_losses"][-1])
        reduction = (1 - result["meta_train_losses"][-1] / result["meta_train_losses"][0]) * 100
        print("  Reduction:    %.1f%%" % reduction)

    print("\n--- Patent Mapping ---")
    print("  This embodiment demonstrates:")
    print("  1. Meta-learning (MAML) for protocol adapter initialisation")
    print("  2. Source-domain training on known protocols (Wi-Fi + BLE)")
    print("  3. 10-shot adaptation to unknown protocol (4G/5G NR)")
    print("  4. No retraining of downstream fusion engine or LLM")
    print("  5. Cross-city deployment: Chengdu model → Shenzhen with 10 samples")


if __name__ == "__main__":
    main()
