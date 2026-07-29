"""
meta_learner.py — MAML-based few-shot protocol adapter.

When a new city deploys a previously unknown RID protocol variant
(e.g. proprietary 5G-Advanced sidelink), the meta-learner adapts
the tokenizer and encoder with only 10 labeled samples —
without retraining the downstream fusion or LLM modules.

Algorithm: Model-Agnostic Meta-Learning (MAML, Finn et al. 2017)
applied to the protocol encoding task.

Source domains: Wi-Fi Beacon, BLE ADVB (known protocols)
Target domain:  4G/5G NR Broadcast (unknown protocol, adapted with 10 samples)
"""

from __future__ import annotations
import copy
import logging
import numpy as np
from typing import Optional
from rid_fusion.models import SpatioTemporalToken, ProtocolType

logger = logging.getLogger(__name__)


class MAMLAdapter:
    """
    MAML-based adapter that learns a good initialisation for the
    protocol-specific encoder, so that adapting to a new protocol
    requires only a few gradient steps (inner loop).
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 128,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        n_inner_steps: int = 5,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.n_inner_steps = n_inner_steps

        # The meta-parameters (θ): a projection matrix + bias
        self.W = rng.normal(0, 0.02, (input_dim, hidden_dim))
        self.b = np.zeros(hidden_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (batch, input_dim) → (batch, hidden_dim)"""
        return x @ self.W + self.b

    def _loss(self, embeddings: np.ndarray, targets: np.ndarray) -> float:
        """Mean squared error between predicted and target embeddings."""
        pred = self.forward(embeddings)
        return np.mean((pred - targets) ** 2)

    def _inner_update(
        self,
        W: np.ndarray,
        b: np.ndarray,
        support_x: np.ndarray,
        support_y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Inner loop: gradient descent on support set."""
        for _ in range(self.n_inner_steps):
            # Forward
            pred = support_x @ W + b
            # Gradient w.r.t. W
            grad_W = 2 * support_x.T @ (pred - support_y) / len(support_x)
            grad_b = 2 * np.mean(pred - support_y, axis=0)
            # Update
            W = W - self.inner_lr * grad_W
            b = b - self.inner_lr * grad_b
        return W, b

    def meta_train(
        self,
        source_protocols: list[str],
        source_tokens: list[list[SpatioTemporalToken]],
        source_embeddings: list[np.ndarray],
        n_episodes: int = 100,
    ) -> list[float]:
        """
        Meta-training on known protocols (Wi-Fi, BLE).
        
        Args:
            source_protocols: list of protocol labels
            source_tokens: list of token batches (one per protocol)
            source_embeddings: list of embedding arrays (one per protocol, via Tokenizer)
            n_episodes: number of meta-training episodes
        
        Returns:
            training loss history
        """
        losses = []

        for episode in range(n_episodes):
            # 1. Sample a source protocol
            idx = np.random.randint(0, len(source_protocols))
            all_x = source_embeddings[idx]
            all_y = source_embeddings[idx]  # target = input (auto-encoder style)

            n = len(all_x)
            if n < 12:
                continue

            # 2. Split into support and query sets
            perm = np.random.permutation(n)
            support_idx = perm[:10]
            query_idx = perm[10:20]

            support_x = all_x[support_idx]
            support_y = all_y[support_idx]
            query_x = all_x[query_idx]
            query_y = all_y[query_idx]

            # 3. Inner loop: adapt on support set
            W_adapted, b_adapted = self._inner_update(
                copy.deepcopy(self.W), copy.deepcopy(self.b),
                support_x, support_y,
            )

            # 4. Outer loop: compute loss on query set
            pred = query_x @ W_adapted + b_adapted
            outer_loss = np.mean((pred - query_y) ** 2)

            # 5. Simplified outer-loop update (first-order MAML approximation;
            #    full MAML would backprop through the inner-loop optimisation steps)
            grad_W_outer = 2 * query_x.T @ (pred - query_y) / len(query_x)
            grad_b_outer = 2 * np.mean(pred - query_y, axis=0)

            self.W = self.W - self.outer_lr * grad_W_outer
            self.b = self.b - self.outer_lr * grad_b_outer

            losses.append(float(outer_loss))

        return losses

    def adapt_to_new_protocol(
        self,
        support_tokens: list[SpatioTemporalToken],
        support_embeddings: np.ndarray,
        n_shots: int = 10,
    ) -> "MAMLAdapter":
        """
        Few-shot adaptation to a new protocol.
        Returns a new MAMLAdapter with adapted weights,
        leaving the original (meta-trained) weights intact.
        """
        if len(support_embeddings) < n_shots:
            n_shots = len(support_embeddings)

        W_new, b_new = self._inner_update(
            copy.deepcopy(self.W),
            copy.deepcopy(self.b),
            support_embeddings[:n_shots],
            support_embeddings[:n_shots],  # auto-encoder target
        )

        adapted = MAMLAdapter(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            inner_lr=self.inner_lr,
            outer_lr=self.outer_lr,
            n_inner_steps=self.n_inner_steps,
        )
        adapted.W = W_new
        adapted.b = b_new
        return adapted


def simulate_meta_learning_demo(
    seed: int = 42,
) -> dict:
    """
    Demonstrate meta-learning for cross-city protocol adaptation.

    Scenario:
      - Train on Wi-Fi + BLE tokens (Chengdu)
      - Adapt to 4G/5G NR tokens (Shenzhen) with 10 samples
      - Measure adaptation loss
    """
    from rid_fusion.signals import RIDSignalSimulator, generate_drone_trajectory
    from rid_fusion.tokenizer import Tokenizer

    rng = np.random.default_rng(seed)
    tokenizer = Tokenizer(hidden_dim=128, seed=seed)

    # ── Source domain: Wi-Fi + BLE (Chengdu) ──
    sim_wifi = RIDSignalSimulator([ProtocolType.WIFI_BEACON], seed=seed)
    sim_ble  = RIDSignalSimulator([ProtocolType.BLE_ADVB], seed=seed + 100)

    traj = generate_drone_trajectory(
        "DJI-Chengdu-001",
        start_lat=30.57, start_lon=104.07, start_alt=100.0,
        duration_s=60.0,
    )

    wifi_tokens = []
    ble_tokens = []
    for pt in traj:
        wifi_tokens.extend(sim_wifi.observe(**{k: pt[k] for k in pt if k in pt}))
        ble_tokens.extend(sim_ble.observe(**{k: pt[k] for k in pt if k in pt}))

    wifi_emb = tokenizer.encode_batch(wifi_tokens)
    ble_emb  = tokenizer.encode_batch(ble_tokens)

    # ── Target domain: 4G NR (Shenzhen) ──
    sim_nr = RIDSignalSimulator([ProtocolType.NR_BROADCAST], seed=seed + 200)
    nr_tokens = []
    for pt in traj:
        nr_tokens.extend(sim_nr.observe(**{k: pt[k] for k in pt if k in pt}))
    nr_emb = tokenizer.encode_batch(nr_tokens)

    # ── Meta-train on source protocols ──
    maml = MAMLAdapter(input_dim=128, hidden_dim=128, seed=seed)
    losses = maml.meta_train(
        source_protocols=["WiFi", "BLE"],
        source_tokens=[wifi_tokens, ble_tokens],
        source_embeddings=[wifi_emb, ble_emb],
        n_episodes=50,
    )

    # ── Adapt to new protocol (10-shot) ──
    adapted = maml.adapt_to_new_protocol(
        support_tokens=nr_tokens,
        support_embeddings=nr_emb,
        n_shots=10,
    )

    # ── Evaluate adaptation ──
    # Compare: (a) naive projection (random weights) vs (b) meta-learned adaptation
    rng_rand = np.random.default_rng(seed + 999)
    W_random = rng_rand.normal(0, 0.02, (128, 128))
    loss_random = np.mean((nr_emb[:20] @ W_random - nr_emb[:20]) ** 2)
    loss_adapted = np.mean((nr_emb[:20] @ adapted.W - nr_emb[:20]) ** 2)

    return {
        "meta_train_losses": losses,
        "random_projection_loss": float(loss_random),
        "adapted_loss": float(loss_adapted),
        "improvement_factor": float(loss_random / max(loss_adapted, 1e-8)),
        "n_source_tokens": {"wifi": len(wifi_tokens), "ble": len(ble_tokens)},
        "n_target_tokens": len(nr_tokens),
        "n_adaptation_shots": 10,
    }
