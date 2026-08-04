"""
encoders.py — Protocol-specific encoders and cross-modal attention fusion.

Each protocol (Wi-Fi, BLE, 4G/5G NR) has its own encoder that maps tokens
to a protocol-specific embedding space. A cross-modal attention layer then
fuses these into a protocol-agnostic semantic vector.

Design rationale:
  - Protocol-specific encoders handle the unique field availability of each protocol
  - Cross-modal attention learns which protocol to trust more in which context
    (e.g. 5G NR position > Wi-Fi position, but Wi-Fi velocity > BLE velocity)
  - The output embedding is independent of the input protocol — downstream
    LLM models never need to know which protocol produced the data
"""

from __future__ import annotations
import logging
import math
import numpy as np
from typing import Optional
from rid_fusion.models import SpatioTemporalToken, ProtocolType, FusedToken
from rid_fusion.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class ProtocolSpecificEncoder:
    """
    Applies a protocol-specific projection to token embeddings.
    This allows the model to learn protocol-specific representations
    before the cross-modal fusion step.
    """

    def __init__(self, input_dim: int, hidden_dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.02, (input_dim, hidden_dim))
        self.b = np.zeros(hidden_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (N, input_dim) or (input_dim,) → (N, hidden_dim) or (hidden_dim,)"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
            return (x @ self.W + self.b).flatten()
        return x @ self.W + self.b


class MultiHeadCrossAttention:
    """
    Cross-modal attention: allows tokens from different protocols
    to attend to each other before fusion.

    For an RID fusion scenario:
      - Spatial encoder output serves as Query
      - Temporal encoder output serves as Key
      - Signal encoder output serves as Value
      - Context (weather/geofence) is added as a bias

    This architecture ensures that the fused representation
    understands not just WHERE the drone is, but WHEN it was observed,
    HOW strong the signal was, and IN WHAT environmental conditions.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        seed: int = 0,
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        rng = np.random.default_rng(seed)
        self.W_q = rng.normal(0, 0.02, (d_model, d_model))
        self.W_k = rng.normal(0, 0.02, (d_model, d_model))
        self.W_v = rng.normal(0, 0.02, (d_model, d_model))
        self.W_o = rng.normal(0, 0.02, (d_model, d_model))

    def _scaled_dot_product_attention(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Q,K,V: (batch, seq_len, d_k)"""
        d_k = Q.shape[-1]
        scores = Q @ K.transpose(0, 2, 1) / math.sqrt(d_k)  # (batch, seq_q, seq_k)

        if mask is not None:
            scores = scores + mask

        attn_weights = self._softmax(scores)
        return attn_weights @ V, attn_weights

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x_max = x.max(axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / exp_x.sum(axis=-1, keepdims=True)

    def forward(
        self,
        query: np.ndarray,   # (batch_q, d_model) or (d_model,)
        key: np.ndarray,     # (batch_k, d_model) or (d_model,)
        value: np.ndarray,   # (batch_v, d_model) or (d_model,)
        context_bias: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Multi-head cross-attention fusion.
        
        Handles both single-vector (d_model,) and batched (N, d_model) inputs.
        """
        # Ensure 2D: (d_model,) → (1, d_model)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if key.ndim == 1:
            key = key.reshape(1, -1)
        if value.ndim == 1:
            value = value.reshape(1, -1)

        batch_q = query.shape[0]
        batch_k = key.shape[0]
        batch_v = value.shape[0]

        # Linear projections
        Q = query @ self.W_q   # (batch_q, d_model)
        K = key @ self.W_k     # (batch_k, d_model)
        V = value @ self.W_v   # (batch_v, d_model)

        # Reshape to (batch, 1, d_model) for single-sequence attention
        Q = Q.reshape(batch_q, 1, self.d_model)
        K = K.reshape(batch_k, 1, self.d_model)
        V = V.reshape(batch_v, 1, self.d_model)

        # Broadcast to same batch size (use max)
        max_b = max(batch_q, batch_k, batch_v)
        if batch_q < max_b:
            Q = np.broadcast_to(Q[:1], (max_b, 1, self.d_model))
        if batch_k < max_b:
            K = np.broadcast_to(K[:1], (max_b, 1, self.d_model))
        if batch_v < max_b:
            V = np.broadcast_to(V[:1], (max_b, 1, self.d_model))

        # Scaled dot-product attention: (max_b, 1, 1)
        scores = (Q @ K.transpose(0, 2, 1)) / math.sqrt(self.d_model)
        if mask is not None:
            scores = scores + mask
        attn = self._softmax(scores)

        # Apply attention: (max_b, 1, d_model)
        output = attn @ V

        if context_bias is not None:
            if context_bias.ndim == 1:
                cb = context_bias.reshape(1, 1, -1)
            else:
                cb = context_bias.reshape(-1, 1, self.d_model)
            output = output + cb[:max_b] * 0.1

        # Output projection
        output = output @ self.W_o  # (max_b, 1, d_model)
        return output.squeeze(1)  # (max_b, d_model)


class FusionEncoder:
    """
    Complete fusion pipeline:
      1. Tokenize raw RID observations
      2. Encode with protocol-specific encoders
      3. Cross-modal attention fusion
      4. Output protocol-agnostic semantic vectors (FusedToken)

    The embedding is conceptually split into three modalities
    (spatial, temporal, signal). Rather than a fixed equal-thirds
    division, per-modality projection matrices learn how to extract
    Q/K/V representations from the full token embedding.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        seed: int = 0,
    ):
        self.tokenizer = Tokenizer(hidden_dim=d_model, seed=seed)
        self.d_model = d_model

        rng = np.random.default_rng(seed + 100)

        # Protocol-specific encoders (one per supported protocol)
        self.protocol_encoders = {
            ProtocolType.WIFI_BEACON:  ProtocolSpecificEncoder(d_model, d_model, seed + 1),
            ProtocolType.BLE_ADVB:     ProtocolSpecificEncoder(d_model, d_model, seed + 2),
            ProtocolType.NR_BROADCAST: ProtocolSpecificEncoder(d_model, d_model, seed + 3),
            ProtocolType.LORAWAN:      ProtocolSpecificEncoder(d_model, d_model, seed + 4),
        }

        self.attention = MultiHeadCrossAttention(d_model, n_heads, seed + 10)

        # Per-modality projection matrices — learn which dimensions of the
        # full embedding contribute to Q (spatial), K (temporal), V (signal)
        limit = np.sqrt(6.0 / (d_model * 2))
        self.W_spatial  = rng.uniform(-limit, limit, (d_model, d_model))
        self.W_temporal = rng.uniform(-limit, limit, (d_model, d_model))
        self.W_signal   = rng.uniform(-limit, limit, (d_model, d_model))

    def encode_protocol_specific(
        self,
        tokens: list[SpatioTemporalToken],
    ) -> np.ndarray:
        """Tokenize and apply protocol-specific encoding."""
        if not tokens:
            return np.zeros((0, self.d_model))

        embeddings = self.tokenizer.encode_batch(tokens)
        # Group by protocol and apply protocol-specific encoder
        output = np.zeros((len(tokens), self.d_model))
        for i, token in enumerate(tokens):
            enc = self.protocol_encoders.get(
                token.protocol,
                self.protocol_encoders[ProtocolType.WIFI_BEACON]
            )
            output[i] = enc.forward(embeddings[i])
        return output

    def fuse(
        self,
        wifi_tokens: list[SpatioTemporalToken],
        ble_tokens: list[SpatioTemporalToken],
        nr_tokens: list[SpatioTemporalToken],
        t0: float = 0.0,
        wind_speed: float = 0.0,
        precip: float = 0.0,
        visibility: float = 10000.0,
    ) -> list[FusedToken]:
        """
        Fuse tokens from multiple protocols into protocol-agnostic vectors.

        Fusion is performed **per unique timestamp** so that each FusedToken
        carries a distinct embedding reflecting the cross-modal attention
        over the signals present at that instant.

        Returns:
            list of FusedToken — one per unique drone_id × timestamp
        """
        # 1. Encode each protocol
        wifi_emb = self.encode_protocol_specific(wifi_tokens) if wifi_tokens else np.zeros((0, self.d_model))
        ble_emb  = self.encode_protocol_specific(ble_tokens)  if ble_tokens  else np.zeros((0, self.d_model))
        nr_emb   = self.encode_protocol_specific(nr_tokens)   if nr_tokens   else np.zeros((0, self.d_model))

        all_tokens = wifi_tokens + ble_tokens + nr_tokens
        if not all_tokens:
            return []

        # 2. Build per-timestamp protocol→embedding maps
        wifi_by_ts: dict[float, list[np.ndarray]] = {}
        ble_by_ts: dict[float, list[np.ndarray]] = {}
        nr_by_ts: dict[float, list[np.ndarray]] = {}

        for i, t in enumerate(wifi_tokens):
            ts = t.timestamp_utc
            wifi_by_ts.setdefault(ts, []).append(wifi_emb[i])
        for i, t in enumerate(ble_tokens):
            ts = t.timestamp_utc
            ble_by_ts.setdefault(ts, []).append(ble_emb[i])
        for i, t in enumerate(nr_tokens):
            ts = t.timestamp_utc
            nr_by_ts.setdefault(ts, []).append(nr_emb[i])

        # 3. Fuse per timestamp
        unique_ts = sorted(set(t.timestamp_utc for t in all_tokens))
        result: list[FusedToken] = []

        for ts in unique_ts:
            # Collect embeddings for this timestamp across protocols
            proto_embs: list[np.ndarray] = []
            sources: list[ProtocolType] = []

            for proto_emb_list, proto_type in [
                (wifi_by_ts.get(ts, []), ProtocolType.WIFI_BEACON),
                (ble_by_ts.get(ts, []),  ProtocolType.BLE_ADVB),
                (nr_by_ts.get(ts, []),   ProtocolType.NR_BROADCAST),
            ]:
                if proto_emb_list:
                    # Mean-pool within the same protocol at this timestamp
                    proto_embs.append(np.mean(proto_emb_list, axis=0))
                    sources.append(proto_type)

            if not proto_embs:
                continue

            # Determine drone_id from any token at this timestamp
            ts_tokens = [t for t in all_tokens if abs(t.timestamp_utc - ts) < 1e-3]
            drone_id = ts_tokens[0].drone_id

            # Single-protocol case: no fusion needed, use the embedding directly
            if len(proto_embs) == 1:
                fused_vec = proto_embs[0]
            else:
                # Multi-protocol: cross-modal attention fusion
                stacked = np.stack(proto_embs)  # (n_protocols, d_model)

                # Per-modality projections (learnable) instead of fixed equal-thirds slicing
                Q_raw = stacked @ self.W_spatial   # spatial → query
                K_raw = stacked @ self.W_temporal  # temporal → key
                V_raw = stacked @ self.W_signal    # signal → value

                # Mean-pool across protocols to obtain single Q/K/V vectors
                Q_vec = Q_raw.mean(axis=0)
                K_vec = K_raw.mean(axis=0)
                V_vec = V_raw.mean(axis=0)

                fused_out = self.attention.forward(Q_vec, K_vec, V_vec)
                # fused_out: (1, d_model) after squeeze in MultiHeadCrossAttention.forward
                fused_vec = fused_out.flatten() if fused_out.ndim > 0 else fused_out

            ft = FusedToken(
                drone_id=drone_id,
                timestamp_utc=ts,
                embedding=fused_vec.tolist(),
                source_protocols=sources,
                anomaly_score=0.0,
                risk_level=0.0,
            )
            result.append(ft)

        return result
