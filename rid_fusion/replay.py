"""Bucket imported observations into deterministic replay frames."""

from __future__ import annotations

from dataclasses import dataclass
from .models import RIDObservation


@dataclass(frozen=True)
class ReplayFrame:
    index: int
    timestamp_utc: float
    observations: list[RIDObservation]


class ReplaySession:
    def __init__(self, observations: list[RIDObservation], bucket_s: float = 1.0):
        if bucket_s <= 0:
            raise ValueError("bucket_s must be positive")
        self.observations = sorted(observations, key=lambda o: (o.source_timestamp_utc, o.observation_id))
        self.bucket_s = bucket_s
        self.frames: list[ReplayFrame] = []
        if self.observations:
            start = self.observations[0].source_timestamp_utc
            buckets: dict[int, list[RIDObservation]] = {}
            for observation in self.observations:
                index = int((observation.source_timestamp_utc - start) // bucket_s)
                buckets.setdefault(index, []).append(observation)
            self.frames = [
                ReplayFrame(index, start + index * bucket_s, items)
                for index, items in sorted(buckets.items())
            ]

    def frame(self, index: int) -> ReplayFrame:
        return self.frames[index]

    def summary(self) -> dict:
        if not self.frames:
            return {"frame_count": 0, "start": None, "end": None, "bucket_s": self.bucket_s}
        return {
            "frame_count": len(self.frames),
            "start": self.frames[0].timestamp_utc,
            "end": self.frames[-1].timestamp_utc,
            "bucket_s": self.bucket_s,
        }
