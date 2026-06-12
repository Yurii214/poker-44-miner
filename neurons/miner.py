"""Reference Poker44 miner with a trained ML bot detector."""

import os
import time
from pathlib import Path
from typing import Tuple

import bittensor as bt

from poker44.base.miner import BaseMinerNeuron
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
    sha256_file,
)
from poker44.validator.synapse import DetectionSynapse
from poker44_ml.inference import DEFAULT_MODEL_PATH, Poker44Model

DEFAULT_MAX_POSITIVE_RATE = 0.10
SCORE_CAP_EPSILON = 1e-6


class Miner(BaseMinerNeuron):
    """Serve chunk-level bot-risk scores from a trained benchmark model."""

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        repo_root = Path(__file__).resolve().parents[1]
        model_path = Path(
            getattr(getattr(config, "model", None), "path", None) or DEFAULT_MODEL_PATH
        )
        if not model_path.is_absolute():
            model_path = repo_root / model_path

        self.detector = Poker44Model(model_path=model_path)
        self.max_positive_rate = float(
            self.detector.metadata.get("live_max_positive_rate", DEFAULT_MAX_POSITIVE_RATE)
            or DEFAULT_MAX_POSITIVE_RATE
        )
        metrics = self.detector.metrics or {}
        bt.logging.info(
            f"ML Poker44 Miner started | model={model_path} "
            f"samples={metrics.get('samples', metrics.get('training_samples', 'unknown'))} "
            f"ap={metrics.get('average_precision', 'unknown')} "
            f"fpr={metrics.get('reward_meta', {}).get('fpr', 'unknown')} "
            f"reward={metrics.get('reward', 'unknown')}"
        )

        implementation_files = [
            Path(__file__).resolve(),
            repo_root / "poker44_ml" / "features.py",
            repo_root / "poker44_ml" / "inference.py",
            repo_root / "poker44_ml" / "innovative_model.py",
            repo_root / "poker44_ml" / "stacked.py",
            repo_root / "poker44_ml" / "rank_stack.py",
            repo_root / "poker44_ml" / "calibration.py",
        ]
        repo_url = os.getenv(
            "POKER44_MODEL_REPO_URL",
            "https://github.com/Yurii214/poker-44-miner",
        ).strip()
        artifact_rel = "models/bot_detector_v1.joblib"
        artifact_path = repo_root / artifact_rel
        artifact_sha256 = (
            sha256_file(artifact_path) if artifact_path.is_file() else ""
        )
        data_statement = (
            "Trained on public Poker44 benchmark releases fetched from "
            "https://api.poker44.net/api/v1/benchmark using miner-visible hand payloads."
        )
        self.model_manifest = build_local_model_manifest(
            repo_root=repo_root,
            implementation_files=implementation_files,
            defaults={
                "model_name": self.detector.metadata.get(
                    "model_name",
                    "poker44-reference-stack",
                ),
                "model_version": self.detector.model_version,
                "framework": "lightgbm-xgboost-batch-rank-stack-quantile-remap",
                "license": "MIT",
                "repo_url": repo_url,
                "notes": "Dual-branch benchmark stack with bounded live calibration.",
                "open_source": True,
                "inference_mode": "remote",
                "training_data_statement": data_statement,
                "training_data_sources": [
                    "https://api.poker44.net/api/v1/benchmark/releases",
                ],
                "private_data_attestation": (
                    "This miner does not train on validator-only live evaluation labels."
                ),
                "data_attestation": (
                    "Training data is limited to public Poker44 benchmark releases "
                    "and miner-visible hand payloads from api.poker44.net. "
                    "No validator-private live labels, human PII, or proprietary "
                    "table data were used."
                ),
                "artifact_url": artifact_rel,
                "artifact_sha256": artifact_sha256,
                "model_card_url": f"{repo_url.rstrip('/')}/blob/main/README.md",
            },
        )
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        self._log_manifest_startup(repo_root)
        bt.logging.info(f"Axon created: {self.axon}")

    def _log_manifest_startup(self, repo_root: Path) -> None:
        bt.logging.info("Open-sourced miner manifest standard active for this miner.")
        bt.logging.info(
            f"Miner transparency status: {self.manifest_compliance['status']} "
            f"(missing_fields={self.manifest_compliance['missing_fields']})"
        )
        bt.logging.info(
            f"Manifest summary | model={self.model_manifest.get('model_name', '')} "
            f"version={self.model_manifest.get('model_version', '')} "
            f"repo={self.model_manifest.get('repo_url', '')} "
            f"commit={self.model_manifest.get('repo_commit', '')} "
            f"open_source={self.model_manifest.get('open_source')}"
        )
        bt.logging.info(
            f"Manifest digest={self.manifest_digest} "
            f"inference_mode={self.model_manifest.get('inference_mode', '')}"
        )
        bt.logging.info(
            "Miner prep docs available | "
            f"miner_doc={repo_root / 'docs' / 'miner.md'}"
        )

    def _apply_live_positive_cap(self, scores: list[float]) -> list[float]:
        """Keep live batches under the human-safety cliff while preserving order."""
        if not scores:
            return scores
        max_positive = max(1, int(len(scores) * self.max_positive_rate))
        positive_count = sum(score >= 0.5 for score in scores)
        if positive_count <= max_positive:
            return scores

        sorted_scores = sorted(scores, reverse=True)
        cutoff = sorted_scores[max_positive - 1]
        scale = (0.5 - SCORE_CAP_EPSILON) / max(cutoff, SCORE_CAP_EPSILON)
        capped_scores = [
            score if score >= cutoff else score * scale
            for score in scores
        ]
        capped_positive_count = sum(score >= 0.5 for score in capped_scores)
        bt.logging.warning(
            "Applied live positive cap | "
            f"count={len(scores)} before={positive_count} "
            f"after={capped_positive_count} max_allowed={max_positive} "
            f"rate={self.max_positive_rate:.2f} "
            f"cutoff={cutoff:.6f}"
        )
        return [round(max(0.0, min(1.0, score)), 6) for score in capped_scores]

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        chunks = synapse.chunks or []
        components = self.detector.debug_score_components(chunks)
        raw_scores = components.get("final_scores") or self.detector.predict_chunk_scores(chunks)
        scores = self._apply_live_positive_cap(raw_scores)
        synapse.risk_scores = scores
        synapse.predictions = [score >= 0.5 for score in scores]
        synapse.model_manifest = dict(self.model_manifest)
        if scores:
            sorted_scores = sorted(scores, reverse=True)
            mean_score = sum(scores) / len(scores)
            top_scores = ", ".join(f"{score:.6f}" for score in sorted_scores[:5])
            above_50 = sum(score >= 0.5 for score in scores)
            above_40 = sum(score >= 0.4 for score in scores)
            above_30 = sum(score >= 0.3 for score in scores)
            raw = components.get("raw_scores") or []
            spread = components.get("spread_scores") or []
            if raw and spread:
                bt.logging.info(
                    "Score components | "
                    f"raw_min={min(raw):.6f} raw_max={max(raw):.6f} "
                    f"spread_min={min(spread):.6f} spread_max={max(spread):.6f}"
                )
            bt.logging.info(
                "Score distribution | "
                f"count={len(scores)} min={min(scores):.6f} "
                f"mean={mean_score:.6f} max={max(scores):.6f} "
                f"top5=[{top_scores}] >=0.5={above_50} "
                f">=0.4={above_40} >=0.3={above_30}"
            )
        bt.logging.info(f"Miner predictions: {synapse.predictions}")
        bt.logging.info(f"Scored {len(chunks)} chunks with ML bot-risk scores.")
        return synapse

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("ML miner running...")
        while True:
            bt.logging.info(
                f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}"
            )
            time.sleep(5 * 60)
