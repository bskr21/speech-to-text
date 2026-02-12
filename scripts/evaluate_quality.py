#!/usr/bin/env python3
"""
evaluate_quality.py

Offline quality evaluation for the Speech to Text pipeline.
- Computes SNR (signal-to-noise ratio) before and after enhancement
- Reports dBFS (average loudness) change
- Optionally computes WER if a reference transcript is available

Inputs:
- Uses checkpoint state at Speech to Text/data/cache/state.json to locate original and enhanced WAV paths
- Optionally takes --file to limit evaluation to a specific recording

Outputs:
- Writes JSON reports under Speech to Text/data/output/<stem>/quality_report.json

Usage examples:
  python Speech\ to\ Text/scripts/evaluate_quality.py --config Speech\ to\ Text/configs/config.yaml
  python Speech\ to\ Text/scripts/evaluate_quality.py --config Speech\ to\ Text/configs/config.yaml --file Speech\ to\ Text/data/input/meeting.wav

Notes:
- Pure offline; no network calls
- WER is computed with a simple Levenshtein distance if a reference text file is found:
  - Looks for reference at Speech to Text/data/output/<stem>/reference.txt
  - Hypothesis built by concatenating segment texts from <stem>.json produced by the pipeline
"""
import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import yaml
import wave
import numpy as np


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def load_config(cfg_path: Path) -> Dict[str, Any]:
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_wav_mono(path: Path) -> Tuple[np.ndarray, int]:
    """Read PCM WAV as mono float32 array in [-1, 1] and return (audio, sample_rate)."""
    with wave.open(str(path), "rb") as w:
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        framerate = w.getframerate()
        n_frames = w.getnframes()
        raw = w.readframes(n_frames)
    if sampwidth != 2:
        raise ValueError(f"Expected 16-bit PCM WAV, got sampwidth={sampwidth} for {path}")
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels == 2:
        audio = audio.reshape(-1, 2).mean(axis=1)
    return audio, framerate


def frame_signal(x: np.ndarray, sr: int, frame_ms: float = 20.0, hop_ms: float = 10.0) -> np.ndarray:
    """Create overlapping frames for short-time analysis."""
    frame_len = int(sr * frame_ms / 1000.0)
    hop_len = int(sr * hop_ms / 1000.0)
    if frame_len <= 0 or hop_len <= 0:
        raise ValueError("Invalid frame or hop length")
    n_frames = max(0, 1 + (len(x) - frame_len) // hop_len)
    frames = np.lib.stride_tricks.as_strided(
        x,
        shape=(n_frames, frame_len),
        strides=(x.strides[0] * hop_len, x.strides[0]),
        writeable=False,
    )
    return frames


def estimate_snr(x: np.ndarray, sr: int) -> float:
    """Estimate SNR via short-time energy percentile separation (robust heuristic)."""
    frames = frame_signal(x, sr, frame_ms=20, hop_ms=10)
    if frames.shape[0] == 0:
        return float("nan")
    energies = (frames ** 2).mean(axis=1)
    # Noise floor as median of lowest 20% energies; signal as median of top 20%
    k = max(1, int(0.2 * len(energies)))
    sorted_e = np.sort(energies)
    noise = float(np.median(sorted_e[:k]))
    signal = float(np.median(sorted_e[-k:]))
    noise = max(noise, 1e-12)
    signal = max(signal, noise + 1e-12)
    snr = 10.0 * math.log10(signal / noise)
    return snr


def avg_dbfs(x: np.ndarray) -> float:
    """Average loudness in dBFS using RMS."""
    rms = float(np.sqrt(np.mean(x ** 2)) + 1e-12)
    return 20.0 * math.log10(rms)


def levenshtein(a: List[str], b: List[str]) -> int:
    """Word-level Levenshtein distance."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
    return dp[m][n]


def compute_wer(ref_text: str, hyp_text: str) -> float:
    ref_tokens = ref_text.strip().split()
    hyp_tokens = hyp_text.strip().split()
    if not ref_tokens:
        return float("nan")
    dist = levenshtein(ref_tokens, hyp_tokens)
    return dist / len(ref_tokens)


def load_checkpoint(state_path: Path) -> Dict[str, Any]:
    if not state_path.exists():
        return {}
    with state_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_hypothesis_text(output_dir: Path, stem: str) -> Optional[str]:
    json_path = output_dir / f"{stem}.json"
    if not json_path.exists():
        return None
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        segs = data.get("segments", [])
        parts = [s.get("text", "") for s in segs]
        return " ".join(parts).strip()
    except Exception:
        return None


def find_reference_text(output_dir: Path) -> Optional[str]:
    """Try to find a reference.txt file for WER in the output directory."""
    ref = output_dir / "reference.txt"
    if ref.exists():
        try:
            return ref.read_text(encoding="utf-8").strip()
        except Exception:
            return None
    return None


def evaluate_one(file_key: str, file_state: Dict[str, Any], project_root: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    orig_path = Path(file_state.get("wav_path", ""))
    enh_path = Path(file_state.get("enhanced_wav", ""))
    if not orig_path.exists() or not enh_path.exists():
        raise FileNotFoundError(f"Missing audio paths for {file_key}")

    x_orig, sr_orig = read_wav_mono(orig_path)
    x_enh, sr_enh = read_wav_mono(enh_path)
    if sr_orig != sr_enh:
        logging.warning(f"Sample rate mismatch: {sr_orig} vs {sr_enh}")

    snr_orig = estimate_snr(x_orig, sr_orig)
    snr_enh = estimate_snr(x_enh, sr_enh)
    dbfs_orig = avg_dbfs(x_orig)
    dbfs_enh = avg_dbfs(x_enh)

    stem = orig_path.stem.replace("_16k_mono", "")
    output_dir = project_root / "data" / "output" / stem
    report = {
        "file": file_key,
        "snr_before_db": snr_orig,
        "snr_after_db": snr_enh,
        "snr_improvement_db": (snr_enh - snr_orig) if (not math.isnan(snr_orig) and not math.isnan(snr_enh)) else float("nan"),
        "avg_dbfs_before": dbfs_orig,
        "avg_dbfs_after": dbfs_enh,
        "avg_dbfs_delta": (dbfs_enh - dbfs_orig),
    }

    # Optional WER if reference and hypothesis exist
    hyp_text = build_hypothesis_text(output_dir, stem)
    ref_text = find_reference_text(output_dir)
    if hyp_text and ref_text:
        report["wer"] = compute_wer(ref_text, hyp_text)
    else:
        report["wer"] = None

    # Write report
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "quality_report.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logging.info(f"Quality report written: {out_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Offline quality evaluation (SNR, dBFS, optional WER)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--file", type=str, default=None, help="Specific input audio file to evaluate")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(Path(args.config))

    project_root = Path("Speech to Text").resolve()
    state_rel = cfg.get("checkpoint", {}).get("storage_path", "data/cache/state.json")
    state_path = project_root / state_rel
    state = load_checkpoint(state_path)
    if not state:
        logging.error(f"Checkpoint state not found: {state_path}")
        return

    targets = []
    if args.file:
        fk = Path(args.file).resolve()
        fk_str = str(fk)
        if fk_str in state:
            targets.append((fk_str, state[fk_str]))
        else:
            logging.error(f"File not in checkpoint state: {fk_str}")
            return
    else:
        targets = [(k, v) for k, v in state.items()]

    logging.info(f"Evaluating {len(targets)} recording(s)")
    for file_key, file_state in targets:
        try:
            evaluate_one(file_key, file_state, project_root, cfg)
        except Exception as e:
            logging.exception(f"Evaluation failed for {file_key}: {e}")


if __name__ == "__main__":
    main()
