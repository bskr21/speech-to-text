#!/usr/bin/env python3
"""
prepare_models.py

Offline model preparation utility for the Speech to Text project.
- Validates and organizes local model assets for CPU-only offline operation.
- Supports placing Whisper (faster-whisper CTranslate2) models, Resemblyzer resources, and RNNoise library.

What this script does (no internet used):
1) Creates expected directories under project_root/models/
2) Optionally copies your locally available model folders/files into those directories
3) Verifies presence of required files for each component
4) Writes a summary of model readiness

Expected model layout after preparation:
- models/whisper/<model_name>/model.bin (CTranslate2)
- models/resemblyzer/ (resource files for embeddings; e.g., pretrained weights)
- models/rnnoise/librnnoise.dylib (macOS RNNoise native lib)

Usage examples:
  # Dry-run: only report status; do not copy
  python Speech\ to\ Text/scripts/prepare_models.py --dry-run

  # Copy local CTranslate2 whisper model from a USB drive into project models dir
  python Speech\ to\ Text/scripts/prepare_models.py \
    --whisper-src /Volumes/USB/whisper-ct2/large-v3 \
    --model-name large-v3

  # Copy Resemblyzer assets and RNNoise library
  python Speech\ to\ Text/scripts/prepare_models.py \
    --resemblyzer-src /Volumes/USB/resemblyzer \
    --rnnoise-src /Volumes/USB/rnnoise/librnnoise.dylib

Notes:
- This utility performs local file operations only.
- If you already have models in place, use --dry-run to validate.
- Ensure the config points to these folders (see configs/config.yaml models section).
"""
import argparse
import logging
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple

# Optional imports; we avoid hard dependencies here
try:
    from faster_whisper import WhisperModel  # type: ignore
except Exception:
    WhisperModel = None


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_item(src: Path, dst: Path, overwrite: bool = False) -> None:
    if src.is_dir():
        if dst.exists():
            if overwrite:
                shutil.rmtree(dst)
            else:
                raise FileExistsError(f"Destination exists: {dst}. Use --force to overwrite.")
        shutil.copytree(src, dst)
    else:
        ensure_dir(dst.parent)
        if dst.exists() and not overwrite:
            raise FileExistsError(f"Destination file exists: {dst}. Use --force to overwrite.")
        shutil.copy2(src, dst)


def is_ct2_whisper_model_dir(path: Path) -> bool:
    # Minimal check: CTranslate2 model should have model.bin; often also vocabulary.txt, tokenizer.json
    return path.is_dir() and (path / "model.bin").exists()


def verify_whisper_model(whisper_dir: Path, model_name: Optional[str] = None) -> Tuple[bool, Optional[Path]]:
    if model_name:
        candidate = whisper_dir / model_name
        if is_ct2_whisper_model_dir(candidate):
            return True, candidate
    # Else scan for any usable CT2 model
    for p in sorted(whisper_dir.glob("*")):
        if is_ct2_whisper_model_dir(p):
            return True, p
    return False, None


def verify_resemblyzer_dir(res_dir: Path) -> bool:
    # Heuristic: directory exists and contains at least one file (e.g., pretrained weights)
    if not res_dir.exists() or not res_dir.is_dir():
        return False
    files = list(res_dir.glob("**/*"))
    return any(f.is_file() for f in files)


def verify_rnnoise_lib(lib_path: Path) -> bool:
    return lib_path.exists() and lib_path.is_file()


def summarize(whisper_ready: bool, whisper_model_path: Optional[Path], rese_ready: bool, rn_ready: bool) -> None:
    logging.info("\nModel readiness summary:")
    logging.info(f"- Whisper CT2: {'OK' if whisper_ready else 'MISSING'}" + (f" ({whisper_model_path})" if whisper_model_path else ""))
    logging.info(f"- Resemblyzer: {'OK' if rese_ready else 'MISSING'}")
    logging.info(f"- RNNoise lib: {'OK' if rn_ready else 'MISSING'}")


def main():
    parser = argparse.ArgumentParser(description="Offline model preparation for Speech to Text")
    parser.add_argument("--dry-run", action="store_true", help="Only validate; do not copy")
    parser.add_argument("--force", action="store_true", help="Overwrite destination if exists")
    parser.add_argument("--model-name", type=str, default=None, help="Whisper CT2 model name (e.g., large-v3, medium)")
    parser.add_argument("--whisper-src", type=str, default=None, help="Source path to existing CTranslate2 Whisper model dir")
    parser.add_argument("--resemblyzer-src", type=str, default=None, help="Source path to Resemblyzer assets dir")
    parser.add_argument("--rnnoise-src", type=str, default=None, help="Source path to RNNoise native library file (.dylib)")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    args = parser.parse_args()

    setup_logging(args.log_level)

    # Resolve project root as the parent of this script directory
    project_root = Path(__file__).resolve().parents[1]
    models_root = project_root / "models"
    whisper_dir = models_root / "whisper"
    resemblyzer_dir = models_root / "resemblyzer"
    rnnoise_lib_path = models_root / "rnnoise" / "librnnoise.dylib"

    for d in [models_root, whisper_dir, resemblyzer_dir, rnnoise_lib_path.parent]:
        ensure_dir(d)

    # Copy sources if provided and not dry-run
    if not args.dry_run:
        if args.whisper_src:
            src = Path(args.whisper_src)
            if not src.exists():
                logging.error(f"Whisper source not found: {src}")
            else:
                dst = whisper_dir / (args.model_name or src.name)
                logging.info(f"Copying Whisper CT2 model: {src} -> {dst}")
                copy_item(src, dst, overwrite=args.force)

        if args.resemblyzer_src:
            src = Path(args.resemblyzer_src)
            if not src.exists():
                logging.error(f"Resemblyzer source not found: {src}")
            else:
                dst = resemblyzer_dir
                logging.info(f"Copying Resemblyzer assets: {src} -> {dst}")
                copy_item(src, dst, overwrite=args.force)

        if args.rnnoise_src:
            src = Path(args.rnnoise_src)
            if not src.exists() or src.is_dir():
                logging.error(f"RNNoise source must be a file (.dylib): {src}")
            else:
                logging.info(f"Copying RNNoise lib: {src} -> {rnnoise_lib_path}")
                copy_item(src, rnnoise_lib_path, overwrite=args.force)

    # Verify readiness
    whisper_ready, whisper_model_path = verify_whisper_model(whisper_dir, args.model_name)
    rese_ready = verify_resemblyzer_dir(resemblyzer_dir)
    rn_ready = verify_rnnoise_lib(rnnoise_lib_path)
    summarize(whisper_ready, whisper_model_path, rese_ready, rn_ready)

    # Set environment caches to ensure ASR loads locally
    if whisper_ready:
        os.environ.setdefault("CT2_CACHE_DIR", str(whisper_dir))
        os.environ.setdefault("WHISPER_CACHE_DIR", str(whisper_dir))
        logging.info(f"Environment CT2/WHISPER cache set to: {whisper_dir}")

    # Optional: sanity load of Whisper model (CPU) if available
    if whisper_ready and WhisperModel is not None and whisper_model_path is not None:
        try:
            model_name = whisper_model_path.name
            logging.info(f"Sanity loading Whisper model '{model_name}' (device=cpu) ...")
            _ = WhisperModel(model_name, device="cpu", compute_type="int8_float32")
            logging.info("Whisper model loaded successfully.")
        except Exception as e:
            logging.warning(f"Whisper model sanity load failed (this may be normal if model_name is not resolvable by faster-whisper): {e}")

    logging.info("Model preparation finished.")


if __name__ == "__main__":
    main()
