#!/usr/bin/env python3
"""
Offline Voice-to-Text pipeline (CPU-only) for long meeting recordings.
- Ingestion + transcoding (ffmpeg) → 16kHz mono PCM WAV
- Enhancement (ffmpeg filters: afftdn + dynaudnorm) for noise reduction and normalization
- Segmentation (ffmpeg segment muxer) into manageable chunks
- ASR (faster-whisper CPU) with Indonesian-English code-switching
- Optional diarization placeholder (all segments as SPK1; integrate Resemblyzer later)
- Export JSON and SRT with timestamps
- Checkpoint/resume via data/cache/state.json

Requirements (install offline via your internal mirror):
- Python packages: pyyaml, faster-whisper, ffmpeg-python (optional), docx (python-docx; optional for DOCX export)
- ffmpeg binary available in PATH
- Local Whisper models cached under models/whisper (run prepare_models.py once you add it)

Usage:
  python -m scripts.process_audio --config "Speech to Text/configs/config.yaml" --resume

Notes:
- This script prioritizes robustness and clarity with CPU-only constraints.
- Diarization is a placeholder; implement Resemblyzer-based diarization in src/diarization for real separation.
"""
import argparse
import json
import logging
import os
import sys
import uuid
import librosa
from tqdm import tqdm
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import yaml
import subprocess
import shlex

# Attempt to import faster_whisper; if unavailable, raise a clear error later
try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

@dataclass
class Segment:
    id: str
    source_file: str
    chunk_index: int
    start_s: float
    end_s: float
    path: str
    speaker: str = "SPK1"  # placeholder diarization
    text: Optional[str] = None
    language: Optional[str] = None
    confidence: Optional[float] = None


def load_config(cfg_path: Path) -> Dict[str, Any]:
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(log_path: Path, level: str = "INFO") -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def run_cmd(cmd: str) -> Tuple[int, str, str]:
    """Run a shell command, return (code, stdout, stderr)."""
    logging.debug(f"Executing: {cmd}")
    proc = subprocess.Popen(
        shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    out, err = proc.communicate()
    return proc.returncode, out.decode("utf-8", "ignore"), err.decode("utf-8", "ignore")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def discover_audio_files(input_dir: Path) -> List[Path]:
    exts = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}
    files = []
    for p in sorted(input_dir.rglob("*")):
        if p.suffix.lower() in exts and p.is_file():
            files.append(p)
    return files


def transcode_to_wav16k_mono(src: Path, dst_dir: Path, sample_rate: int, channels: int) -> Path:
    ensure_dir(dst_dir)
    out = dst_dir / (src.stem + "_16k_mono.wav")
    # Use ffmpeg to transcode to 16kHz mono PCM s16le WAV
    cmd = (
    f"ffmpeg -y -hide_banner -loglevel error "
    f"-i {shlex.quote(str(src))} "
    f"-ar {sample_rate} -ac {channels} -c:a pcm_s16le "
    f"{shlex.quote(str(out))}"
)
    code, _, err = run_cmd(cmd)
    if code != 0:
        logging.error(f"Transcode failed for {src}: {err}")
        raise RuntimeError(f"ffmpeg transcode error: {err}")
    logging.info(f"Transcoded: {src} -> {out}")
    return out


def speaker_based_segmentation(enhanced_wav: Path, diar_cfg: Dict[str, Any], resemblyzer_dir: str, max_chunk_s: float = 120.0) -> List[Segment]:
    from resemblyzer import VoiceEncoder, preprocess_wav
    import numpy as np
    import librosa
    
    logging.info("Mulai potong berdasarkan ganti speaker...")
    
    encoder = VoiceEncoder(device="cpu")
    
    # Buka audio
    audio, sr = librosa.load(str(enhanced_wav), sr=16000, mono=True)
    
    # Hitung durasi dengan cara aman
    audio_duration = len(audio) / sr
    logging.info(f"Durasi audio: {audio_duration:.2f} detik")
    
    window_s = 5.0
    hop_s = 2.5
    
    embeddings = []
    times = []
    
    for start_s in np.arange(0, audio_duration - window_s, hop_s):
        end_s = start_s + window_s
        chunk = audio[int(start_s * sr):int(end_s * sr)]
        emb = encoder.embed_utterance(chunk)
        embeddings.append(emb)
        times.append((start_s, end_s))
    
    if not embeddings:
        logging.warning("Tidak ada embedding untuk diarization")
        return []
    
    embeddings = np.array(embeddings)
    
    # Clustering seperti sebelumnya, tapi dengan threshold untuk temuin ganti
    from sklearn.cluster import AgglomerativeClustering
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1,  # coba ubah ini kalau terlalu sedikit/banyak speaker
        linkage='ward',
        metric='euclidean'
    )
    labels = clustering.fit_predict(embeddings)
    
    logging.info(f"Ditemukan {len(set(labels))} speaker unik")
    
    # Temuin tempat ganti speaker
    change_points = [0]  # mulai dari awal
    for i in range(1, len(labels)):
        if labels[i] != labels[i-1]:
            change_points.append(times[i][0])  # waktu mulai ganti (detik)
    change_points.append(times[-1][1])  # akhir audio
    
    # Buat segmen berdasarkan ganti, tapi periksa panjang < max_chunk_s
    segments = []
    for idx, (start_s, end_s) in enumerate(zip(change_points[:-1], change_points[1:])):
        dur = end_s - start_s
        if dur > max_chunk_s:
            # Kalau terlalu panjang, potong jadi kecil-kecil
            num_sub = int(np.ceil(dur / max_chunk_s))
            sub_dur = dur / num_sub
            sub_start = start_s
            for j in range(num_sub):
                sub_end = sub_start + sub_dur
                seg = Segment(
                    id=str(uuid.uuid4()),
                    source_file=str(enhanced_wav),
                    chunk_index=len(segments),
                    start_s=sub_start,
                    end_s=sub_end,
                    path=str(enhanced_wav),  # pakai audio besar dulu, nanti potong beneran
                    speaker=f"SPK{labels[idx]}"  # nama speaker sama
                )
                segments.append(seg)
                sub_start = sub_end
        else:
            seg = Segment(
                id=str(uuid.uuid4()),
                source_file=str(enhanced_wav),
                chunk_index=len(segments),
                start_s=start_s,
                end_s=end_s,
                path=str(enhanced_wav),
                speaker=f"SPK{labels[idx]}"
            )
            segments.append(seg)
    
    logging.info(f"Buat {len(segments)} segmen berdasarkan ganti speaker")
    return segments


def enhance_with_ffmpeg(in_wav: Path, out_wav: Path, cfg: Dict[str, Any]) -> Path:
    """
    Enhancement using ffmpeg filters:
    - afftdn: FFT-based denoiser (light spectral gating approximation)
    - highpass: remove low-frequency rumble
    - dynaudnorm: dynamic audio normalization (AGC-like)
    Optional RNNoise via arnndn is not enabled here because it requires a local .model file.
    """
    ensure_dir(out_wav.parent)
    enh = cfg.get("enhancement", {})
    if not enh.get("enable", True):
        # Pass-through copy
        cmd = f"ffmpeg -y -hide_banner -loglevel error -i {shlex.quote(str(in_wav))} -c:a pcm_s16le {shlex.quote(str(out_wav))}"
        code, _, err = run_cmd(cmd)
        if code != 0:
            logging.error(f"Enhancement bypass copy failed: {err}")
            raise RuntimeError("ffmpeg copy failed")
        logging.info(f"Enhancement disabled, copied: {out_wav}")
        return out_wav

    # Build filter chain
    spectral = enh.get("spectral_gating", {})
    # Threshold mapping: use afftdn with default NR level, optionally adjust via config
    threshold_db = spectral.get("threshold_db", -35)
    # Map threshold_db to afftdn nr strength (rough heuristic)
    nr_strength = max(10, min(40, int(abs(threshold_db))))
    filters = [
        f"highpass=f=80",  # rumble removal
        f"afftdn=nr={nr_strength}",  # denoise
        "dynaudnorm=p=1:s=5",  # normalization
    ]
    filter_complex = ",".join(filters)
    cmd = (
        f"ffmpeg -y -hide_banner -loglevel error -i {shlex.quote(str(in_wav))} "
        f"-af {shlex.quote(filter_complex)} -c:a pcm_s16le {shlex.quote(str(out_wav))}"
    )
    code, _, err = run_cmd(cmd)
    if code != 0:
        logging.error(f"Enhancement failed: {err}")
        raise RuntimeError("ffmpeg enhancement failed")
    logging.info(f"Enhanced: {out_wav}")
    return out_wav


def segment_audio_ffmpeg(in_wav: Path, out_dir: Path, max_chunk_s: float) -> List[Segment]:
    """Segment audio into fixed-size chunks using ffmpeg segment muxer."""
    ensure_dir(out_dir)
    pattern = out_dir / "%03d.wav"
    cmd = (
        f"ffmpeg -y -hide_banner -loglevel error -i {shlex.quote(str(in_wav))} "
        f"-f segment -segment_time {max_chunk_s} -c:a pcm_s16le {shlex.quote(str(pattern))}"
    )
    code, _, err = run_cmd(cmd)
    if code != 0:
        logging.error(f"Segmentation failed: {err}")
        raise RuntimeError("ffmpeg segmentation failed")

    # Enumerate produced chunks
    chunks = sorted(out_dir.glob("*.wav"))
    segments: List[Segment] = []
    start = 0.0
    for idx, ch in enumerate(chunks):
        seg_id = str(uuid.uuid4())
        # We don't know exact end without probing; assume contiguous segments
        end = start + max_chunk_s
        segments.append(
            Segment(
                id=seg_id,
                source_file=str(in_wav),
                chunk_index=idx,
                start_s=start,
                end_s=end,
                path=str(ch),
            )
        )
        start = end
    logging.info(f"Created {len(segments)} segments in {out_dir}")
    return segments


def initialize_asr(asr_cfg: Dict[str, Any], models_cfg: Dict[str, Any]):
    if WhisperModel is None:
        raise RuntimeError(
            "faster-whisper is not installed. Install it offline and place models under models/whisper."
        )

    model_size = asr_cfg.get("model_size", "large-v3")
    compute_type = asr_cfg.get("compute_type", "int8")  # safe choice for M3 CPU

    whisper_dir = models_cfg.get("whisper_dir", "models/whisper")

    # Tell faster-whisper to look in our local folder (no internet needed)
    os.environ.setdefault("CT2_CACHE_DIR", whisper_dir)
    os.environ.setdefault("WHISPER_CACHE_DIR", whisper_dir)

    logging.info(
        f"Loading Whisper model {model_size} (compute_type={compute_type}) from {whisper_dir}"
    )

    model_path = Path(whisper_dir) / model_size

    if not (model_path / "model.bin").exists():
        raise FileNotFoundError(f"Cannot find model.bin in {model_path}")

    logging.info(f"Found model.bin in {model_path}")

    model = WhisperModel(
        model_size,
        device="cpu",
        compute_type=compute_type
    )

    return model


def transcribe_segments(model, segments: List[Segment], asr_cfg: Dict[str, Any]) -> List[Segment]:
    from tqdm import tqdm
    language_hints = asr_cfg.get("language_hints", ["id", "en"]) or None
    word_timestamps = asr_cfg.get("word_timestamps", False)
    beam_size = asr_cfg.get("beam_size", 5)

    for seg in tqdm(segments, 
                    desc="Listening with Turbo",  # tulisan di atas jam
                    unit="segment",                               # satuan: potong
                    colour="green"):
        logging.debug(f"Transcribing segment {seg.chunk_index}: {seg.path}")
        try:
            # faster-whisper returns (segments, info)
            # We use decode_options with beam_size and language hints if provided
            opts = {
                "beam_size": beam_size,
            }
            if language_hints:
                # faster-whisper doesn't accept language_hints directly; we can set multilingual and rely on autodetect
                pass
            if word_timestamps:
                opts["word_timestamps"] = True

            res_segments, info = model.transcribe(seg.path, **opts)
            # Concatenate text from segments
            text_parts = []
            lang = None
            confs = []
            for s in res_segments:
                text_parts.append(s.text)
                if hasattr(s, "avg_logprob") and s.avg_logprob is not None:
                    confs.append(s.avg_logprob)
                if hasattr(s, "language") and s.language:
                    lang = s.language
            seg.text = " ".join(text_parts).strip()
            seg.language = lang
            seg.confidence = float(sum(confs) / len(confs)) if confs else None
        except Exception as e:
            logging.error(f"ASR failed for segment {seg.chunk_index}: {e}")
            seg.text = None
    return segments


def export_json(transcript: Dict[str, Any], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)
    logging.info(f"Exported JSON: {out_path}")


def format_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def export_srt(segments: List[Segment], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    lines = []
    for i, seg in enumerate(segments, start=1):
        start = format_srt_time(seg.start_s)
        end = format_srt_time(seg.end_s)
        speaker = seg.speaker
        text = seg.text or "(unrecognized)"
        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(f"{speaker}: {text}")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logging.info(f"Exported SRT: {out_path}")


def save_checkpoint(state_path: Path, state: Dict[str, Any]) -> None:
    ensure_dir(state_path.parent)
    with state_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def load_checkpoint(state_path: Path) -> Dict[str, Any]:
    if not state_path.exists():
        return {}
    with state_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def process_file(src_path: Path, cfg: Dict[str, Any], paths: Dict[str, Path], resume: bool) -> None:
    input_cfg = cfg.get("input", {})
    enh_cfg = cfg.get("enhancement", {})
    vad_cfg = cfg.get("vad", {})
    diar_cfg = cfg.get("diarization", {})
    asr_cfg = cfg.get("asr", {})
    export_cfg = cfg.get("export", {})

    # Directories
    processed_dir = paths["processed_dir"] / src_path.stem
    ensure_dir(processed_dir)

    # Checkpoint
    state_path = paths["state_path"]
    state = load_checkpoint(state_path)
    file_key = str(src_path)
    file_state = state.get(file_key, {})

    # 1) Transcode
    if resume and file_state.get("transcoded"):
        wav_path = Path(file_state["wav_path"])
        logging.info(f"Resume: using existing transcoded {wav_path}")
    else:
        wav_path = transcode_to_wav16k_mono(
            src_path,
            processed_dir,
            input_cfg.get("target_sample_rate", 16000),
            input_cfg.get("target_channels", 1),
        )
        file_state["transcoded"] = True
        file_state["wav_path"] = str(wav_path)
        state[file_key] = file_state
        save_checkpoint(state_path, state)

    # 2) Enhancement
    enhanced_wav = processed_dir / (src_path.stem + "_enhanced.wav")
    if resume and file_state.get("enhanced") and file_state.get("enhanced_wav"):
        enhanced_wav = Path(file_state["enhanced_wav"])
        logging.info(f"Resume: using existing enhanced {enhanced_wav}")
    else:
        enhanced_wav = enhance_with_ffmpeg(wav_path, enhanced_wav, cfg)
        file_state["enhanced"] = True
        file_state["enhanced_wav"] = str(enhanced_wav)
        state[file_key] = file_state
        save_checkpoint(state_path, state)

    # 3) Potong berdasarkan ganti suara (pakai fungsi baru)
    resemblyzer_dir = cfg.get("models", {}).get("resemblyzer_dir", "models/resemblyzer")
    segments = speaker_based_segmentation(enhanced_wav, diar_cfg, resemblyzer_dir, vad_cfg.get("max_chunk_s", 120.0))

    # 4) Potong audio beneran jadi file kecil berdasarkan segmen baru
    seg_dir = processed_dir / "segments"
    ensure_dir(seg_dir)
    for seg in segments:
        seg_path = seg_dir / f"{seg.chunk_index:03d}.wav"
        cmd = f"ffmpeg -y -i {shlex.quote(str(enhanced_wav))} -ss {seg.start_s} -to {seg.end_s} -c:a pcm_s16le {shlex.quote(str(seg_path))}"
        code, _, err = run_cmd(cmd)
        if code != 0:
            logging.error(f"Gagal potong segmen {seg.chunk_index}: {err}")
        else:
            seg.path = str(seg_path)

    file_state["segmented"] = True
    file_state["diarized"] = True
    file_state["segments"] = [asdict(s) for s in segments]
    state[file_key] = file_state
    save_checkpoint(state_path, state)

    # 5) ASR
    if resume and file_state.get("transcribed"):
        segments = [Segment(**s) for s in file_state.get("segments", [])]
        logging.info("Resume: segments already transcribed")
    else:
        model = initialize_asr(asr_cfg, cfg.get("models", {}))
        segments = transcribe_segments(model, segments, asr_cfg)
        file_state["transcribed"] = True
        file_state["segments"] = [asdict(s) for s in segments]
        state[file_key] = file_state
        save_checkpoint(state_path, state)

    print("\n=== DEBUG: SPEAKER SEBELUM EXPORT ===")
    for i, seg in enumerate(segments):
        print(f"Segment {i}: speaker = {seg.speaker}, text = {seg.text[:30] if seg.text else 'kosong'}")

    # 6) Export
    output_dir = paths["output_dir"] / src_path.stem
    ensure_dir(output_dir)
    transcript = {
        "file": str(src_path),
        "segments": [asdict(s) for s in segments],
    }
    formats = export_cfg.get("formats", ["json", "srt"])  # default limited
    if "json" in formats:
        export_json(transcript, output_dir / (src_path.stem + ".json"))
    if "srt" in formats:
        export_srt(segments, output_dir / (src_path.stem + ".srt"))

    logging.info(f"Completed processing for {src_path}")


def main():
    parser = argparse.ArgumentParser(description="Offline Voice-to-Text pipeline (CPU-only)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)

    # Resolve project root relative to the config path
    # Assume the config lives under Speech to Text/configs/...
    project_root = Path(__file__).parent.parent.resolve()

    # Paths
    input_dir = project_root / cfg.get("input", {}).get("source_path", "data/input")
    processed_dir = project_root / "data/processed"
    cache_dir = project_root / "data/cache"
    output_dir = project_root / "data/output"
    state_path = cache_dir / "state.json"
    log_path = Path(cfg.get("logging", {}).get("file_path", str(cache_dir / "pipeline.log")))

    setup_logging(log_path, cfg.get("logging", {}).get("level", "INFO"))
    logging.info(f"Project root: {project_root}")

    for d in [input_dir, processed_dir, cache_dir, output_dir]:
        ensure_dir(d)

    paths = {
        "processed_dir": processed_dir,
        "cache_dir": cache_dir,
        "output_dir": output_dir,
        "state_path": state_path,
    }

    files = discover_audio_files(input_dir)
    if not files:
        logging.warning(f"No audio files found in {input_dir}. Place recordings there.")
        return

    logging.info(f"Found {len(files)} audio files to process")
    for f in files:
        try:
            process_file(f, cfg, paths, resume=args.resume)
        except Exception as e:
            logging.exception(f"Failed processing {f}: {e}")


if __name__ == "__main__":
    main()