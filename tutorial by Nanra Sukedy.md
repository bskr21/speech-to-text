# Speech To Text — Tutorial Penggunaan

Dokumen ini menjelaskan cara menyiapkan lingkungan, mengonfigurasi, dan menjalankan pipeline Speech To Text (offline, CPU-only) untuk transkripsi rekaman meeting berdurasi panjang. Pipeline utama diorkestrasi oleh [`scripts.process_audio.main()`](Speech to Text/scripts/process_audio.py:392), dengan utilitas persiapan model di [`scripts.prepare_models.main()`](Speech to Text/scripts/prepare_models.py:115) dan evaluasi kualitas di [`scripts.evaluate_quality.main()`](Speech to Text/scripts/evaluate_quality.py:211).

## 1) Prasyarat Lingkungan
- Sistem operasi: macOS, Linux, atau Windows (disarankan macOS/Linux untuk kemudahan ffmpeg)
- Python: 3.9–3.11
- ffmpeg: harus tersedia di PATH
  - macOS (online): `brew install ffmpeg`
  - Offline: gunakan paket portable ffmpeg dan tambahkan binarinya ke PATH
- Paket Python (offline jika perlu):
  - pyyaml
  - faster-whisper (CTranslate2)
  - ffmpeg-python (opsional)
  - python-docx (opsional, untuk ekspor DOCX)
- Aset model lokal:
  - Whisper CTranslate2 (mis. large-v3, medium, distil-large-v3)
  - Resemblyzer (opsional; untuk diarization di masa depan)
  - RNNoise library (opsional; untuk denoise berbasis RNNoise)

Direktori model default diatur di [`configs/config.yaml`](Speech to Text/configs/config.yaml) bagian `models`.

## 2) Struktur Direktori Proyek
- Input audio: [`data/input/.gitkeep`](Speech to Text/data/input/.gitkeep)
- Output hasil: [`data/output/.gitkeep`](Speech to Text/data/output/.gitkeep)
- File sementara/hasil proses: [`data/processed/.gitkeep`](Speech to Text/data/processed/.gitkeep)
- Cache & checkpoint: [`data/cache/.gitkeep`](Speech to Text/data/cache/.gitkeep)
- Konfigurasi:
  - Utama: [`configs/config.yaml`](Speech to Text/configs/config.yaml)
  - Contoh: [`configs/samples/cpu_offline_id_en.yaml`](Speech to Text/configs/samples/cpu_offline_id_en.yaml)

Catatan: letakkan rekaman audio mentah ke folder `Speech to Text/data/input/`. Format yang didukung: WAV/MP3/M4A/AAC/FLAC/OGG. Pipeline akan melakukan transcode ke 16kHz mono.

## 3) Menyiapkan Model Secara Offline
Gunakan utilitas persiapan model: [`scripts.prepare_models.main()`](Speech to Text/scripts/prepare_models.py:115)

Contoh penggunaan:
- Validasi saja (tanpa copy):
  - `python Speech to Text/scripts/prepare_models.py --dry-run`
- Menyalin Whisper CT2 dari USB ke direktori proyek:
  - `python Speech to Text/scripts/prepare_models.py --whisper-src /Volumes/USB/whisper-ct2/large-v3 --model-name large-v3`
- Menyalin Resemblyzer dan RNNoise:
  - `python Speech to Text/scripts/prepare_models.py --resemblyzer-src /Volumes/USB/resemblyzer --rnnoise-src /Volumes/USB/rnnoise/librnnoise.dylib`

Ekspektasi layout setelah siap:
- `models/whisper/<model_name>/model.bin`
- `models/resemblyzer/...` (berkas aset)
- `models/rnnoise/librnnoise.dylib`

Script akan merangkum kesiapan model dan mengatur environment cache (CT2/WHISPER) ke folder `models/whisper` jika model terdeteksi.

## 4) Konfigurasi Pipeline
File konfigurasi utama: [`configs/config.yaml`](Speech to Text/configs/config.yaml)

Bidang penting yang umum disesuaikan:
- `input.source_path`: path sumber audio (default `data/input`)
- `enhancement.*`: kontrol denoise, normalisasi, spectral gating
- `vad.max_chunk_s`: durasi segmentasi (default ±45 detik)
- `diarization.enable`: saat ini placeholder (semua sebagai SPK1)
- `asr.model_size` dan `asr.compute_type`: pilih model Whisper dan tipe komputasi CPU
- `export.formats`: format keluaran (default JSON & SRT di script; file config mencantumkan juga VTT/DOCX)
- `checkpoint.storage_path`: lokasi file state (mis. `data/cache/state.json`)
- `models.*`: lokasi direktori model lokal

Jika baru mulai, Anda bisa menyalin contoh ke file utama lalu sesuaikan: [`configs/samples/cpu_offline_id_en.yaml`](Speech to Text/configs/samples/cpu_offline_id_en.yaml)

## 5) Menjalankan Pipeline Transkripsi
Langkah eksekusi utama di [`scripts.process_audio.main()`](Speech to Text/scripts/process_audio.py:392):
1. Pastikan rekaman ada di `Speech to Text/data/input/`
2. Jalankan perintah:
   - `python -m scripts.process_audio --config "Speech to Text/configs/config.yaml" --resume`
3. Pipeline tahap demi tahap:
   - Transcode ke 16kHz mono oleh [`scripts.process_audio.transcode_to_wav16k_mono()`](Speech to Text/scripts/process_audio.py:98)
   - Enhancement (denoise/normalize) oleh [`scripts.process_audio.enhance_with_ffmpeg()`](Speech to Text/scripts/process_audio.py:115)
   - Segmentasi dengan ffmpeg oleh [`scripts.process_audio.segment_audio_ffmpeg()`](Speech to Text/scripts/process_audio.py:159)
   - (Placeholder) Diarization penetapan speaker oleh [`scripts.process_audio.process_file()`](Speech to Text/scripts/process_audio.py:294)
   - ASR memuat model dan transkripsi oleh [`scripts.process_audio.initialize_asr()`](Speech to Text/scripts/process_audio.py:195) dan [`scripts.process_audio.transcribe_segments()`](Speech to Text/scripts/process_audio.py:211)
   - Ekspor hasil: JSON oleh [`scripts.process_audio.export_json()`](Speech to Text/scripts/process_audio.py:250) dan SRT oleh [`scripts.process_audio.export_srt()`](Speech to Text/scripts/process_audio.py:265)

Keluaran disimpan di `Speech to Text/data/output/<nama_berkas>/` sebagai `<nama_berkas>.json` dan `<nama_berkas>.srt`.

Checkpoint/resume disimpan di `Speech to Text/data/cache/state.json` untuk melanjutkan proses tanpa mengulang dari awal.

## 6) Evaluasi Kualitas (Opsional)
Gunakan evaluator: [`scripts.evaluate_quality.main()`](Speech to Text/scripts/evaluate_quality.py:211)

Contoh penggunaan:
- Semua berkas:
  - `python Speech to Text/scripts/evaluate_quality.py --config Speech to Text/configs/config.yaml`
- Satu berkas spesifik:
  - `python Speech to Text/scripts/evaluate_quality.py --config Speech to Text/configs/config.yaml --file Speech to Text/data/input/meeting.wav`

Evaluator akan menghitung SNR sebelum/sesudah enhancement, mengukur perubahan dBFS, dan mencoba WER bila ada `reference.txt` di folder output yang sama. Laporan ditulis ke `Speech to Text/data/output/<stem>/quality_report.json`.

## 7) Tips & Troubleshooting
- ffmpeg tidak ditemukan:
  - Pastikan ffmpeg ada di PATH. Pada macOS, cek `which ffmpeg`. Jika offline, gunakan build portable.
- Model Whisper tidak terdeteksi:
  - Pastikan folder `models/whisper/<model>` berisi `model.bin`. Jalankan ulang [`scripts.prepare_models.main()`](Speech to Text/scripts/prepare_models.py:115) dengan `--dry-run` untuk verifikasi.
- Kecepatan transkripsi lambat di CPU:
  - Pertimbangkan model yang lebih ringan (`medium` atau `distil-large-v3`) dan turunkan `asr.compute_type` jika kompatibel.
- Hasil teks kosong pada beberapa segmen:
  - Tingkatkan kualitas audio (pengurangan noise), sesuaikan `enhancement.spectral_gating.threshold_db`, atau kurangi `vad.max_chunk_s`.
- Resume tidak memuat ulang dari awal:
  - Periksa checkpoint di `data/cache/state.json`. Hapus entri berkas terkait bila perlu sebelum menjalankan ulang.

## 8) Ringkasan Alur
1. Siapkan lingkungan & ffmpeg
2. Siapkan model lokal dengan [`scripts.prepare_models.main()`](Speech to Text/scripts/prepare_models.py:115)
3. Sesuaikan [`configs/config.yaml`](Speech to Text/configs/config.yaml) atau gunakan [`configs/samples/cpu_offline_id_en.yaml`](Speech to Text/configs/samples/cpu_offline_id_en.yaml)
4. Taruh audio di [`data/input/.gitkeep`](Speech to Text/data/input/.gitkeep) (folder input)
5. Jalankan pipeline di [`scripts.process_audio.main()`](Speech to Text/scripts/process_audio.py:392)
6. Cek hasil di [`data/output/.gitkeep`](Speech to Text/data/output/.gitkeep) (folder output)
7. (Opsional) Evaluasi kualitas dengan [`scripts.evaluate_quality.main()`](Speech to Text/scripts/evaluate_quality.py:211)
