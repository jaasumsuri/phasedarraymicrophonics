#!/usr/bin/env python3
"""
Build a manifest CSV for MICRECORD recordings.

Expected dataset structure:
MICRECORD/
  <recording_id>/
    INDIV/
      Mic1_<recording_id>.wav
      Mic2_<recording_id>.wav
      Mic3_<recording_id>.wav
"""

from pathlib import Path
import csv
import wave


def wav_info(path: Path):
    """Return (sample_rate_hz, duration_sec) for a WAV file, or (None, None) if unreadable."""
    try:
        with wave.open(str(path), "rb") as wf:
            sr = wf.getframerate()
            nframes = wf.getnframes()
            duration = nframes / sr if sr else None
            return sr, duration
    except Exception:
        return None, None


def main():
    repo_root = Path(__file__).resolve().parents[1]
    micrecord_dir = repo_root / "MICRECORD"
    out_dir = repo_root / "data"
    out_path = out_dir / "manifest.csv"

    if not micrecord_dir.exists():
        raise FileNotFoundError(f"Could not find MICRECORD at: {micrecord_dir}")

    out_dir.mkdir(exist_ok=True)

    rows = []
    max_mic_index_seen = 0

    # Iterate recording directories like 10321800, 10321801, etc.
    for rec_dir in sorted(micrecord_dir.iterdir()):
        if not rec_dir.is_dir() or not rec_dir.name.isdigit():
            continue

        indiv = rec_dir / "INDIV"
        if not indiv.exists():
            continue

        wavs = sorted(indiv.glob("Mic*.wav"))
        if not wavs:
            continue

        # Build mic_index -> wav_path mapping
        mic_map = {}
        for w in wavs:
            name = w.stem  # e.g. "Mic1_10321800"
            if not name.startswith("Mic"):
                continue

            after_mic = name[3:]                # "1_10321800"
            mic_part = after_mic.split("_")[0]  # "1"
            mic_idx = int(mic_part) if mic_part.isdigit() else (len(mic_map) + 1)

            mic_map[mic_idx] = w

        if not mic_map:
            continue

        max_mic_index_seen = max(max_mic_index_seen, max(mic_map.keys()))

        # Use mic1 if available, else first mic, for stats
        ref = mic_map.get(1, mic_map[min(mic_map.keys())])
        sr, dur = wav_info(ref)

        row = {
            "recording_id": rec_dir.name,
            "num_mics": len(mic_map),
            "sample_rate_hz": sr,
            "duration_sec": round(dur, 3) if dur is not None else None,
        }

        # Store paths as relative paths for portability
        for idx, path in sorted(mic_map.items()):
            row[f"mic{idx}_path"] = str(path.relative_to(repo_root))

        rows.append(row)

    # Create headers up to the largest mic index seen
    headers = ["recording_id", "num_mics", "sample_rate_hz", "duration_sec"]
    headers += [f"mic{i}_path" for i in range(1, max_mic_index_seen + 1)]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"✅ Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
