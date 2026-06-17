from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .models import TrainingSessionManifest


def default_session_manifest_path(csv_path: str | Path) -> Path:
    path = Path(csv_path)
    return path.with_suffix(".manifest.json")


def write_training_session_manifest(
    manifest: TrainingSessionManifest | dict[str, object],
    path: str | Path,
) -> Path:
    out_path = Path(path)
    if isinstance(manifest, TrainingSessionManifest):
        payload = manifest.to_dict()
    else:
        payload = dict(manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out_path


def read_training_session_manifest(path: str | Path) -> TrainingSessionManifest:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return TrainingSessionManifest.from_dict(raw)


def iter_training_session_manifests(root: str | Path) -> Iterable[TrainingSessionManifest]:
    base = Path(root)
    for path in sorted(base.rglob("*.manifest.json")):
        try:
            yield read_training_session_manifest(path)
        except Exception:
            continue

