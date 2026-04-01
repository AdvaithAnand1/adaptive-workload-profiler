"""
Minimal oracle adapter for profile switching.

Assumes an optional `oracle` module may exist in the future with:
    set_profile(profile: str) -> any
    get_profile() -> str | None  (optional)

For demo use, this file falls back to an in-memory stub.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

Profile = Literal["silent", "balanced", "performance"]
_VALID_PROFILES = {"silent", "balanced", "performance"}

WORKLOAD_TO_PROFILE: dict[str, Profile] = {
    "idle": "silent",
    "light": "balanced",
    "browsing": "balanced",
    "gaming": "performance",
    "rendering": "performance",
    "heavy": "performance",
}


def profile_for_label(label: str) -> Profile:
    return WORKLOAD_TO_PROFILE.get(label.strip().lower(), "balanced")


@dataclass
class OracleResult:
    ok: bool
    applied_profile: Profile | None = None
    message: str = ""


class DemoOracle:
    """In-memory fallback used when no external oracle module is available."""

    def __init__(self):
        self._profile: Profile = "balanced"

    def set_profile(self, profile: Profile):
        self._profile = profile
        return {
            "ok": True,
            "applied_profile": profile,
            "message": "demo stub applied profile",
        }

    def get_profile(self):
        return self._profile


class OracleClient:
    """
    Normalized oracle interface for the app.

    - Tries external `oracle` module first.
    - Falls back to DemoOracle.
    - Converts varying return formats into OracleResult.
    """

    def __init__(self):
        backend, backend_name = self._load_backend()
        self._backend = backend
        self.backend_name = backend_name
        self._last_profile: Profile = "balanced"

    def _load_backend(self):
        try:
            import oracle as oracle_module  # type: ignore
        except Exception:
            return DemoOracle(), "demo_stub"
        return oracle_module, "oracle_module"

    def set_profile(self, profile: Profile, dry_run: bool = False) -> OracleResult:
        if dry_run:
            self._last_profile = profile
            return OracleResult(ok=True, applied_profile=profile, message="dry run")

        try:
            if hasattr(self._backend, "set_profile"):
                raw = self._backend.set_profile(profile)
            elif callable(self._backend):
                raw = self._backend(profile)
            else:
                raise RuntimeError("Oracle backend does not expose set_profile(profile)")
            result = self._normalize_result(raw, requested=profile)
            if result.ok and result.applied_profile is not None:
                self._last_profile = result.applied_profile
            return result
        except Exception as e:
            return OracleResult(ok=False, message=f"{type(e).__name__}: {e}")

    def get_profile(self) -> Profile | None:
        # Keep demo behavior deterministic: return the last profile we accepted.
        # This includes dry-run transitions that intentionally skip backend writes.
        return self._last_profile

    def _normalize_result(self, raw, requested: Profile) -> OracleResult:
        if isinstance(raw, dict):
            ok = bool(raw.get("ok", True))
            applied = self._coerce_profile(
                raw.get("applied_profile"), fallback=requested if ok else None
            )
            message = str(raw.get("message", ""))
            return OracleResult(ok=ok, applied_profile=applied, message=message)

        if isinstance(raw, bool):
            return OracleResult(
                ok=raw,
                applied_profile=requested if raw else None,
            )

        if isinstance(raw, str):
            applied = self._coerce_profile(raw, fallback=requested)
            return OracleResult(ok=True, applied_profile=applied)

        if raw is None:
            return OracleResult(ok=True, applied_profile=requested)

        return OracleResult(
            ok=True,
            applied_profile=requested,
            message=f"unrecognized return type: {type(raw).__name__}",
        )

    @staticmethod
    def _coerce_profile(value, fallback: Profile | None) -> Profile | None:
        if isinstance(value, str):
            norm = value.strip().lower()
            if norm in _VALID_PROFILES:
                return cast(Profile, norm)
        return fallback
