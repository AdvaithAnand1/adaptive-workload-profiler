from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import psutil

from .models import DetectedProcessContext, TrainingAppCatalog, TrainingAppRule

CATALOG_FILE = Path("training_app_catalog.json")
CATALOG_EXAMPLE_FILE = Path("training_app_catalog.json.example")


def _default_training_app_catalog() -> TrainingAppCatalog:
    rules = {
        "browser": TrainingAppRule(
            rule_id="browser",
            label="browsing",
            display_name="Browsers",
            executable_names=("chrome.exe", "msedge.exe", "opera.exe", "firefox.exe", "brave.exe"),
            path_contains=("chrome", "edge", "opera", "firefox", "brave"),
            notes="General browser workloads stay in the browsing category.",
        ),
        "office": TrainingAppRule(
            rule_id="office",
            label="office",
            display_name="Office Apps",
            executable_names=("winword.exe", "excel.exe", "powerpnt.exe", "outlook.exe", "libreoffice.exe", "soffice.bin"),
            path_contains=("office", "microsoft office", "libreoffice", "soffice"),
            notes="Office suites and document editors map to office-style workloads.",
        ),
        "media": TrainingAppRule(
            rule_id="media",
            label="media",
            display_name="Media Players",
            executable_names=("vlc.exe", "spotify.exe", "wmplayer.exe", "mpc-hc64.exe", "mpv.exe"),
            path_contains=("vlc", "spotify", "media player", "mpv"),
            notes="Media playback and audio-only sessions map to media.",
        ),
        "lunar": TrainingAppRule(
            rule_id="lunar",
            label="gaming",
            display_name="Lunar Client",
            executable_names=("lunar client.exe", "lunarclient.exe", "javaw.exe"),
            path_contains=("lunar", "lunarclient"),
            parent_names=("lunar client", "lunarclient", "lunar client launcher"),
            command_line_markers=("-jar", "lunar"),
            notes="Generic javaw.exe must not match without Lunar-specific process context.",
        ),
        "dev": TrainingAppRule(
            rule_id="dev",
            label="rendering",
            display_name="Developer Tools",
            executable_names=("code.exe", "devenv.exe", "idea64.exe", "pycharm64.exe", "clion64.exe", "java.exe", "javaw.exe"),
            path_contains=("visual studio", "jetbrains", "vscode", "code", "intellij", "pycharm"),
            notes="Developer tools are conservative stand-ins for heavier mixed workloads.",
        ),
    }
    return TrainingAppCatalog(rules=rules)


def default_training_app_catalog() -> TrainingAppCatalog:
    return _default_training_app_catalog()


def load_training_app_catalog(path: str | Path = CATALOG_FILE) -> TrainingAppCatalog:
    catalog_path = Path(path)
    if not catalog_path.exists():
        if CATALOG_EXAMPLE_FILE.exists():
            raw = json.loads(CATALOG_EXAMPLE_FILE.read_text(encoding="utf-8"))
            return TrainingAppCatalog.from_dict(raw)
        return _default_training_app_catalog()
    raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    return TrainingAppCatalog.from_dict(raw)


def _sanitize_command_line(cmdline: list[str]) -> tuple[str, ...]:
    tokens: list[str] = []
    for token in cmdline:
        raw = str(token).strip().lower()
        if not raw:
            continue
        if len(raw) > 128:
            continue
        if any(marker in raw for marker in {"http://", "https://", "file://"}):
            continue
        tokens.append(raw)
    return tuple(tokens)


def detect_foreground_process() -> DetectedProcessContext | None:
    if os.name != "nt":
        return None

    try:
        import ctypes
        from ctypes import wintypes
    except Exception:
        return None

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return None

    pid = wintypes.DWORD()
    thread_id = user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    if not thread_id or not pid.value:
        return None

    try:
        process = psutil.Process(pid.value)
        parent = process.parent()
        exe_path = process.exe()
        exe_name = Path(exe_path).name
        normalized_path = str(Path(exe_path).expanduser()).replace("/", "\\").lower()
        parent_name = parent.name() if parent else ""
        parent_exe_name = Path(parent.exe()).name if parent else ""
        parent_exe_path = parent.exe() if parent else ""
        cmdline = _sanitize_command_line(process.cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        return None

    return DetectedProcessContext(
        pid=process.pid,
        process_name=process.name(),
        executable_name=exe_name,
        executable_path=exe_path,
        normalized_executable_path=normalized_path,
        parent_pid=parent.pid if parent else None,
        parent_name=parent_name,
        parent_executable_name=parent_exe_name,
        parent_executable_path=parent_exe_path,
        command_line_markers=cmdline,
    )


def match_training_app(
    catalog: TrainingAppCatalog,
    context: DetectedProcessContext | None,
) -> TrainingAppRule | None:
    if context is None:
        return None
    return catalog.match(context)
