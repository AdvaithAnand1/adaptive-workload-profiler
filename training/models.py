from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class DetectedProcessContext:
    pid: int
    process_name: str
    executable_name: str
    executable_path: str
    normalized_executable_path: str
    parent_pid: int | None = None
    parent_name: str = ""
    parent_executable_name: str = ""
    parent_executable_path: str = ""
    command_line_markers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "process_name": self.process_name,
            "executable_name": self.executable_name,
            "executable_path": self.executable_path,
            "normalized_executable_path": self.normalized_executable_path,
            "parent_pid": self.parent_pid,
            "parent_name": self.parent_name,
            "parent_executable_name": self.parent_executable_name,
            "parent_executable_path": self.parent_executable_path,
            "command_line_markers": list(self.command_line_markers),
        }


@dataclass(frozen=True)
class TrainingAppRule:
    rule_id: str
    label: str
    display_name: str
    executable_names: tuple[str, ...] = ()
    path_contains: tuple[str, ...] = ()
    parent_names: tuple[str, ...] = ()
    command_line_markers: tuple[str, ...] = ()
    notes: str = ""
    enabled: bool = True

    def matches(self, context: DetectedProcessContext) -> bool:
        if not self.enabled:
            return False

        exe_name = context.executable_name.lower()
        parent_name = context.parent_name.lower()
        normalized_path = context.normalized_executable_path.lower()
        markers = " ".join(context.command_line_markers).lower()

        if self.executable_names and exe_name not in {name.lower() for name in self.executable_names}:
            return False
        if self.path_contains and not any(fragment.lower() in normalized_path for fragment in self.path_contains):
            return False
        if self.parent_names and parent_name not in {name.lower() for name in self.parent_names}:
            return False
        if self.command_line_markers and not all(
            marker.lower() in markers for marker in self.command_line_markers
        ):
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "label": self.label,
            "display_name": self.display_name,
            "executable_names": list(self.executable_names),
            "path_contains": list(self.path_contains),
            "parent_names": list(self.parent_names),
            "command_line_markers": list(self.command_line_markers),
            "notes": self.notes,
            "enabled": self.enabled,
        }


@dataclass(frozen=True)
class TrainingAppCatalog:
    rules: dict[str, TrainingAppRule]
    schema_version: int = 1
    updated_at: str = field(default_factory=_now_iso)

    def match(self, context: DetectedProcessContext) -> TrainingAppRule | None:
        for rule in self.rules.values():
            if rule.matches(context):
                return rule
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "updated_at": self.updated_at,
            "rules": {rule_id: rule.to_dict() for rule_id, rule in sorted(self.rules.items())},
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TrainingAppCatalog":
        if not isinstance(raw, Mapping):
            raise ValueError("training app catalog root must be an object")
        rules_raw = raw.get("rules", {})
        if not isinstance(rules_raw, Mapping):
            raise ValueError("'rules' must be an object")
        rules = {
            str(rule_id): TrainingAppRule(
                rule_id=str(rule_id),
                label=str(rule_raw.get("label", "")).strip().lower(),
                display_name=str(rule_raw.get("display_name", "")).strip(),
                executable_names=tuple(str(x).strip().lower() for x in rule_raw.get("executable_names", []) if str(x).strip()),
                path_contains=tuple(str(x).strip().lower() for x in rule_raw.get("path_contains", []) if str(x).strip()),
                parent_names=tuple(str(x).strip().lower() for x in rule_raw.get("parent_names", []) if str(x).strip()),
                command_line_markers=tuple(str(x).strip().lower() for x in rule_raw.get("command_line_markers", []) if str(x).strip()),
                notes=str(rule_raw.get("notes", "")).strip(),
                enabled=bool(rule_raw.get("enabled", True)),
            )
            for rule_id, rule_raw in rules_raw.items()
        }
        return cls(
            rules=rules,
            schema_version=int(raw.get("schema_version", 1) or 1),
            updated_at=str(raw.get("updated_at", _now_iso())),
        )


@dataclass(frozen=True)
class TrainingSessionManifest:
    session_id: str
    label: str
    proposed_label: str
    approved_label: str
    app_rule_id: str | None = None
    data_path: str = ""
    created_at: str = field(default_factory=_now_iso)
    started_at: str | None = None
    ended_at: str | None = None
    context: DetectedProcessContext | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "label": self.label,
            "proposed_label": self.proposed_label,
            "approved_label": self.approved_label,
            "app_rule_id": self.app_rule_id,
            "data_path": self.data_path,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "context": self.context.to_dict() if self.context else None,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TrainingSessionManifest":
        if not isinstance(raw, Mapping):
            raise ValueError("training session manifest must be an object")
        context_raw = raw.get("context")
        context = None
        if isinstance(context_raw, Mapping):
            context = DetectedProcessContext(
                pid=int(context_raw.get("pid", 0) or 0),
                process_name=str(context_raw.get("process_name", "")).strip(),
                executable_name=str(context_raw.get("executable_name", "")).strip(),
                executable_path=str(context_raw.get("executable_path", "")).strip(),
                normalized_executable_path=str(context_raw.get("normalized_executable_path", "")).strip(),
                parent_pid=(
                    int(context_raw["parent_pid"])
                    if context_raw.get("parent_pid") is not None
                    else None
                ),
                parent_name=str(context_raw.get("parent_name", "")).strip(),
                parent_executable_name=str(context_raw.get("parent_executable_name", "")).strip(),
                parent_executable_path=str(context_raw.get("parent_executable_path", "")).strip(),
                command_line_markers=tuple(
                    str(x).strip().lower()
                    for x in context_raw.get("command_line_markers", [])
                    if str(x).strip()
                ),
            )

        return cls(
            session_id=str(raw.get("session_id", "")).strip(),
            label=str(raw.get("label", "")).strip(),
            proposed_label=str(raw.get("proposed_label", "")).strip(),
            approved_label=str(raw.get("approved_label", "")).strip(),
            app_rule_id=(str(raw["app_rule_id"]).strip() if raw.get("app_rule_id") else None),
            data_path=str(raw.get("data_path", "")).strip(),
            created_at=str(raw.get("created_at", _now_iso())),
            started_at=(str(raw["started_at"]).strip() if raw.get("started_at") else None),
            ended_at=(str(raw["ended_at"]).strip() if raw.get("ended_at") else None),
            context=context,
            notes=str(raw.get("notes", "")).strip(),
        )

