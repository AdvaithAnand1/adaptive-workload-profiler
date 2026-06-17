"""Tk editor for unlimited PerfAnalyze power-plan definitions."""

from __future__ import annotations

import tkinter as tk
import queue
import threading
from collections.abc import Callable
from pathlib import Path
from tkinter import messagebox, ttk

from powerplans.catalog import PORTABLE_SETTING_SPECS
from powerplans.editor_state import PowerPlanEditorState
from powerplans.models import (
    PlanDefinition,
    PlanSettingValue,
    default_power_plan_catalog,
)
from powerplans.parser import DiscoveredPowerSetting
from powerplans.store import POWER_PLAN_CATALOG_FILE
from powerplans.windows import PowerCfgDiscovery
from ui_theme import (
    APP_BG,
    MUTED,
    PRIMARY_SOFT,
    SUCCESS,
    SURFACE,
    AutoScrollbar,
    RoundedButton,
    RoundedCard,
    Tooltip,
)


class PowerPlansEditor(ttk.Frame):
    """Scrollable master-detail editor that never mutates Windows power plans."""

    def __init__(
        self,
        parent,
        *,
        catalog_path: str | Path = POWER_PLAN_CATALOG_FILE,
        on_log: Callable[[str], None] | None = None,
        confirm_remove: Callable[[str], bool] | None = None,
    ):
        super().__init__(parent)
        self.on_log = on_log
        self.confirm_remove = confirm_remove or self._confirm_remove
        self.selected_plan_id: str | None = None
        self.support_by_key: dict[str, DiscoveredPowerSetting] | None = None
        self.setting_vars: dict[str, tuple[tk.StringVar, tk.StringVar]] = {}
        self.support_labels: dict[str, ttk.Label] = {}
        self.support_tooltips: dict[str, Tooltip] = {}
        self.plan_buttons: dict[str, RoundedButton] = {}
        self._support_check_running = False
        self._support_results: queue.Queue[
            tuple[dict[str, DiscoveredPowerSetting] | None, str]
        ] = queue.Queue()

        try:
            self.state = PowerPlanEditorState.load(catalog_path)
            initial_status = f"Catalog: {self.state.path}"
        except (OSError, ValueError) as exc:
            self.state = PowerPlanEditorState(
                default_power_plan_catalog(),
                catalog_path,
            )
            initial_status = f"Catalog load issue: {exc}"

        self.status_var = tk.StringVar(value=initial_status)
        self.plan_id_var = tk.StringVar()
        self.name_var = tk.StringVar()
        self.cpu_intent_var = tk.StringVar()
        self.gpu_intent_var = tk.StringVar()
        self.total_power_var = tk.StringVar()
        self.enabled_var = tk.BooleanVar(value=True)
        self.base_scheme_var = tk.StringVar()
        self.windows_guid_var = tk.StringVar()
        self.description_var = tk.StringVar()
        self.mapped_classes_var = tk.StringVar()

        self._build_ui()
        self._render_plan_list()
        if self.state.plan_ids:
            self._select_plan(self.state.plan_ids[0], commit_current=False)
        self.after(150, self._refresh_support)

    def _build_ui(self):
        toolbar = ttk.Frame(self)
        toolbar.pack(fill=tk.X, pady=(0, 14))
        ttk.Label(
            toolbar,
            textvariable=self.status_var,
            justify=tk.LEFT,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.refresh_support_button = RoundedButton(
            toolbar,
            text="Refresh Support",
            command=self._refresh_support,
            canvas_bg=APP_BG,
        )
        self.refresh_support_button.pack(side=tk.RIGHT)
        RoundedButton(
            toolbar,
            text="Reload",
            command=self._reload,
            canvas_bg=APP_BG,
        ).pack(side=tk.RIGHT, padx=(0, 6))
        RoundedButton(
            toolbar,
            text="Save Catalog",
            command=self._save,
            variant="primary",
            canvas_bg=APP_BG,
        ).pack(side=tk.RIGHT, padx=(0, 6))

        body = ttk.Frame(self)
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(0, minsize=320)
        body.columnconfigure(2, weight=1)
        body.rowconfigure(0, weight=1)

        plan_rail = ttk.Frame(body)
        plan_rail.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        plan_rail.rowconfigure(0, weight=1)
        plan_rail.columnconfigure(0, weight=1)

        self.plan_canvas = tk.Canvas(
            plan_rail,
            width=300,
            background=APP_BG,
            highlightthickness=0,
            borderwidth=0,
        )
        plan_scrollbar = AutoScrollbar(
            plan_rail,
            orient="vertical",
            command=self.plan_canvas.yview,
        )
        self.plan_list_frame = ttk.Frame(self.plan_canvas)
        self.plan_canvas_window = self.plan_canvas.create_window(
            (0, 0),
            window=self.plan_list_frame,
            anchor="nw",
        )
        self.plan_canvas.configure(yscrollcommand=plan_scrollbar.set)
        self.plan_canvas.grid(row=0, column=0, sticky="nsew")
        plan_scrollbar.grid(row=0, column=1, sticky="ns")
        self.plan_list_frame.bind(
            "<Configure>",
            lambda _event: self.plan_canvas.configure(
                scrollregion=self.plan_canvas.bbox("all")
            ),
        )
        self.plan_canvas.bind(
            "<Configure>",
            lambda event: self.plan_canvas.itemconfigure(
                self.plan_canvas_window,
                width=event.width,
            ),
        )
        RoundedButton(
            plan_rail,
            text="+ Add Plan",
            command=self._add_plan,
            variant="primary",
            canvas_bg=APP_BG,
        ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        ttk.Separator(body, orient="vertical").grid(
            row=0,
            column=1,
            sticky="ns",
            padx=(0, 10),
        )

        editor_host = ttk.Frame(body)
        editor_host.grid(row=0, column=2, sticky="nsew")
        editor_host.rowconfigure(0, weight=1)
        editor_host.columnconfigure(0, weight=1)

        self.editor_canvas = tk.Canvas(
            editor_host,
            background=APP_BG,
            highlightthickness=0,
            borderwidth=0,
        )
        editor_scrollbar = AutoScrollbar(
            editor_host,
            orient="vertical",
            command=self.editor_canvas.yview,
        )
        self.form_frame = ttk.Frame(self.editor_canvas)
        self.editor_canvas_window = self.editor_canvas.create_window(
            (0, 0),
            window=self.form_frame,
            anchor="nw",
        )
        self.editor_canvas.configure(yscrollcommand=editor_scrollbar.set)
        self.editor_canvas.grid(row=0, column=0, sticky="nsew")
        editor_scrollbar.grid(row=0, column=1, sticky="ns")
        self.form_frame.bind(
            "<Configure>",
            lambda _event: self.editor_canvas.configure(
                scrollregion=self.editor_canvas.bbox("all")
            ),
        )
        self.editor_canvas.bind(
            "<Configure>",
            lambda event: self.editor_canvas.itemconfigure(
                self.editor_canvas_window,
                width=event.width,
            ),
        )
        self.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        self.bind_all("<Button-4>", self._on_mousewheel, add="+")
        self.bind_all("<Button-5>", self._on_mousewheel, add="+")

        self._build_form()
        self.after_idle(self._refresh_scrollregions)

    @staticmethod
    def _is_descendant(widget, ancestor) -> bool:
        while widget is not None:
            if widget == ancestor:
                return True
            widget = getattr(widget, "master", None)
        return False

    def _on_mousewheel(self, event):
        widget = getattr(event, "widget", None)
        if widget is None:
            widget = self.winfo_containing(event.x_root, event.y_root)
        if widget is None:
            return
        if self._is_descendant(widget, self.plan_canvas):
            canvas = self.plan_canvas
        elif self._is_descendant(widget, self.editor_canvas):
            canvas = self.editor_canvas
        else:
            return

        if getattr(event, "num", None) == 4:
            units = -3
        elif getattr(event, "num", None) == 5:
            units = 3
        else:
            delta = int(getattr(event, "delta", 0))
            units = -max(1, abs(delta) // 120) if delta > 0 else max(1, abs(delta) // 120)
        canvas.yview_scroll(units, "units")
        return "break"

    def _refresh_scrollregions(self):
        self.plan_canvas.configure(scrollregion=self.plan_canvas.bbox("all"))
        self.editor_canvas.configure(scrollregion=self.editor_canvas.bbox("all"))

    def _build_form(self):
        identity_card = RoundedCard(
            self.form_frame,
            title="Plan",
            subtitle="Define the identity and performance intent for this profile.",
        )
        identity_card.pack(fill=tk.X)
        identity = identity_card.body
        identity.columnconfigure(1, weight=1)

        fields = (
            ("Plan ID", self.plan_id_var, True),
            ("Name", self.name_var, False),
            ("Mapped classifications", self.mapped_classes_var, True),
            ("CPU intent (0-4)", self.cpu_intent_var, False),
            ("GPU intent (0-4)", self.gpu_intent_var, False),
            ("Total power level (0-4)", self.total_power_var, False),
            ("Base Windows scheme", self.base_scheme_var, False),
            ("Windows scheme GUID (backend managed)", self.windows_guid_var, True),
            ("Description", self.description_var, False),
        )
        for row, (label, variable, readonly) in enumerate(fields):
            ttk.Label(identity, text=label, style="Card.TLabel").grid(
                row=row,
                column=0,
                sticky="w",
                padx=(0, 10),
                pady=3,
            )
            entry = ttk.Entry(identity, textvariable=variable)
            if readonly:
                entry.configure(state="readonly")
            entry.grid(row=row, column=1, sticky="ew", pady=3)

        ttk.Label(identity, text="Enabled", style="Card.TLabel").grid(
            row=len(fields),
            column=0,
            sticky="w",
            padx=(0, 10),
            pady=3,
        )
        ttk.Checkbutton(
            identity,
            text="Available for mapping and activation",
            variable=self.enabled_var,
        ).grid(row=len(fields), column=1, sticky="w", pady=3)

        settings_card = RoundedCard(
            self.form_frame,
            title="Power settings",
        )
        settings_card.pack(fill=tk.X, pady=(14, 0))
        settings_frame = settings_card.body
        settings_frame.columnconfigure(1, weight=1)

        ttk.Label(
            settings_frame,
            text="Parameter",
            style="CardTitle.TLabel",
        ).grid(
            row=0,
            column=0,
            sticky="w",
            padx=(0, 10),
        )
        ttk.Label(
            settings_frame,
            text="Support",
            style="CardTitle.TLabel",
        ).grid(
            row=0,
            column=1,
            sticky="w",
            padx=(0, 10),
        )
        ttk.Label(
            settings_frame,
            text="AC",
            style="CardTitle.TLabel",
        ).grid(row=0, column=2, sticky="w")
        ttk.Label(
            settings_frame,
            text="Battery",
            style="CardTitle.TLabel",
        ).grid(
            row=0,
            column=3,
            sticky="w",
            padx=(8, 0),
        )

        row = 1
        for spec in PORTABLE_SETTING_SPECS:
            ttk.Label(
                settings_frame,
                text=spec.label,
                style="Card.TLabel",
            ).grid(
                row=row,
                column=0,
                sticky="w",
                padx=(0, 10),
                pady=7,
            )

            support_label = ttk.Label(
                settings_frame,
                text="Not checked",
                style="Card.TLabel",
                foreground=MUTED,
                wraplength=230,
                justify=tk.LEFT,
            )
            support_label.grid(
                row=row,
                column=1,
                sticky="w",
                padx=(0, 10),
                pady=7,
            )
            ac_var = tk.StringVar()
            dc_var = tk.StringVar()
            ttk.Entry(
                settings_frame,
                textvariable=ac_var,
                width=10,
            ).grid(row=row, column=2, sticky="w", pady=4)
            ttk.Entry(
                settings_frame,
                textvariable=dc_var,
                width=10,
            ).grid(row=row, column=3, sticky="w", padx=(8, 0), pady=4)
            self.setting_vars[spec.key] = (ac_var, dc_var)
            self.support_labels[spec.key] = support_label
            self.support_tooltips[spec.key] = Tooltip(
                support_label,
                spec.description,
            )
            row += 1

        ttk.Label(
            settings_frame,
            text="Blank values leave a setting unchanged.",
            style="Muted.Card.TLabel",
            justify=tk.LEFT,
            wraplength=720,
        ).grid(
            row=row,
            column=0,
            columnspan=4,
            sticky="w",
            pady=(10, 0),
        )

    def _render_plan_list(self):
        for child in self.plan_list_frame.winfo_children():
            child.destroy()
        self.plan_buttons.clear()

        for row, plan_id in enumerate(self.state.plan_ids):
            plan = self.state.catalog.plans[plan_id]
            item = ttk.Frame(self.plan_list_frame)
            item.grid(row=row, column=0, sticky="ew", pady=(0, 4))
            item.columnconfigure(0, weight=1)

            button = RoundedButton(
                item,
                text=plan.name,
                command=lambda value=plan_id: self._select_plan(value),
                width=248,
                canvas_bg=APP_BG,
            )
            button.grid(row=0, column=0, sticky="ew")
            if plan_id == self.selected_plan_id:
                button.configure(state="disabled", selected=True)
            self.plan_buttons[plan_id] = button

            menu_button = ttk.Menubutton(item, text="...", width=3)
            menu = tk.Menu(
                menu_button,
                tearoff=False,
                background=SURFACE,
                foreground=MUTED,
                activebackground=PRIMARY_SOFT,
                activeforeground="#174ea6",
                relief=tk.FLAT,
                borderwidth=1,
            )
            menu.add_command(
                label="Remove from draft",
                command=lambda value=plan_id: self._remove_plan(value),
            )
            menu_button.configure(menu=menu)
            menu_button.grid(row=0, column=1, padx=(4, 0))

        self.plan_list_frame.columnconfigure(0, weight=1)

    def _select_plan(self, plan_id: str, *, commit_current: bool = True):
        if (
            commit_current
            and self.selected_plan_id is not None
            and not self._commit_current()
        ):
            return
        self.selected_plan_id = plan_id
        self._populate_form(self.state.catalog.plans[plan_id])
        self._render_plan_list()
        self.status_var.set(f"Editing {plan_id}")

    def _populate_form(self, plan: PlanDefinition):
        self.plan_id_var.set(plan.plan_id)
        self.name_var.set(plan.name)
        self.cpu_intent_var.set(str(plan.cpu_intent))
        self.gpu_intent_var.set(str(plan.graphics_intent))
        self.total_power_var.set(str(plan.total_power_level))
        self.enabled_var.set(plan.enabled)
        self.base_scheme_var.set(plan.base_scheme)
        self.windows_guid_var.set(plan.windows_guid or "")
        self.description_var.set(plan.description)
        mapped_classes = sorted(
            label
            for label, plan_id in self.state.catalog.classification_to_plan.items()
            if plan_id == plan.plan_id
        )
        self.mapped_classes_var.set(", ".join(mapped_classes) or "Unmapped")

        for key, (ac_var, dc_var) in self.setting_vars.items():
            value = plan.settings.get(key)
            ac_var.set("" if value is None or value.ac is None else str(value.ac))
            dc_var.set("" if value is None or value.dc is None else str(value.dc))

    @staticmethod
    def _parse_int(raw: str, label: str, *, maximum: int | None = None) -> int:
        try:
            value = int(raw.strip())
        except ValueError as exc:
            raise ValueError(f"{label} must be an integer.") from exc
        if value < 0 or (maximum is not None and value > maximum):
            suffix = f" between 0 and {maximum}" if maximum is not None else " >= 0"
            raise ValueError(f"{label} must be{suffix}.")
        return value

    @classmethod
    def _parse_optional_int(cls, raw: str, label: str) -> int | None:
        if not raw.strip():
            return None
        return cls._parse_int(raw, label)

    def _validate_discovered_value(
        self,
        key: str,
        source: str,
        value: int | None,
    ):
        if value is None or self.support_by_key is None:
            return
        setting = self.support_by_key.get(key.upper())
        if setting is None:
            return

        if setting.options:
            valid_indexes = {option.index for option in setting.options}
            if value not in valid_indexes:
                choices = ", ".join(
                    f"{option.index} ({option.name})"
                    for option in setting.options
                )
                raise ValueError(
                    f"{key} {source} must use a discovered option: {choices}."
                )
            return

        if setting.minimum is not None and value < setting.minimum:
            raise ValueError(
                f"{key} {source} must be at least {setting.minimum}."
            )
        if setting.maximum is not None and value > setting.maximum:
            raise ValueError(
                f"{key} {source} must be at most {setting.maximum}."
            )
        if (
            setting.increment
            and setting.minimum is not None
            and (value - setting.minimum) % setting.increment != 0
        ):
            raise ValueError(
                f"{key} {source} must follow increment {setting.increment} "
                f"from {setting.minimum}."
            )

    def _plan_from_form(self) -> PlanDefinition:
        settings: dict[str, PlanSettingValue] = {}
        for key, (ac_var, dc_var) in self.setting_vars.items():
            ac = self._parse_optional_int(ac_var.get(), f"{key} AC")
            dc = self._parse_optional_int(dc_var.get(), f"{key} Battery")
            self._validate_discovered_value(key, "AC", ac)
            self._validate_discovered_value(key, "Battery", dc)
            if ac is not None or dc is not None:
                settings[key] = PlanSettingValue(ac=ac, dc=dc)

        if self.selected_plan_id is None:
            raise ValueError("No power plan is selected.")
        current_plan = self.state.catalog.plans[self.selected_plan_id]
        return PlanDefinition(
            plan_id=current_plan.plan_id,
            name=self.name_var.get().strip(),
            cpu_intent=self._parse_int(
                self.cpu_intent_var.get(),
                "CPU intent",
                maximum=4,
            ),
            graphics_intent=self._parse_int(
                self.gpu_intent_var.get(),
                "GPU intent",
                maximum=4,
            ),
            total_power_level=self._parse_int(
                self.total_power_var.get(),
                "Total power level",
                maximum=4,
            ),
            enabled=self.enabled_var.get(),
            base_scheme=self.base_scheme_var.get().strip(),
            windows_guid=current_plan.windows_guid,
            description=self.description_var.get().strip(),
            settings=settings,
        )

    def _commit_current(self) -> bool:
        if self.selected_plan_id is None:
            return True
        try:
            plan = self._plan_from_form()
            self.state.update_plan(plan)
        except (KeyError, ValueError) as exc:
            self.status_var.set(f"Cannot keep edits: {exc}")
            return False
        return True

    def _add_plan(self):
        if not self._commit_current():
            return
        plan = self.state.add_plan()
        self._select_plan(plan.plan_id, commit_current=False)
        self.status_var.set(f"Added {plan.plan_id}; save the catalog to persist it.")

    def _remove_plan(self, plan_id: str):
        if plan_id != self.selected_plan_id and not self._commit_current():
            return
        mapped_labels = tuple(
            sorted(
                label
                for label, mapped_plan_id
                in self.state.catalog.classification_to_plan.items()
                if mapped_plan_id == plan_id
            )
        )
        warning = f"Remove '{self.state.catalog.plans[plan_id].name}' from the draft?"
        if mapped_labels:
            warning += (
                "\n\nThe following classification mappings will also be removed:\n"
                + ", ".join(mapped_labels)
            )
        if not self.confirm_remove(warning):
            self.status_var.set(f"Removal canceled for {plan_id}.")
            return
        try:
            removed_mappings = self.state.remove_plan(plan_id)
        except (KeyError, ValueError) as exc:
            self.status_var.set(str(exc))
            return

        next_plan_id = self.state.plan_ids[0]
        self.selected_plan_id = next_plan_id
        self._populate_form(self.state.catalog.plans[next_plan_id])
        self._render_plan_list()
        detail = ""
        if removed_mappings:
            detail = f" Removed mappings: {', '.join(removed_mappings)}."
        self.status_var.set(
            f"Removed {plan_id} from the draft.{detail} Save to persist."
        )

    def _save(self):
        if not self._commit_current():
            return
        try:
            saved_path = self.state.save()
        except (OSError, ValueError) as exc:
            self.status_var.set(f"Save failed: {exc}")
            return
        self.status_var.set(f"Saved catalog to {saved_path}")
        self._emit_log(f"power-plan catalog saved: {saved_path}")

    def _reload(self):
        try:
            self.state.reload()
        except (OSError, ValueError) as exc:
            self.status_var.set(f"Reload failed: {exc}")
            return
        self.selected_plan_id = self.state.plan_ids[0]
        self._populate_form(self.state.catalog.plans[self.selected_plan_id])
        self._render_plan_list()
        self.status_var.set(f"Reloaded catalog from {self.state.path}")

    def _refresh_support(self):
        if self._support_check_running:
            return
        self._support_check_running = True
        self.refresh_support_button.configure(state=tk.DISABLED)
        self.status_var.set("Checking Windows power-setting support...")
        for label in self.support_labels.values():
            label.configure(text="Checking...", foreground=MUTED)

        threading.Thread(
            target=self._discover_support,
            daemon=True,
        ).start()
        self.after(50, self._poll_support_results)

    def _discover_support(self):
        if not PowerCfgDiscovery.is_supported():
            self._support_results.put(
                ({}, "Windows powercfg discovery is unavailable.")
            )
            return
        try:
            snapshot = PowerCfgDiscovery().discover_settings()
        except (OSError, RuntimeError, ValueError) as exc:
            self._support_results.put(
                (None, f"Read-only discovery failed: {exc}")
            )
            return
        support = {
            setting.key.upper(): setting for setting in snapshot.settings
        }
        self._support_results.put(
            (support, f"Support checked from {snapshot.scheme_name}.")
        )

    def _poll_support_results(self):
        try:
            support, message = self._support_results.get_nowait()
        except queue.Empty:
            if self._support_check_running:
                self.after(50, self._poll_support_results)
            return
        self._finish_support_refresh(support, message)

    def _finish_support_refresh(
        self,
        support: dict[str, DiscoveredPowerSetting] | None,
        message: str,
    ):
        self._support_check_running = False
        self.refresh_support_button.configure(state=tk.NORMAL)
        if support is not None:
            self.support_by_key = support
        self._render_support_status()
        self.status_var.set(message)
        self.after_idle(self._refresh_scrollregions)

    def _render_support_status(self):
        for spec in PORTABLE_SETTING_SPECS:
            label = self.support_labels[spec.key]
            if self.support_by_key is None:
                label.configure(text="Not checked", foreground=MUTED)
                self.support_tooltips[spec.key].set_text(spec.description)
                continue
            setting = self.support_by_key.get(spec.key.upper())
            if setting is None:
                label.configure(
                    text="Unavailable",
                    foreground="#9b0000",
                )
                self.support_tooltips[spec.key].set_text(
                    f"{spec.description}\nNot exposed by the active Windows power scheme."
                )
                continue
            status = "Supported"
            if setting.minimum is not None and setting.maximum is not None:
                suffix = setting.units or ""
                status += f" · {setting.minimum}-{setting.maximum}{suffix}"
            elif setting.options:
                status += f" · {len(setting.options)} modes"
            label.configure(
                text=status,
                foreground=SUCCESS,
            )
            tooltip_lines = [
                spec.description,
                f"Current: AC {setting.ac_value}, battery {setting.dc_value}.",
            ]
            if setting.options:
                tooltip_lines.append(
                    "Modes: "
                    + ", ".join(
                        f"{option.index} {option.name}"
                        for option in setting.options
                    )
                )
            self.support_tooltips[spec.key].set_text("\n".join(tooltip_lines))

    def _confirm_remove(self, message: str) -> bool:
        return messagebox.askyesno(
            "Remove Power Plan",
            message,
            parent=self.winfo_toplevel(),
        )

    def _emit_log(self, message: str):
        if self.on_log is not None:
            self.on_log(message)
