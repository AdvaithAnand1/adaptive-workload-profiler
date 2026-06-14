"""
Small operator GUI for local Ollama:
- start/stop the local server
- pick a model and send a prompt
- monitor server/process health in real time
"""

from __future__ import annotations

import json
import queue
import shutil
import subprocess
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from urllib import error, request

try:
    import psutil
except ModuleNotFoundError:
    psutil = None


OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_PORT = 11434
MONITOR_INTERVAL_MS = 2000
PSUTIL_AVAILABLE = psutil is not None


class OllamaControlApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Ollama Control Panel")
        self.root.geometry("1100x760")

        self.base_url = OLLAMA_BASE_URL
        self.ollama_process: subprocess.Popen[str] | None = None
        self.log_queue: queue.Queue[str] = queue.Queue()

        self.status_var = tk.StringVar(value="checking...")
        self.pid_var = tk.StringVar(value="-")
        self.uptime_var = tk.StringVar(value="-")
        self.cpu_var = tk.StringVar(value="-")
        self.mem_var = tk.StringVar(value="-")
        self.loaded_var = tk.StringVar(value="-")
        self.last_request_var = tk.StringVar(value="no request yet")
        self.model_var = tk.StringVar(value="")
        self.temperature_var = tk.StringVar(value="0.2")
        self.max_tokens_var = tk.StringVar(value="512")

        self._process_cache: psutil.Process | None = None
        self._process_cache_pid: int | None = None
        self._monitor_after_id: str | None = None
        self._last_tags_refresh = 0.0

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        if not PSUTIL_AVAILABLE:
            self._log("psutil not installed: process-level monitoring is limited")
            self.root.after(
                150,
                lambda: messagebox.showwarning(
                    "Limited Monitoring",
                    "psutil is not installed for this Python runtime.\n"
                    "Server status and prompting still work, but PID/CPU/RAM metrics are limited.\n\n"
                    "Install with: pip install psutil",
                ),
            )
        self._refresh_models_async()
        self._schedule_monitor()

    def _build_ui(self):
        root_frame = ttk.Frame(self.root, padding=10)
        root_frame.pack(fill=tk.BOTH, expand=True)

        server_frame = ttk.LabelFrame(root_frame, text="Server Control", padding=10)
        server_frame.pack(fill=tk.X)

        row1 = ttk.Frame(server_frame)
        row1.pack(fill=tk.X)
        ttk.Label(row1, text="Status").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Label(row1, textvariable=self.status_var).grid(
            row=0,
            column=1,
            sticky="w",
            padx=(0, 12),
        )
        ttk.Label(row1, text="PID").grid(row=0, column=2, sticky="w", padx=(0, 8))
        ttk.Label(row1, textvariable=self.pid_var).grid(row=0, column=3, sticky="w")
        ttk.Label(row1, text="Uptime").grid(row=0, column=4, sticky="w", padx=(16, 8))
        ttk.Label(row1, textvariable=self.uptime_var).grid(row=0, column=5, sticky="w")
        ttk.Label(row1, text="CPU").grid(row=0, column=6, sticky="w", padx=(16, 8))
        ttk.Label(row1, textvariable=self.cpu_var).grid(row=0, column=7, sticky="w")
        ttk.Label(row1, text="RAM").grid(row=0, column=8, sticky="w", padx=(16, 8))
        ttk.Label(row1, textvariable=self.mem_var).grid(row=0, column=9, sticky="w")
        ttk.Label(row1, text="Loaded").grid(row=0, column=10, sticky="w", padx=(16, 8))
        ttk.Label(row1, textvariable=self.loaded_var).grid(row=0, column=11, sticky="w")

        row2 = ttk.Frame(server_frame)
        row2.pack(fill=tk.X, pady=(8, 0))
        self.start_btn = ttk.Button(row2, text="Start Ollama", command=self._start_ollama)
        self.stop_btn = ttk.Button(row2, text="Stop Ollama", command=self._stop_ollama)
        self.refresh_btn = ttk.Button(
            row2,
            text="Refresh Models",
            command=self._refresh_models_async,
        )
        self.start_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.refresh_btn.pack(side=tk.LEFT, padx=(0, 6))
        ttk.Label(
            row2,
            text=f"API: {OLLAMA_BASE_URL} (no /v1)",
        ).pack(side=tk.LEFT, padx=(12, 0))

        prompt_frame = ttk.LabelFrame(root_frame, text="Prompt", padding=10)
        prompt_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        top = ttk.Frame(prompt_frame)
        top.pack(fill=tk.X)
        ttk.Label(top, text="Model").pack(side=tk.LEFT, padx=(0, 8))
        self.model_combo = ttk.Combobox(
            top,
            textvariable=self.model_var,
            state="readonly",
            width=36,
        )
        self.model_combo.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(top, text="Temp").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Entry(top, textvariable=self.temperature_var, width=7).pack(
            side=tk.LEFT,
            padx=(0, 10),
        )
        ttk.Label(top, text="Max Tokens").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Entry(top, textvariable=self.max_tokens_var, width=8).pack(
            side=tk.LEFT,
            padx=(0, 10),
        )
        self.send_btn = ttk.Button(top, text="Send Prompt", command=self._send_prompt)
        self.send_btn.pack(side=tk.LEFT)

        ttk.Label(top, textvariable=self.last_request_var).pack(side=tk.RIGHT)

        self.prompt_box = tk.Text(prompt_frame, height=10, wrap="word")
        self.prompt_box.pack(fill=tk.BOTH, expand=False, pady=(8, 6))
        self.prompt_box.insert(
            "1.0",
            "Explain what system state this telemetry pattern suggests, and what profile action is best.",
        )

        response_frame = ttk.LabelFrame(root_frame, text="Response", padding=10)
        response_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.response_box = tk.Text(response_frame, height=12, wrap="word")
        self.response_box.pack(fill=tk.BOTH, expand=True)

        log_frame = ttk.LabelFrame(root_frame, text="Monitor Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.log_box = tk.Text(log_frame, height=10, wrap="word", state=tk.DISABLED)
        self.log_box.pack(fill=tk.BOTH, expand=True)

    def _schedule_monitor(self):
        self._monitor_tick()
        self._monitor_after_id = self.root.after(MONITOR_INTERVAL_MS, self._schedule_monitor)

    def _monitor_tick(self):
        self._drain_log_queue()
        server_up = self._is_server_up()
        listener_pid = self._find_listener_pid()

        self.status_var.set("running" if server_up else "stopped")
        self.pid_var.set(str(listener_pid) if listener_pid is not None else "-")

        if not PSUTIL_AVAILABLE:
            self.uptime_var.set("-")
            self.cpu_var.set("-")
            self.mem_var.set("-")
            now = time.time()
            if now - self._last_tags_refresh >= 6.0 and server_up:
                self._last_tags_refresh = now
                try:
                    payload = self._request_json("GET", "/api/ps", timeout=2)
                    names = [m.get("name", "?") for m in payload.get("models", [])]
                    self.loaded_var.set(", ".join(names) if names else "(none)")
                except Exception:
                    self.loaded_var.set("unknown")
            if not server_up:
                self.loaded_var.set("-")
            return

        if listener_pid is None:
            self.uptime_var.set("-")
            self.cpu_var.set("-")
            self.mem_var.set("-")
            self.loaded_var.set("-")
            self._process_cache = None
            self._process_cache_pid = None
            return

        if self._process_cache_pid != listener_pid or self._process_cache is None:
            try:
                self._process_cache = psutil.Process(listener_pid)
                self._process_cache.cpu_percent(interval=None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self._process_cache = None
            self._process_cache_pid = listener_pid

        process = self._process_cache
        if process is None:
            self.uptime_var.set("-")
            self.cpu_var.set("-")
            self.mem_var.set("-")
            return

        try:
            uptime_sec = max(0.0, time.time() - process.create_time())
            self.uptime_var.set(self._format_duration(uptime_sec))
            self.cpu_var.set(f"{process.cpu_percent(interval=None):.1f}%")
            self.mem_var.set(f"{process.memory_info().rss / (1024 * 1024):.1f} MB")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            self.uptime_var.set("-")
            self.cpu_var.set("-")
            self.mem_var.set("-")

        # Keep this lightweight: refresh loaded model list at most every 6s.
        now = time.time()
        if now - self._last_tags_refresh >= 6.0 and server_up:
            self._last_tags_refresh = now
            try:
                payload = self._request_json("GET", "/api/ps", timeout=2)
                names = [m.get("name", "?") for m in payload.get("models", [])]
                self.loaded_var.set(", ".join(names) if names else "(none)")
            except Exception:
                self.loaded_var.set("unknown")

    def _on_close(self):
        if self._monitor_after_id is not None:
            self.root.after_cancel(self._monitor_after_id)
            self._monitor_after_id = None
        self.root.destroy()

    def _log(self, message: str):
        stamp = time.strftime("%H:%M:%S")
        self.log_queue.put(f"[{stamp}] {message}")

    def _drain_log_queue(self):
        drained = False
        self.log_box.configure(state=tk.NORMAL)
        while True:
            try:
                line = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self.log_box.insert(tk.END, line + "\n")
            drained = True
        if drained:
            self.log_box.see(tk.END)
        self.log_box.configure(state=tk.DISABLED)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        sec = int(seconds)
        h, rem = divmod(sec, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _find_ollama_binary(self) -> str | None:
        from_path = shutil.which("ollama")
        if from_path:
            return from_path

        candidates = [
            Path.home() / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe",
            Path("C:/Program Files/Ollama/ollama.exe"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return None

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
        timeout: int = 10,
    ) -> dict:
        data = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")

        req = request.Request(
            self.base_url + path,
            data=data,
            headers=headers,
            method=method,
        )

        with request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}

    def _is_server_up(self) -> bool:
        try:
            self._request_json("GET", "/api/tags", timeout=2)
            return True
        except Exception:
            return False

    def _find_listener_pid(self) -> int | None:
        if not PSUTIL_AVAILABLE:
            return None
        try:
            for conn in psutil.net_connections(kind="tcp"):
                if (
                    conn.status == psutil.CONN_LISTEN
                    and conn.laddr
                    and conn.laddr.port == OLLAMA_PORT
                    and conn.pid is not None
                ):
                    return conn.pid
        except (psutil.AccessDenied, psutil.Error):
            return None
        return None

    def _read_process_output(self, proc: subprocess.Popen[str]):
        if proc.stdout is None:
            return
        for raw_line in proc.stdout:
            line = raw_line.strip()
            if line:
                self._log(f"ollama: {line}")

    def _start_ollama(self):
        def worker():
            if self._is_server_up():
                self._log("server already running on 127.0.0.1:11434")
                return

            binary = self._find_ollama_binary()
            if not binary:
                self._log("failed to start: could not find ollama executable")
                self.root.after(
                    0,
                    lambda: messagebox.showerror(
                        "Ollama Not Found",
                        "Could not find 'ollama'. Install Ollama first.",
                    ),
                )
                return

            try:
                flags = 0
                if hasattr(subprocess, "CREATE_NO_WINDOW"):
                    flags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
                proc = subprocess.Popen(
                    [binary, "serve"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    creationflags=flags,
                )
                self.ollama_process = proc
                self._log(f"started ollama serve (pid={proc.pid})")
                threading.Thread(
                    target=self._read_process_output,
                    args=(proc,),
                    daemon=True,
                ).start()
            except Exception as exc:  # noqa: BLE001
                self._log(f"failed to start server: {exc}")
                return

            for _ in range(40):
                if self._is_server_up():
                    self._log("server is responding")
                    self.root.after(0, self._refresh_models_async)
                    return
                time.sleep(0.25)
            self._log("start command sent, but server did not answer in time")

        threading.Thread(target=worker, daemon=True).start()

    def _stop_ollama(self):
        def worker():
            stopped_any = False

            if self.ollama_process is not None and self.ollama_process.poll() is None:
                try:
                    self.ollama_process.terminate()
                    self.ollama_process.wait(timeout=5)
                    self._log(f"stopped launched process pid={self.ollama_process.pid}")
                    stopped_any = True
                except Exception:  # noqa: BLE001
                    try:
                        self.ollama_process.kill()
                        self._log(
                            f"killed launched process pid={self.ollama_process.pid}",
                        )
                        stopped_any = True
                    except Exception as exc:  # noqa: BLE001
                        self._log(f"could not stop launched process: {exc}")

            pid = self._find_listener_pid()
            if pid is not None and PSUTIL_AVAILABLE:
                try:
                    proc = psutil.Process(pid)
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except psutil.TimeoutExpired:
                        proc.kill()
                    self._log(f"stopped listener on port {OLLAMA_PORT} (pid={pid})")
                    stopped_any = True
                except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
                    self._log(f"could not stop listener pid={pid}: {exc}")
            elif self.ollama_process is None and not PSUTIL_AVAILABLE:
                self._log(
                    "psutil not available: can only stop instances launched from this GUI",
                )

            if not stopped_any:
                self._log("no running ollama listener found")

        threading.Thread(target=worker, daemon=True).start()

    def _refresh_models_async(self):
        def worker():
            names: list[str] = []

            if self._is_server_up():
                try:
                    payload = self._request_json("GET", "/api/tags", timeout=5)
                    names = [m.get("name", "") for m in payload.get("models", []) if m.get("name")]
                except Exception as exc:  # noqa: BLE001
                    self._log(f"model refresh via API failed: {exc}")

            if not names:
                binary = self._find_ollama_binary()
                if binary:
                    try:
                        result = subprocess.run(
                            [binary, "list"],
                            capture_output=True,
                            text=True,
                            timeout=30,
                            check=False,
                        )
                        lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
                        for line in lines[1:]:
                            parts = line.split()
                            if parts:
                                names.append(parts[0])
                    except Exception as exc:  # noqa: BLE001
                        self._log(f"model refresh via CLI failed: {exc}")

            deduped = []
            seen = set()
            for name in names:
                if name and name not in seen:
                    seen.add(name)
                    deduped.append(name)

            def apply():
                self.model_combo["values"] = deduped
                if deduped and self.model_var.get() not in deduped:
                    self.model_var.set(deduped[0])
                if deduped:
                    self._log(f"models available: {', '.join(deduped)}")
                else:
                    self._log("no models found yet")

            self.root.after(0, apply)

        threading.Thread(target=worker, daemon=True).start()

    def _send_prompt(self):
        model = self.model_var.get().strip()
        prompt = self.prompt_box.get("1.0", tk.END).strip()
        if not model:
            messagebox.showwarning("Model Required", "Select a model first.")
            return
        if not prompt:
            messagebox.showwarning("Prompt Required", "Enter a prompt first.")
            return

        try:
            temperature = float(self.temperature_var.get().strip())
        except ValueError:
            messagebox.showwarning("Invalid Temperature", "Temperature must be a number.")
            return
        try:
            max_tokens = int(self.max_tokens_var.get().strip())
        except ValueError:
            messagebox.showwarning("Invalid Max Tokens", "Max Tokens must be an integer.")
            return

        self.send_btn.configure(state=tk.DISABLED)
        self.last_request_var.set("running...")
        self._log(f"sending prompt to {model}")

        def worker():
            started = time.time()
            try:
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                }
                response_payload = self._request_json(
                    "POST",
                    "/api/generate",
                    payload=payload,
                    timeout=1800,
                )
                text = response_payload.get("response", "")
                elapsed = time.time() - started
                prompt_tokens = response_payload.get("prompt_eval_count")
                out_tokens = response_payload.get("eval_count")

                def on_ok():
                    self.response_box.delete("1.0", tk.END)
                    self.response_box.insert("1.0", text)
                    token_info = ""
                    if prompt_tokens is not None and out_tokens is not None:
                        token_info = f" | tok in/out: {prompt_tokens}/{out_tokens}"
                    self.last_request_var.set(f"{elapsed:.1f}s{token_info}")
                    self.send_btn.configure(state=tk.NORMAL)
                    self._log(f"request finished in {elapsed:.1f}s")

                self.root.after(0, on_ok)
            except error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
                msg = f"HTTP {exc.code}: {body or exc.reason}"
                self.root.after(
                    0,
                    lambda: self._request_failed(msg),
                )
            except Exception as exc:  # noqa: BLE001
                self.root.after(
                    0,
                    lambda: self._request_failed(str(exc)),
                )

        threading.Thread(target=worker, daemon=True).start()

    def _request_failed(self, message: str):
        self.last_request_var.set("failed")
        self.send_btn.configure(state=tk.NORMAL)
        self._log(f"request failed: {message}")
        messagebox.showerror("Prompt Failed", message)


def main():
    root = tk.Tk()
    app = OllamaControlApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
