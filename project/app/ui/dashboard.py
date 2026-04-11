from __future__ import annotations

import json
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Dict, Optional, Tuple

from app.config import AppConfig
from app.pipeline import Pipeline


def _ui_state_path() -> str:
    # Keep it under data/ so it stays local (gitignored).
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "ui_state.json"))


def _safe_int(s: str, default: int) -> int:
    try:
        return int(str(s).strip())
    except Exception:
        return int(default)


def _apply_modern_theme(root: tk.Tk) -> None:
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    # Palette
    bg = "#0b1220"  # deep navy
    panel = "#0f1b33"  # card surface
    field = "#0b1220"  # inputs
    border = "#22314f"
    text = "#e6eefc"
    muted = "#b2c3da"
    accent = "#3b82f6"  # blue-500
    accent_active = "#2563eb"  # blue-600
    danger = "#ef4444"  # red-500

    root.configure(bg=bg)
    style.configure(".", background=bg, foreground=text, fieldbackground=field, bordercolor=border)
    style.configure("TFrame", background=bg)
    style.configure("Card.TFrame", background=panel)
    style.configure("TLabel", background=bg, foreground=text)
    style.configure("Muted.TLabel", background=bg, foreground=muted)
    style.configure("Card.TLabel", background=panel, foreground=text)
    style.configure("CardMuted.TLabel", background=panel, foreground=muted)

    style.configure(
        "Dark.TEntry",
        padding=8,
        relief="flat",
        borderwidth=1,
        foreground=text,
        fieldbackground=field,
        insertcolor=text,
    )
    style.map("Dark.TEntry", fieldbackground=[("disabled", field)])

    style.configure(
        "Dark.TCombobox",
        padding=6,
        relief="flat",
        foreground=text,
        fieldbackground=field,
        arrowsize=14,
    )
    # Readonly combobox defaults can be low-contrast on Windows; force colors.
    style.map(
        "Dark.TCombobox",
        fieldbackground=[("readonly", field), ("disabled", field)],
        foreground=[("readonly", text), ("disabled", muted)],
        selectforeground=[("readonly", text)],
        selectbackground=[("readonly", field)],
    )
    # Dropdown listbox colors
    try:
        root.option_add("*TCombobox*Listbox.background", field)
        root.option_add("*TCombobox*Listbox.foreground", text)
        root.option_add("*TCombobox*Listbox.selectBackground", accent_active)
        root.option_add("*TCombobox*Listbox.selectForeground", "white")
    except Exception:
        pass

    style.configure(
        "Primary.TButton",
        padding=(14, 10),
        relief="flat",
        background=accent,
        foreground="white",
        borderwidth=0,
        focusthickness=0,
    )
    style.map(
        "Primary.TButton",
        background=[("active", accent_active), ("pressed", accent_active), ("disabled", "#274169")],
        foreground=[("disabled", "#cbd5e1")],
    )

    style.configure(
        "Danger.TButton",
        padding=(14, 10),
        relief="flat",
        background=danger,
        foreground="white",
        borderwidth=0,
        focusthickness=0,
    )
    style.map(
        "Danger.TButton",
        background=[("active", "#dc2626"), ("pressed", "#dc2626"), ("disabled", "#5b1a1a")],
        foreground=[("disabled", "#cbd5e1")],
    )

    style.configure(
        "Secondary.TButton",
        padding=(12, 9),
        relief="flat",
        background="#1f2a44",
        foreground=text,
        borderwidth=0,
        focusthickness=0,
    )
    style.map("Secondary.TButton", background=[("active", "#243252"), ("pressed", "#243252")])

    style.configure("TLabelframe", background=bg, foreground=text)
    style.configure("TLabelframe.Label", background=bg, foreground=muted)
    style.configure("TSeparator", background=border)


class Dashboard(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        _apply_modern_theme(self)
        self.title("Prison Video Analytics (MVP)")
        self.geometry("720x460")
        self.minsize(720, 460)

        self._runner_thread: Optional[threading.Thread] = None
        self._state_path = _ui_state_path()
        self._last_source_cfg: Optional[Dict] = None

        self.mode_var = tk.StringVar(value="FIGHT")
        self.source_type_var = tk.StringVar(value="file")
        self.file_path_var = tk.StringVar(value="")
        self.webcam_index_var = tk.StringVar(value="0")
        self.rtsp_url_var = tk.StringVar(value="rtsp://")

        self.status_var = tk.StringVar(value="Hazır. Bir kaynak seçip analizi başlat.")
        self.hint_var = tk.StringVar(value="")

        self._load_state()
        self._build()
        self._refresh_source_fields()

    def _build(self) -> None:
        pad = 18

        root = ttk.Frame(self, padding=pad)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=1)

        # Header
        header = ttk.Frame(root)
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(header, text="Prison Video Analytics", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ttk.Label(
            header,
            text="Fight + Monitoring | Erken uyarı MVP (heuristic + opsiyonel model)",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        # Left card: mode/source
        left = ttk.Frame(root, style="Card.TFrame", padding=16)
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 10), pady=(14, 0))
        left.columnconfigure(0, weight=1)

        ttk.Label(left, text="Ayarlar", style="Card.TLabel", font=("Segoe UI", 12, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(left, text="Mod ve kaynak seçimi", style="CardMuted.TLabel").grid(row=1, column=0, sticky="w", pady=(2, 10))

        row = 2
        ttk.Label(left, text="Mod", style="CardMuted.TLabel").grid(row=row, column=0, sticky="w")
        row += 1
        ttk.Combobox(
            left,
            textvariable=self.mode_var,
            values=["FIGHT", "MONITORING"],
            state="readonly",
            width=18,
            style="Dark.TCombobox",
        ).grid(row=row, column=0, sticky="ew", pady=(4, 12))
        row += 1

        ttk.Label(left, text="Kaynak", style="CardMuted.TLabel").grid(row=row, column=0, sticky="w")
        row += 1
        ttk.Combobox(
            left,
            textvariable=self.source_type_var,
            values=["file", "webcam", "rtsp"],
            state="readonly",
            width=18,
            style="Dark.TCombobox",
        ).grid(row=row, column=0, sticky="ew", pady=(4, 10))
        self.source_type_var.trace_add("write", lambda *_: self._refresh_source_fields())
        row += 1

        self.file_frame = ttk.Frame(left, style="Card.TFrame")
        self.file_frame.grid(row=row, column=0, sticky="ew")
        self.file_frame.columnconfigure(0, weight=1)
        ttk.Label(self.file_frame, text="Video dosyası", style="CardMuted.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 6))
        ttk.Entry(self.file_frame, textvariable=self.file_path_var, style="Dark.TEntry").grid(row=1, column=0, sticky="ew")
        ttk.Button(self.file_frame, text="Seç", style="Secondary.TButton", command=self._pick_file).grid(
            row=1, column=1, padx=(10, 0), sticky="e"
        )

        self.webcam_frame = ttk.Frame(left, style="Card.TFrame")
        self.webcam_frame.grid(row=row, column=0, sticky="ew")
        self.webcam_frame.columnconfigure(0, weight=1)
        ttk.Label(self.webcam_frame, text="Kamera index", style="CardMuted.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 6))
        ttk.Entry(self.webcam_frame, textvariable=self.webcam_index_var, style="Dark.TEntry").grid(row=1, column=0, sticky="ew")
        ttk.Label(self.webcam_frame, text="Örn: 0", style="CardMuted.TLabel").grid(row=1, column=1, padx=(10, 0), sticky="w")

        self.rtsp_frame = ttk.Frame(left, style="Card.TFrame")
        self.rtsp_frame.grid(row=row, column=0, sticky="ew")
        self.rtsp_frame.columnconfigure(0, weight=1)
        ttk.Label(self.rtsp_frame, text="RTSP URL", style="CardMuted.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 6))
        ttk.Entry(self.rtsp_frame, textvariable=self.rtsp_url_var, style="Dark.TEntry").grid(row=1, column=0, sticky="ew")
        row += 1

        # Right card: actions/status
        right = ttk.Frame(root, style="Card.TFrame", padding=16)
        right.grid(row=1, column=1, sticky="nsew", padx=(10, 0), pady=(14, 0))
        right.columnconfigure(0, weight=1)

        ttk.Label(right, text="Çalıştır", style="Card.TLabel", font=("Segoe UI", 12, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(right, text="Kayıtlar ve kontroller", style="CardMuted.TLabel").grid(row=1, column=0, sticky="w", pady=(2, 10))

        btn_row = ttk.Frame(right, style="Card.TFrame")
        btn_row.grid(row=2, column=0, sticky="ew")
        ttk.Button(btn_row, text="Analizi Başlat", style="Primary.TButton", command=self._start).pack(side="left")
        ttk.Button(btn_row, text="Çıkış", style="Secondary.TButton", command=self.destroy).pack(side="right")

        ttk.Separator(right).grid(row=3, column=0, sticky="ew", pady=14)

        ttk.Label(right, text="Durum", style="CardMuted.TLabel").grid(row=4, column=0, sticky="w")
        ttk.Label(right, textvariable=self.status_var, style="Card.TLabel", wraplength=300, justify="left").grid(
            row=5, column=0, sticky="ew", pady=(6, 0)
        )
        ttk.Label(right, textvariable=self.hint_var, style="CardMuted.TLabel", wraplength=300, justify="left").grid(
            row=6, column=0, sticky="ew", pady=(6, 0)
        )

        ttk.Separator(right).grid(row=7, column=0, sticky="ew", pady=14)

        ttk.Label(right, text="Kısayollar (analiz penceresi)", style="CardMuted.TLabel").grid(row=8, column=0, sticky="w")
        ttk.Label(
            right,
            text=(
                "Space: Duraklat | r: Başa sar | +/-: Zoom | WASD: Kaydır | 0: Sıfırla | q/ESC: Çık\n"
                "UYARI/ALARM: ham kare data/snapshots altına kaydedilir (opsiyonel önizleme penceresi)"
            ),
            style="CardMuted.TLabel",
            wraplength=300,
            justify="left",
        ).grid(row=9, column=0, sticky="ew", pady=(6, 0))

    def _refresh_source_fields(self) -> None:
        t = self.source_type_var.get()
        # show/hide frames
        self.file_frame.grid_remove()
        self.webcam_frame.grid_remove()
        self.rtsp_frame.grid_remove()

        if t == "file":
            self.file_frame.grid()
            self.hint_var.set("İpucu: Büyük dosyalarda ilk açılış birkaç saniye sürebilir.")
        elif t == "webcam":
            self.webcam_frame.grid()
            self.hint_var.set("İpucu: Kamera index 0/1/2... şeklinde değişebilir.")
        else:
            self.rtsp_frame.grid()
            self.hint_var.set("İpucu: RTSP bağlantısı için ağ/kimlik bilgileri gerekebilir.")

    def _pick_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Video seç",
            filetypes=[("Video", "*.mp4 *.avi *.mkv *.mov"), ("Tümü", "*.*")],
        )
        if path:
            # Use absolute path on Windows to avoid relative + unicode path issues
            self.file_path_var.set(os.path.abspath(path))

    def _build_source_cfg(self) -> Dict:
        t = self.source_type_var.get()
        if t == "file":
            p = self.file_path_var.get().strip()
            if not p:
                raise ValueError("Video dosyası seçilmedi.")
            return {"type": "file", "path": p}
        if t == "webcam":
            idx = _safe_int(self.webcam_index_var.get(), 0)
            return {"type": "webcam", "index": idx}
        url = self.rtsp_url_var.get().strip()
        if not url or url == "rtsp://":
            raise ValueError("RTSP URL boş.")
        return {"type": "rtsp", "url": url}

    def _load_state(self) -> None:
        try:
            if not os.path.exists(self._state_path):
                return
            with open(self._state_path, "r", encoding="utf-8") as f:
                st = json.load(f)
            mode = str(st.get("mode") or "").strip().upper()
            if mode in ("FIGHT", "MONITORING"):
                self.mode_var.set(mode)
            t = str(st.get("source_type") or "").strip().lower()
            if t in ("file", "webcam", "rtsp"):
                self.source_type_var.set(t)
            self.file_path_var.set(str(st.get("file_path") or "").strip())
            self.webcam_index_var.set(str(st.get("webcam_index") or "0").strip())
            self.rtsp_url_var.set(str(st.get("rtsp_url") or "rtsp://").strip())
        except Exception:
            # ignore corrupted state file
            pass

    def _save_state(self, source_cfg: Dict) -> None:
        try:
            os.makedirs(os.path.dirname(self._state_path), exist_ok=True)
            st = {
                "mode": self.mode_var.get().strip().upper(),
                "source_type": self.source_type_var.get().strip().lower(),
                "file_path": self.file_path_var.get().strip(),
                "webcam_index": self.webcam_index_var.get().strip(),
                "rtsp_url": self.rtsp_url_var.get().strip(),
                "last_source_cfg": source_cfg,
            }
            with open(self._state_path, "w", encoding="utf-8") as f:
                json.dump(st, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _start(self) -> None:
        if self._runner_thread and self._runner_thread.is_alive():
            messagebox.showinfo("Bilgi", "Analiz zaten çalışıyor. Analiz penceresinden q/ESC ile kapatabilirsin.")
            return

        try:
            source_cfg = self._build_source_cfg()
        except Exception as e:
            messagebox.showerror("Hata", str(e))
            return

        # AppConfig is frozen; set via constructor / object.__setattr__
        cfg = AppConfig(mode=self.mode_var.get().strip().upper())
        object.__setattr__(cfg, "VIDEO_SOURCE", source_cfg)

        self._last_source_cfg = source_cfg
        self._save_state(source_cfg)
        self.status_var.set(f"Analiz başlatıldı. Kaynak: {source_cfg.get('type')}.")
        self.hint_var.set("Analiz penceresinden q/ESC ile çıkabilirsin.")

        def run() -> None:
            import sys
            import traceback

            try:
                Pipeline(cfg).run()
            except Exception as e:
                tb = traceback.format_exc()
                # Ensure we see the real reason in terminal too
                print(tb, file=sys.stderr)

                msg = str(e).strip()
                if not msg or msg.lower() == "none":
                    msg = repr(e)
                full = f"{msg}\n\n--- TRACEBACK ---\n{tb}"
                self.after(0, lambda: messagebox.showerror("Çalışma Hatası", full))
            finally:
                self.after(0, lambda: self.status_var.set("Analiz bitti. Yeni kaynak seçip tekrar başlatabilirsin."))
                self.after(0, lambda: self.hint_var.set(""))

        self._runner_thread = threading.Thread(target=run, daemon=True)
        self._runner_thread.start()


def launch_dashboard() -> None:
    Dashboard().mainloop()

