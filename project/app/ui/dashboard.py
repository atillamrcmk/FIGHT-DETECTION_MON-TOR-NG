from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Dict, Optional

from app.config import AppConfig
from app.pipeline import Pipeline


class Dashboard(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Video Analiz Paneli (MVP)")
        self.geometry("560x330")
        self.resizable(False, False)

        self._runner_thread: Optional[threading.Thread] = None

        self.mode_var = tk.StringVar(value="FIGHT")
        self.source_type_var = tk.StringVar(value="file")
        self.file_path_var = tk.StringVar(value="")
        self.webcam_index_var = tk.StringVar(value="0")
        self.rtsp_url_var = tk.StringVar(value="rtsp://")

        self.status_var = tk.StringVar(value="Hazır. Video seçip başlatabilirsin.")

        self._build()
        self._refresh_source_fields()

    def _build(self) -> None:
        pad = 10

        top = ttk.Frame(self, padding=pad)
        top.pack(fill="x")

        ttk.Label(top, text="Mod").grid(row=0, column=0, sticky="w")
        ttk.Combobox(top, textvariable=self.mode_var, values=["FIGHT", "MONITORING"], state="readonly", width=16).grid(
            row=0, column=1, sticky="w", padx=(8, 0)
        )

        ttk.Label(top, text="Kaynak").grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Combobox(
            top, textvariable=self.source_type_var, values=["file", "webcam", "rtsp"], state="readonly", width=16
        ).grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(10, 0))
        self.source_type_var.trace_add("write", lambda *_: self._refresh_source_fields())

        self.file_frame = ttk.LabelFrame(self, text="Video Dosyası", padding=pad)
        self.file_frame.pack(fill="x", padx=pad, pady=(0, pad))
        ttk.Entry(self.file_frame, textvariable=self.file_path_var, width=56).grid(row=0, column=0, sticky="w")
        ttk.Button(self.file_frame, text="Seç...", command=self._pick_file).grid(row=0, column=1, padx=(8, 0))

        self.webcam_frame = ttk.LabelFrame(self, text="Canlı Kamera", padding=pad)
        self.webcam_frame.pack(fill="x", padx=pad, pady=(0, pad))
        ttk.Label(self.webcam_frame, text="Kamera index (0/1/2...)").grid(row=0, column=0, sticky="w")
        ttk.Entry(self.webcam_frame, textvariable=self.webcam_index_var, width=10).grid(row=0, column=1, sticky="w", padx=(8, 0))

        self.rtsp_frame = ttk.LabelFrame(self, text="RTSP", padding=pad)
        self.rtsp_frame.pack(fill="x", padx=pad, pady=(0, pad))
        ttk.Entry(self.rtsp_frame, textvariable=self.rtsp_url_var, width=56).grid(row=0, column=0, sticky="w")

        btns = ttk.Frame(self, padding=pad)
        btns.pack(fill="x")
        ttk.Button(btns, text="Analizi Başlat", command=self._start).pack(side="left")
        ttk.Button(btns, text="Çıkış", command=self.destroy).pack(side="right")

        status = ttk.Label(self, textvariable=self.status_var, padding=pad)
        status.pack(fill="x")

        helpbox = ttk.Label(
            self,
            text=(
                "Kontroller (analiz penceresi): Space=Duraklat, r=Başa sar, +/-=Zoom, WASD=Kaydır, 0=Sıfırla, q/ESC=Çık"
            ),
            padding=(pad, 0, pad, pad),
            foreground="#444",
        )
        helpbox.pack(fill="x")

    def _refresh_source_fields(self) -> None:
        t = self.source_type_var.get()
        # show/hide frames
        self.file_frame.pack_forget()
        self.webcam_frame.pack_forget()
        self.rtsp_frame.pack_forget()

        pad = 10
        if t == "file":
            self.file_frame.pack(fill="x", padx=pad, pady=(0, pad))
        elif t == "webcam":
            self.webcam_frame.pack(fill="x", padx=pad, pady=(0, pad))
        else:
            self.rtsp_frame.pack(fill="x", padx=pad, pady=(0, pad))

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
            idx = int(self.webcam_index_var.get().strip() or "0")
            return {"type": "webcam", "index": idx}
        url = self.rtsp_url_var.get().strip()
        if not url or url == "rtsp://":
            raise ValueError("RTSP URL boş.")
        return {"type": "rtsp", "url": url}

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

        self.status_var.set(f"Analiz başlatıldı. Kaynak: {source_cfg.get('type')}.")

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

        self._runner_thread = threading.Thread(target=run, daemon=True)
        self._runner_thread.start()


def launch_dashboard() -> None:
    Dashboard().mainloop()

