"""Tkinter interface for the local Vaani listening pipeline."""
import os
import queue
import tkinter as tk
from datetime import datetime
from tkinter import ttk

from core.gui_backend import VaaniBackend

BACKGROUND = "#10141c"
CARD = "#191f2b"
CARD_ALT = "#202838"
TEXT = "#f2f5f9"
MUTED = "#a4afc0"
ACCENT = "#65d6a6"
RED = "#ff7272"
AMBER = "#f4c46a"
BLUE = "#8ab4ff"
DECISION_COLORS = {
    "FULL_ALERT": RED,
    "SOFT_ALERT": AMBER,
    "IGNORE": MUTED,
}


class VaaniApp:
    def __init__(self, root):
        self.root = root
        self.root.title("VAANI — Intelligent Listening Assistant")
        self.root.configure(bg=BACKGROUND)
        self.root.geometry("1120x800")
        self.root.minsize(900, 680)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

        self.events = queue.Queue()
        self.backend = None
        self.backend_done = True
        self.closing = False
        self.stopping = False
        self.counts = {"FULL_ALERT": 0, "SOFT_ALERT": 0, "IGNORE": 0}
        self.status_var = tk.StringVar(value="Offline")
        self.detail_var = tk.StringVar(value="Enter your name, then start listening.")
        self.name_var = tk.StringVar(value=os.environ.get("VAANI_USER_NAME", ""))

        self._configure_styles()
        self._build_layout()
        self.root.after(100, self._poll_events)

    def _configure_styles(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("App.TFrame", background=BACKGROUND)
        style.configure("Card.TFrame", background=CARD)
        style.configure("CardAlt.TFrame", background=CARD_ALT)
        style.configure(
            "TLabel", background=CARD, foreground=TEXT, font=("Segoe UI", 10)
        )
        style.configure(
            "Title.TLabel",
            background=BACKGROUND,
            foreground=TEXT,
            font=("Segoe UI", 25, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background=BACKGROUND,
            foreground=MUTED,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Section.TLabel",
            background=CARD,
            foreground=TEXT,
            font=("Segoe UI", 13, "bold"),
        )
        style.configure(
            "Muted.TLabel", background=CARD, foreground=MUTED, font=("Segoe UI", 9)
        )
        style.configure(
            "Status.TLabel",
            background=BACKGROUND,
            foreground=ACCENT,
            font=("Segoe UI", 10, "bold"),
        )
        style.configure(
            "Start.TButton",
            background="#2a9d72",
            foreground="#ffffff",
            padding=(18, 11),
            font=("Segoe UI", 10, "bold"),
        )
        style.map(
            "Start.TButton",
            background=[("disabled", "#35443f"), ("active", "#35b985")],
        )
        style.configure(
            "Stop.TButton",
            background="#343c4b",
            foreground=TEXT,
            padding=(18, 11),
            font=("Segoe UI", 10, "bold"),
        )
        style.map("Stop.TButton", background=[("active", "#4a5568")])
        style.configure(
            "TEntry",
            fieldbackground=CARD_ALT,
            foreground=TEXT,
            insertcolor=TEXT,
            padding=7,
        )
        style.configure(
            "Treeview",
            background=CARD_ALT,
            fieldbackground=CARD_ALT,
            foreground=TEXT,
            borderwidth=0,
            rowheight=32,
            font=("Segoe UI", 9),
        )
        style.configure(
            "Treeview.Heading",
            background=CARD,
            foreground=MUTED,
            borderwidth=0,
            font=("Segoe UI", 9, "bold"),
        )
        style.map(
            "Treeview",
            background=[("selected", "#31445a")],
            foreground=[("selected", TEXT)],
        )

    def _build_layout(self):
        outer = ttk.Frame(self.root, style="App.TFrame", padding=(28, 22))
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=4, uniform="main")
        outer.columnconfigure(1, weight=6, uniform="main")
        outer.rowconfigure(1, weight=1)

        header = ttk.Frame(outer, style="App.TFrame")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 20))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="VAANI", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            header,
            text="Your Intelligent Listening Assistant",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(1, 0))

        profile = ttk.Frame(header, style="App.TFrame")
        profile.grid(row=0, column=1, rowspan=2, sticky="e", padx=(18, 0))
        ttk.Label(
            profile, text="CONFIGURED USER", style="Subtitle.TLabel"
        ).grid(row=0, column=0, sticky="w", pady=(0, 4))
        self.name_entry = ttk.Entry(
            profile, textvariable=self.name_var, width=23, font=("Segoe UI", 10)
        )
        self.name_entry.grid(row=1, column=0, sticky="ew")
        status = ttk.Frame(profile, style="App.TFrame")
        status.grid(row=1, column=1, sticky="w", padx=(18, 0))
        self.status_dot = tk.Canvas(
            status, width=12, height=12, bg=BACKGROUND, highlightthickness=0
        )
        self.status_dot.pack(side="left", padx=(0, 7))
        self.status_dot.create_oval(2, 2, 10, 10, fill=MUTED, outline="")
        self.status_label = ttk.Label(
            status, textvariable=self.status_var, style="Status.TLabel"
        )
        self.status_label.pack(side="left")

        left = ttk.Frame(outer, style="App.TFrame")
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 12))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=5)
        left.rowconfigure(1, weight=3)

        mic_card = ttk.Frame(left, style="Card.TFrame", padding=20)
        mic_card.grid(row=0, column=0, sticky="nsew", pady=(0, 12))
        mic_card.columnconfigure(0, weight=1)
        ttk.Label(mic_card, text="Microphone", style="Section.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.mic_canvas = tk.Canvas(
            mic_card, height=180, bg=CARD, highlightthickness=0
        )
        self.mic_canvas.grid(row=1, column=0, sticky="ew", pady=(8, 3))
        self._draw_microphone()
        self.mic_status = ttk.Label(
            mic_card, textvariable=self.detail_var, style="Muted.TLabel",
            wraplength=330, justify="center",
        )
        self.mic_status.grid(row=2, column=0, sticky="ew", pady=(0, 15))
        controls = ttk.Frame(mic_card, style="Card.TFrame")
        controls.grid(row=3, column=0, sticky="ew")
        controls.columnconfigure((0, 1), weight=1)
        self.start_button = ttk.Button(
            controls, text="Start Listening", style="Start.TButton", command=self._start
        )
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.stop_button = ttk.Button(
            controls, text="Stop Listening", style="Stop.TButton",
            command=self._stop, state="disabled",
        )
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=(5, 0))

        stats_card = ttk.Frame(left, style="Card.TFrame", padding=18)
        stats_card.grid(row=1, column=0, sticky="nsew")
        ttk.Label(stats_card, text="Session statistics", style="Section.TLabel").pack(
            anchor="w", pady=(0, 13)
        )
        stats = ttk.Frame(stats_card, style="Card.TFrame")
        stats.pack(fill="both", expand=True)
        for index, (decision, label) in enumerate(
            (("FULL_ALERT", "FULL ALERT"), ("SOFT_ALERT", "SOFT ALERT"), ("IGNORE", "IGNORE"))
        ):
            stats.columnconfigure(index, weight=1)
            tile = ttk.Frame(stats, style="CardAlt.TFrame", padding=(9, 12))
            tile.grid(row=0, column=index, sticky="nsew", padx=(0 if index == 0 else 5, 0))
            ttk.Label(
                tile, text=label, background=CARD_ALT,
                foreground=DECISION_COLORS[decision],
                font=("Segoe UI", 8, "bold"),
            ).pack(anchor="w")
            value = ttk.Label(
                tile, text="0", background=CARD_ALT,
                foreground=TEXT, font=("Segoe UI", 22, "bold"),
            )
            value.pack(anchor="w", pady=(5, 0))
            self.count_labels = getattr(self, "count_labels", {})
            self.count_labels[decision] = value

        right = ttk.Frame(outer, style="App.TFrame")
        right.grid(row=1, column=1, sticky="nsew", padx=(12, 0))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        latest = ttk.Frame(right, style="Card.TFrame", padding=20)
        latest.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        latest.columnconfigure(0, weight=1)
        top = ttk.Frame(latest, style="Card.TFrame")
        top.grid(row=0, column=0, sticky="ew")
        ttk.Label(top, text="Latest detection", style="Section.TLabel").pack(
            side="left"
        )
        self.decision_label = tk.Label(
            top, text="WAITING", bg=CARD_ALT, fg=MUTED,
            font=("Segoe UI", 8, "bold"), padx=11, pady=5,
        )
        self.decision_label.pack(side="right")
        self.transcript_label = ttk.Label(
            latest,
            text="Your latest transcript will appear here.",
            font=("Segoe UI", 13),
            wraplength=560,
            justify="left",
        )
        self.transcript_label.grid(row=1, column=0, sticky="ew", pady=(18, 12))
        self.time_label = ttk.Label(
            latest, text="Processing time: —", style="Muted.TLabel"
        )
        self.time_label.grid(row=2, column=0, sticky="w")

        history = ttk.Frame(right, style="Card.TFrame", padding=20)
        history.grid(row=1, column=0, sticky="nsew")
        history.columnconfigure(0, weight=1)
        history.rowconfigure(1, weight=1)
        ttk.Label(history, text="Activity history", style="Section.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 13)
        )
        table_frame = ttk.Frame(history, style="Card.TFrame")
        table_frame.grid(row=1, column=0, sticky="nsew")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        self.history = ttk.Treeview(
            table_frame,
            columns=("time", "decision", "transcript"),
            show="headings",
            selectmode="browse",
        )
        self.history.heading("time", text="TIME")
        self.history.heading("decision", text="CLASSIFICATION")
        self.history.heading("transcript", text="TRANSCRIPT")
        self.history.column("time", width=90, minwidth=82, stretch=False)
        self.history.column("decision", width=125, minwidth=112, stretch=False)
        self.history.column("transcript", width=280, minwidth=120, stretch=True)
        self.history.tag_configure("FULL_ALERT", foreground=RED)
        self.history.tag_configure("SOFT_ALERT", foreground=AMBER)
        self.history.tag_configure("IGNORE", foreground=MUTED)
        scrollbar = ttk.Scrollbar(
            table_frame, orient="vertical", command=self.history.yview
        )
        self.history.configure(yscrollcommand=scrollbar.set)
        self.history.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

    def _draw_microphone(self):
        canvas = self.mic_canvas
        canvas.create_oval(100, 10, 220, 130, fill=CARD_ALT, outline="#2e394b", width=2)
        canvas.create_oval(108, 18, 212, 122, outline="#344257", width=1)
        canvas.create_oval(145, 38, 175, 82, fill=ACCENT, outline="")
        canvas.create_rectangle(145, 57, 175, 77, fill=ACCENT, outline="")
        canvas.create_arc(
            137, 48, 183, 98, start=180, extent=180, style="arc",
            outline=ACCENT, width=4,
        )
        canvas.create_line(160, 100, 160, 110, fill=ACCENT, width=4)
        canvas.create_line(147, 111, 173, 111, fill=ACCENT, width=4)
        canvas.create_text(
            160, 151, text="READY WHEN YOU ARE", fill=MUTED,
            font=("Segoe UI", 8, "bold"),
        )

    def _set_status(self, status, detail):
        colors = {
            "Offline": MUTED,
            "Starting": BLUE,
            "Listening": ACCENT,
            "Processing": AMBER,
            "Error": RED,
        }
        self.status_var.set(status)
        self.detail_var.set(detail)
        self.status_dot.delete("all")
        self.status_dot.create_oval(2, 2, 10, 10, fill=colors.get(status, MUTED), outline="")

    def _start(self):
        user_name = self.name_var.get().strip()
        if not user_name:
            self._set_status("Error", "Please enter your name before starting.")
            self.name_entry.focus_set()
            return
        self.backend = VaaniBackend(user_name, self._queue_event)
        if not self.backend.start():
            return
        self.backend_done = False
        self.stopping = False
        self.name_entry.configure(state="disabled")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self._set_status("Starting", "Preparing the local AI server and speech models...")

    def _stop(self):
        if self.backend is None or self.backend_done:
            return
        self.stopping = True
        self.stop_button.configure(state="disabled")
        self._set_status("Processing", "Stopping microphone and finishing active processing...")
        self.backend.stop()

    def _queue_event(self, event, payload=None):
        self.events.put((event, payload))

    def _poll_events(self):
        try:
            while True:
                event, payload = self.events.get_nowait()
                if event == "status":
                    status, detail = payload
                    if not (self.stopping and status in ("Listening", "Processing")):
                        self._set_status(status, detail)
                elif event == "result":
                    self._show_result(payload)
                elif event == "error":
                    self.stopping = True
                    self._set_status("Error", str(payload))
                    self.stop_button.configure(state="disabled")
                elif event == "processing_error":
                    self._set_status("Error", f"Speech processing failed: {payload}")
                elif event == "stopped":
                    self.backend_done = True
                    self.stop_button.configure(state="disabled")
                    if self.status_var.get() != "Error":
                        detail = "Listening stopped. Restart is disabled for this session."
                        self._set_status("Offline", detail)
                self.events.task_done()
        except queue.Empty:
            pass

        if self.closing and self.backend_done:
            self.root.destroy()
            return
        self.root.after(100, self._poll_events)

    def _show_result(self, result):
        decision = result.decision
        color = DECISION_COLORS[decision]
        self.transcript_label.configure(text=result.transcript)
        self.decision_label.configure(
            text=decision,
            fg=color,
        )
        self.time_label.configure(
            text=f"Processing time: {result.elapsed_seconds:.2f}s"
        )
        self.counts[decision] += 1
        self.count_labels[decision].configure(text=str(self.counts[decision]))
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.history.insert(
            "", 0,
            values=(timestamp, decision, result.transcript),
            tags=(decision,),
        )
        entries = self.history.get_children()
        for item in entries[100:]:
            self.history.delete(item)

    def _close(self):
        if self.backend is None or self.backend_done:
            self.root.destroy()
            return
        self.closing = True
        self.stopping = True
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="disabled")
        self._set_status("Processing", "Shutting down Vaani safely...")
        self.backend.stop()


def main():
    root = tk.Tk()
    VaaniApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
