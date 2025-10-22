#!/usr/bin/env python3
"""
PyQt5-based multi-agent, layered summarization & refinement GUI for Ollama LLM,
with a dark-themed, two-pane layout, a pop-up “All System Prompts” dialog, and a
pop-up log window (500×700) that auto-scrolls by default.

Model: llama3.2:3b
"""

import sys
import os
import json
import threading
import time
import concurrent.futures
import re
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPlainTextEdit, QTextEdit,
    QVBoxLayout, QHBoxLayout, QPushButton, QSpinBox, QGroupBox,
    QFormLayout, QProgressBar, QFrame, QSplitter,
    QScrollArea, QLineEdit, QDialog, QDialogButtonBox, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPalette, QColor
from ollama import Client

# Import JimDialog so we can launch the JIM agent window
from Julia_Jim_Flow_1 import JimDialog

# ────────────────────────────────────────────────────────────────
# Configuration & Defaults
# ────────────────────────────────────────────────────────────────
OLLAMA_HOST = 'http://localhost:11434'
MODEL_NAME = 'llama3.2:3b'

USER_INPUTS_DIR = "user_inputs"
USER_OUTPUTS_DIR = "user_outputs"
LOGS_DIR = "julia_logs"
SYSTEM_PROMPTS_FILE = "system_prompts.json"

os.makedirs(USER_INPUTS_DIR, exist_ok=True)
os.makedirs(USER_OUTPUTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Default prompts to write out on first launch
DEFAULT_PROMPTS = {
    "subject_extraction_prompt": (
        "From the following text excerpt, extract the single most important subject or theme "
        "as a concise phrase."
    ),
    "concept_extraction_prompt": (
        "From the following summary, extract its core concept or key insight as a concise phrase."
    ),
    "layer_system_prompt": (
        "You are an agent summarizer. You are processing a single chunk of a document.\n"
        "Add keywords and core concepts. Be aware you are one of multiple agents.\n"
        "Output must be concise (under 1000 characters)."
    ),
    "batch_summary_prompt": (
        "You are summarizing the combined outputs of multiple agents. Keep it concise."
    ),
    "final_system_prompt": (
        "You are the final summarizer. Without referencing how you arrived at your answer or "
        "mentioning any summaries or intermediate steps, provide the best possible final response "
        "to the user's original input."
    )
}

# Load or create the JSON file that stores all system prompts,
# and merge any new defaults
if not os.path.exists(SYSTEM_PROMPTS_FILE):
    with open(SYSTEM_PROMPTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(DEFAULT_PROMPTS, f, indent=2)
with open(SYSTEM_PROMPTS_FILE, 'r', encoding='utf-8') as f:
    SYSTEM_PROMPTS = json.load(f)
for k, v in DEFAULT_PROMPTS.items():
    if k not in SYSTEM_PROMPTS:
        SYSTEM_PROMPTS[k] = v
with open(SYSTEM_PROMPTS_FILE, 'w', encoding='utf-8') as f:
    json.dump(SYSTEM_PROMPTS, f, indent=2)


# ────────────────────────────────────────────────────────────────
# Helper Function to Validate LLM Responses
# ────────────────────────────────────────────────────────────────
def is_valid_response(text: str) -> bool:
    """Checks if a response from the LLM is a valid summary or a refusal."""
    if not text or not text.strip():
        return False

    lower_text = text.lower()

    # Check for common refusal phrases
    refusal_phrases = [
        "i can't", "i cannot", "unable to", "i'm not able", "i am not able",
        "ready to assist", "ready to help", "ready to summarize",
        "provide the text", "provide the chunk", "what is the chunk",
        "you haven't provided", "please provide", "as an ai", "i am an ai",
        "misunderstanding in my previous response"
    ]
    if any(phrase in lower_text for phrase in refusal_phrases):
        return False

    # Check for responses that are just short, unhelpful questions
    if len(text.split()) < 5 and "?" in text:
        return False

    return True


# ────────────────────────────────────────────────────────────────
# Separate Log Window (500×700)
# ────────────────────────────────────────────────────────────────
class LogWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Processing Log")
        self.setFixedSize(500, 700)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        pause_layout = QHBoxLayout()
        self.pause_button = QPushButton("Pause Auto‐Scroll")
        self.pause_button.clicked.connect(self.toggle_autoscroll)
        pause_layout.addWidget(self.pause_button, alignment=Qt.AlignLeft)
        pause_layout.addStretch()
        layout.addLayout(pause_layout)
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet(
            "background-color: #2d2d44; color: #f0f0f0; border: 1px solid #3a3a5e;"
        )
        layout.addWidget(self.log_output)
        self.auto_scroll = True

    def append_log(self, text: str):
        self.log_output.appendPlainText(text)
        if self.auto_scroll:
            cursor = self.log_output.textCursor()
            cursor.movePosition(cursor.End)
            self.log_output.setTextCursor(cursor)
            self.log_output.ensureCursorVisible()

    def toggle_autoscroll(self):
        self.auto_scroll = not self.auto_scroll
        self.pause_button.setText(
            "Pause Auto‐Scroll" if self.auto_scroll else "Resume Auto‐Scroll"
        )
        if self.auto_scroll:
            cursor = self.log_output.textCursor()
            cursor.movePosition(cursor.End)
            self.log_output.setTextCursor(cursor)
            self.log_output.ensureCursorVisible()


# ────────────────────────────────────────────────────────────────
# System Prompts Management Dialog
# ────────────────────────────────────────────────────────────────
class PromptsWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("All System Prompts")
        self.resize(600, 500)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        self.inner_layout = QVBoxLayout(inner)
        self.inner_layout.setSpacing(12)
        self.load_prompts()
        scroll.setWidget(inner)
        layout.addWidget(scroll)

        add_button = QPushButton("Add New Prompt")
        add_button.setStyleSheet("background-color: #30475e; color: #f0f0f0;")
        add_button.clicked.connect(self.add_new_prompt)
        layout.addWidget(add_button, alignment=Qt.AlignRight)

        btn_box = QDialogButtonBox(QDialogButtonBox.Close)
        btn_box.rejected.connect(self.close)
        layout.addWidget(btn_box)

    def load_prompts(self):
        for i in reversed(range(self.inner_layout.count())):
            item = self.inner_layout.itemAt(i)
            if item and item.widget():
                item.widget().deleteLater()

        for key, content in SYSTEM_PROMPTS.items():
            group = QGroupBox(key.replace("_", " ").title())
            gl = QVBoxLayout()
            editor = QPlainTextEdit()
            editor.setPlainText(content)
            editor.setStyleSheet(
                "background-color: #2d2d44; color: #f0f0f0; border: 1px solid #3a3a5e;"
            )
            upd = QPushButton(f"Update '{key}'")
            upd.setStyleSheet("background-color: #30475e; color: #f0f0f0;")

            def make_up(k, ed):
                def fn():
                    SYSTEM_PROMPTS[k] = ed.toPlainText()
                    with open(SYSTEM_PROMPTS_FILE, 'w', encoding='utf-8') as f:
                        json.dump(SYSTEM_PROMPTS, f, indent=2)
                    QMessageBox.information(self, "Prompt Updated", f"'{k}' has been updated.")

                return fn

            upd.clicked.connect(make_up(key, editor))
            gl.addWidget(editor)
            gl.addWidget(upd, alignment=Qt.AlignRight)
            group.setLayout(gl)
            self.inner_layout.addWidget(group)

    def add_new_prompt(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Add New System Prompt")
        dl = QVBoxLayout(dialog)
        title = QLineEdit()
        title.setPlaceholderText("Enter prompt key (e.g. 'my_custom_prompt')")
        content = QPlainTextEdit()
        content.setPlaceholderText("Enter prompt content here…")
        dl.addWidget(QLabel("Prompt Key:"))
        dl.addWidget(title)
        dl.addWidget(QLabel("Prompt Content:"))
        dl.addWidget(content)

        bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        bb.accepted.connect(dialog.accept)
        bb.rejected.connect(dialog.reject)
        dl.addWidget(bb)

        if dialog.exec_() == QDialog.Accepted:
            k, c = title.text().strip(), content.toPlainText().strip()
            if not k or not c:
                QMessageBox.warning(self, "Invalid Input", "Both key and content must be non-empty.")
                return
            if k in SYSTEM_PROMPTS:
                QMessageBox.warning(self, "Duplicate Key", f"'{k}' already exists.")
                return
            SYSTEM_PROMPTS[k] = c
            with open(SYSTEM_PROMPTS_FILE, 'w', encoding='utf-8') as f:
                json.dump(SYSTEM_PROMPTS, f, indent=2)
            self.load_prompts()


# ────────────────────────────────────────────────────────────────
# Worker Thread: Multi-agent pipeline with subject/concept extraction
# ────────────────────────────────────────────────────────────────
class PipelineWorker(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    scores_signal = pyqtSignal(list)
    timing_signal = pyqtSignal(str, float)
    total_time_signal = pyqtSignal(float)

    def __init__(self, user_input, concurrency, num_layers, prompts, parent=None):
        super().__init__(parent)
        self.user_input = user_input
        self.concurrency = concurrency
        self.num_layers = num_layers
        self.prompts = prompts.copy()
        self.client = Client(host=OLLAMA_HOST)
        self.lock = threading.Lock()
        self.layer_texts = {}
        self.subject = ""
        self.total_steps = 2 * (self.num_layers + 1)
        self.current_step = 0

    def extract_subject(self):
        excerpt = self.user_input[:2000]
        resp = self.client.chat(
            model=MODEL_NAME,
            messages=[{
                'role': 'user',
                'content': f"{self.prompts['subject_extraction_prompt']}\n\n{excerpt}"
            }]
        )
        sub = resp.get('message', {}).get('content', '').strip()
        self.subject = sub
        self.log_signal.emit(f"🗂 Core Subject: {self.subject}\n")

    def extract_concept(self, text, label):
        resp = self.client.chat(
            model=MODEL_NAME,
            messages=[{
                'role': 'user',
                'content': f"{self.prompts['concept_extraction_prompt']}\n\n{text}"
            }]
        )
        concept = resp.get('message', {}).get('content', '').strip()
        self.log_signal.emit(f"💡 {label} Core Concept: {concept}\n")
        return concept

    def split_input(self, text: str):
        lines, chunks, buffer = text.splitlines(keepends=True), [], []

        def flush():
            if buffer:
                prose = ''.join(buffer).strip();
                buffer.clear()
                # Split by sentence, but also handle list-like items without periods
                sentences = re.split(r'(?<=[.?!])\s+', prose)
                for s in sentences:
                    if s.strip():
                        chunks.append(s.strip())

        i = 0
        while i < len(lines):
            ln = lines[i];
            st = ln.lstrip()
            if st.startswith(("import ", "from ")):
                flush()
                cb = []
                while i < len(lines):
                    c = lines[i]
                    if c.strip() == "" or c.startswith((" ", "\t")) or c.lstrip().startswith(
                            ("import ", "from ", "class ", "def ", "#")):
                        cb.append(c);
                        i += 1
                    else:
                        break
                chunks.append("".join(cb).rstrip())
            else:
                buffer.append(ln);
                i += 1
        flush()
        return [c for c in chunks if c]

    def agent_call(self, chunk: str, system_prompt: str):
        subj_prefix = f"Core Subject: {self.subject}\n\n" if self.subject else ""
        prompt = f"{system_prompt}\n\n{subj_prefix}User Chunk:\n{chunk}"
        with self.lock:
            r = self.client.chat(model=MODEL_NAME, messages=[{'role': 'user', 'content': prompt}])
        return r.get('message', {}).get('content', '').strip()

    def process_layer(self, chunks, concurrency, system_prompt, layer_name):
        results = []

        def task(ch):
            # *** FIX STARTS HERE ***
            start_time = time.time()
            rsp = self.agent_call(ch, system_prompt)
            duration = time.time() - start_time
            # *** FIX ENDS HERE ***

            if not is_valid_response(rsp):
                self.log_signal.emit(f"🔹 {layer_name} – ⚠️ REFUSED or invalid response, discarding.\n")
                return None

            # Use the calculated duration instead of the absolute time
            self.log_signal.emit(f"🔹 {layer_name} – chunk (⏱ {duration:.1f}s)\n{rsp}\n")
            return rsp

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
            valid_chunks = [c for c in chunks if c and c.strip()]
            futures = {ex.submit(task, c): c for c in valid_chunks}
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())
        return results

    def update_progress(self):
        self.current_step += 1
        pct = int(self.current_step / self.total_steps * 100)
        self.progress_signal.emit(pct)

    def layered_summary(self, text, concurrency, num_layers, final_prompt, pass_name):
        outputs = self.split_input(text)
        self.log_signal.emit(f"   • {pass_name}: starting with {len(outputs)} chunks.")
        for i in range(num_layers):
            name = f"{pass_name} – Layer {i + 1}"
            t0 = time.time()
            self.log_signal.emit(f"\n🔸 {name}: {len(outputs)} chunks…")
            new_outputs = []
            for b in range(0, len(outputs), concurrency):
                batch = outputs[b:b + concurrency]
                self.log_signal.emit(f"   ⏳ {name} batch {b // concurrency + 1}…")

                res = self.process_layer(batch, concurrency, self.prompts["layer_system_prompt"], name)
                valid_res = [r for r in res if r]

                if not valid_res:
                    self.log_signal.emit(f"   🟡 {name} batch summary: SKIPPED (no valid responses).\n")
                    continue

                if len(valid_res) > 1:
                    cmb = "\n".join(valid_res)
                    summ = self.agent_call(cmb, self.prompts["batch_summary_prompt"])

                    if is_valid_response(summ):
                        self.log_signal.emit(f"   🟡 {name} batch summary:\n{summ}\n")
                        self.extract_concept(summ, f"{name} Batch")
                        new_outputs.append(summ)
                    else:
                        self.log_signal.emit(f"   🟡 {name} ⚠️ batch summary REFUSED, discarding.\n")
                else:
                    new_outputs.extend(valid_res)

            layer_duration = time.time() - t0
            self.layer_texts[name] = "\n".join(new_outputs)
            self.timing_signal.emit(name, layer_duration)
            self.log_signal.emit(f"✅ {name} done in {layer_duration:.1f}s; {len(new_outputs)} remain.\n")

            if not new_outputs:
                self.log_signal.emit(f"🛑 Terminating pass '{pass_name}' early: Layer {i + 1} produced no valid output.")
                break

            outputs = new_outputs
            self.update_progress()

        cname = f"{pass_name} – Consolidation"
        t1 = time.time()
        agg = "\n".join(outputs)

        if not agg.strip():
            self.log_signal.emit(f"🔷 {cname}: SKIPPED (no aggregated content to consolidate).")
            return "Processing failed to produce a consolidated summary."

        self.log_signal.emit(f"🔷 {cname}: consolidating…")
        final = self.agent_call(f"ORIGINAL INPUT:\n{text}\n\nAGGREGATED SUMMARIES:\n{agg}", final_prompt)
        consolidation_duration = time.time() - t1
        self.timing_signal.emit(cname, consolidation_duration)
        self.log_signal.emit(f"✅ {cname} done in {consolidation_duration:.1f}s.\n")
        self.extract_concept(final, cname)
        self.layer_texts[cname] = final
        self.update_progress()
        return final

    def score_text(self, text: str, label: str):
        prompt = (
                "Rate the complexity of the following text on a scale from 1 (very simple) "
                "to 10 (very complex). Only respond with an integer (1–10).\n\nText:\n" + text
        )
        with self.lock:
            r = self.client.chat(model=MODEL_NAME, messages=[{'role': 'user', 'content': prompt}])
        try:
            s = int(r.get('message', {}).get('content', '').strip())
            s = max(1, min(10, s))
        except:
            s = 5
        self.log_signal.emit(f"🏷 Complexity {label}: {s}/10\n")
        return s

    def run(self):
        start = time.time()
        self.extract_subject()
        self.log_signal.emit("🟢 First Pass begins…")
        interm = self.layered_summary(
            self.user_input,
            self.concurrency,
            self.num_layers,
            self.prompts["final_system_prompt"],
            "First Pass"
        )
        self.log_signal.emit(f"\n🔵 Intermediate result:\n{interm}\n")

        if not is_valid_response(interm):
            self.log_signal.emit("🛑 Refinement Pass skipped due to invalid intermediate result.\n")
            self.finished_signal.emit(interm)
            return

        self.log_signal.emit("🔄 Refinement Pass begins…\n")
        refined = self.layered_summary(
            interm,
            self.concurrency,
            self.num_layers,
            self.prompts["final_system_prompt"],
            "Refinement Pass"
        )
        self.log_signal.emit("✅ Refinement Pass complete.\n")
        self.total_time_signal.emit(time.time() - start)

        scores = []
        for lbl, txt in self.layer_texts.items():
            if txt and txt.strip():
                scores.append((lbl, self.score_text(txt, lbl)))
        if is_valid_response(refined):
            scores.append(("Final Output", self.score_text(refined, "Final Output")))
        self.scores_signal.emit(scores)

        self.finished_signal.emit(refined)


# ────────────────────────────────────────────────────────────────
# Main Window: Dark-themed, Two-Pane Layout + Next agent Jim button
# ────────────────────────────────────────────────────────────────
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Agentic LLM Pipeline GUI")
        self.resize(950, 950)
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor("#1e1e2f"))
        palette.setColor(QPalette.WindowText, QColor("#f0f0f0"))
        palette.setColor(QPalette.Base, QColor("#2d2d44"))
        palette.setColor(QPalette.Text, QColor("#f0f0f0"))
        palette.setColor(QPalette.Button, QColor("#252537"))
        palette.setColor(QPalette.ButtonText, QColor("#f0f0f0"))
        self.setPalette(palette)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(4)

        # LEFT PANE
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(8, 8, 8, 8)
        ll.setSpacing(8)

        ig = QGroupBox("📝 User Input")
        ig.setStyleSheet("color:#f0f0f0;")
        il = QVBoxLayout()
        self.input_text = QTextEdit()
        self.input_text.setStyleSheet(
            "background-color:#2d2d44; color:#f0f0f0; border:1px solid #3a3a5e;"
        )
        self.input_text.setPlaceholderText("Paste your document here…")
        il.addWidget(self.input_text)

        sl = QHBoxLayout()
        self.send_button = QPushButton("✉️ Send")
        self.send_button.setStyleSheet("background-color:#30475e; color:#f0f0f0;")
        self.send_button.clicked.connect(self.on_send_clicked)
        self.input_count_label = QLabel("Input Words: 0")
        self.input_count_label.setStyleSheet("color:#f0f0f0;")
        sl.addWidget(self.send_button)
        sl.addWidget(self.input_count_label)
        sl.addStretch()
        il.addLayout(sl)
        ig.setLayout(il)
        ll.addWidget(ig)

        pg = QGroupBox("⚙️ Parameters")
        pg.setStyleSheet("color:#f0f0f0;")
        pfl = QFormLayout()
        self.conc_spinner = QSpinBox()
        self.conc_spinner.setRange(1, 20)
        self.conc_spinner.setValue(5)
        self.conc_spinner.setStyleSheet("background-color:#2d2d44; color:#f0f0f0;")
        self.layer_spinner = QSpinBox()
        self.layer_spinner.setRange(1, 10)
        self.layer_spinner.setValue(2)
        self.layer_spinner.setStyleSheet("background-color:#2d2d44; color:#f0f0f0;")
        pfl.addRow("Parallel Agents:", self.conc_spinner)
        pfl.addRow("Summary Layers:", self.layer_spinner)
        pg.setLayout(pfl)
        ll.addWidget(pg)

        fg = QGroupBox("✏️ Final Agent Prompt")
        fg.setStyleSheet("color:#f0f0f0;")
        fl = QVBoxLayout()
        self.final_prompt_editor = QPlainTextEdit()
        self.final_prompt_editor.setStyleSheet(
            "background-color:#2d2d44; color:#f0f0f0; border:1px solid #3a3a5e;"
        )
        self.final_prompt_editor.setPlainText(
            SYSTEM_PROMPTS.get("final_system_prompt", "")
        )
        fl.addWidget(self.final_prompt_editor)
        self.reset_button = QPushButton("Return to Default")
        self.reset_button.setStyleSheet("background-color:#30475e; color:#f0f0f0;")
        self.reset_button.clicked.connect(self.reset_prompt)
        fl.addWidget(self.reset_button, alignment=Qt.AlignRight)
        fg.setLayout(fl)
        ll.addWidget(fg)

        pb = QPushButton("All System Prompts 📜")
        pb.setStyleSheet("background-color:#30475e; color:#f0f0f0;")
        pb.clicked.connect(self.show_prompts_window)
        ll.addWidget(pb, alignment=Qt.AlignLeft)

        pc = QWidget()
        pcl = QHBoxLayout(pc)
        pcl.setContentsMargins(0, 0, 0, 0)
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(
            "QProgressBar{background-color:#2d2d44; color:#f0f0f0; border:1px solid #3a3a5e;}"
            "QProgressBar::chunk{background-color:#39A000;}"
        )
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.timer_label = QLabel("Elapsed Time: 0.0s")
        self.timer_label.setStyleSheet("color:#f0f0f0;")
        pcl.addWidget(self.progress_bar)
        pcl.addWidget(self.timer_label)
        ll.addWidget(pc)

        splitter.addWidget(left)

        # RIGHT PANE
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(8, 8, 8, 8)
        rl.setSpacing(8)

        vl = QHBoxLayout()
        self.view_log_button = QPushButton("📄 View Log")
        self.view_log_button.setStyleSheet("background-color:#30475e; color:#f0f0f0;")
        self.view_log_button.clicked.connect(self.show_log_window)
        vl.addWidget(self.view_log_button, alignment=Qt.AlignRight)
        rl.addLayout(vl)

        sg = QGroupBox("📊 Complexity Scores")
        sg.setStyleSheet("color:#f0f0f0;")
        slc = QVBoxLayout()
        self.score_container = QVBoxLayout()
        slc.addLayout(self.score_container)
        sg.setLayout(slc)
        rl.addWidget(sg)

        og = QGroupBox("🎯 Final Output")
        og.setStyleSheet("color:#f0f0f0;")
        ol = QVBoxLayout()
        self.final_output_box = QTextEdit()
        self.final_output_box.setReadOnly(True)
        self.final_output_box.setStyleSheet(
            "background-color:#2d2d44; color:#f0f0f0; border:1px solid #3a3a5e;"
        )
        ol.addWidget(self.final_output_box)
        self.output_count_label = QLabel("Output Words: 0")
        self.output_count_label.setStyleSheet("color:#f0f0f0;")
        ol.addWidget(self.output_count_label, alignment=Qt.AlignRight)
        og.setLayout(ol)
        rl.addWidget(og)

        next_btn = QPushButton("Next agent Jim >")
        next_btn.setStyleSheet("background-color:#30475e; color:#f0f0f0;")
        next_btn.clicked.connect(self.on_next_agent)
        rl.addWidget(next_btn, alignment=Qt.AlignRight)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        self.log_window = LogWindow()
        self.prompts_dialog = PromptsWindow(self)
        self.worker = None
        self.start_time = None
        self.start_timer = None

    def on_next_agent(self):
        final_text = self.final_output_box.toPlainText().strip()
        if not final_text:
            self.log_window.append_log("⚠️ Error: No final output to send to Jim.")
            QMessageBox.warning(self, "No Output", "There is no final output from Julia to send to the Jim agent.")
            return

        dialog = JimDialog(final_text, parent=self)
        dialog.exec_()

    def show_prompts_window(self):
        self.final_prompt_editor.setPlainText(
            SYSTEM_PROMPTS.get("final_system_prompt", "")
        )
        self.prompts_dialog.exec_()

    def reset_prompt(self):
        default = DEFAULT_PROMPTS["final_system_prompt"]
        SYSTEM_PROMPTS["final_system_prompt"] = default
        with open(SYSTEM_PROMPTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(SYSTEM_PROMPTS, f, indent=2)
        self.final_prompt_editor.setPlainText(default)

    def update_input_word_count(self):
        text = self.input_text.toPlainText().strip()
        cnt = len(text.split()) if text else 0
        self.input_count_label.setText(f"Input Words: {cnt}")

    def update_output_word_count(self, text):
        cnt = len(text.split()) if text else 0
        self.output_count_label.setText(f"Output Words: {cnt}")

    def append_log(self, text):
        self.log_window.append_log(text)

    def update_progress_bar(self, pct):
        self.progress_bar.setValue(pct)

    def update_timer(self):
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.timer_label.setText(f"Elapsed Time: {elapsed:.1f}s")

    def show_timing(self, layer, dur):
        self.append_log(f"⏱ {layer} took {dur:.1f}s")

    def display_scores(self, scores):
        for i in reversed(range(self.score_container.count())):
            item = self.score_container.itemAt(i)
            if item and item.layout():
                lay = item.layout()
                while lay.count():
                    ch = lay.takeAt(0)
                    if ch.widget():
                        ch.widget().deleteLater()
                self.score_container.removeItem(item)
        for label, score in scores:
            row = QHBoxLayout()
            lbl = QLabel(label + ":")
            lbl.setFixedWidth(200)
            lbl.setStyleSheet("color:#f0f0f0;")
            row.addWidget(lbl)
            for i in range(1, 11):
                box = QFrame()
                box.setFixedSize(QSize(15, 15))
                style = ("background-color:#39A000;border:1px solid #000;"
                         if i <= score else
                         "background-color:#444466;border:1px solid #000;")
                box.setStyleSheet(style)
                row.addWidget(box)
            row.addStretch()
            self.score_container.addLayout(row)

    def on_send_clicked(self):
        txt = self.input_text.toPlainText().strip()
        if not txt:
            self.append_log("⚠️ Error: No input text provided.")
            return
        self.update_input_word_count()
        self.log_window.log_output.clear()
        self.log_window.append_log(f"📝 User Input:\n{txt}\n")
        self.log_window.append_log("🚀 Starting pipeline...\n")
        conc, layers = self.conc_spinner.value(), self.layer_spinner.value()
        SYSTEM_PROMPTS["final_system_prompt"] = self.final_prompt_editor.toPlainText().strip()
        with open(SYSTEM_PROMPTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(SYSTEM_PROMPTS, f, indent=2)
        self.final_output_box.clear()
        self.update_output_word_count("")
        self.progress_bar.setValue(0)
        self.timer_label.setText("Elapsed Time: 0.0s")
        if not self.log_window.isVisible():
            self.log_window.show()
        self.send_button.setEnabled(False)
        self.conc_spinner.setEnabled(False)
        self.layer_spinner.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.input_text.setEnabled(False)
        self.worker = PipelineWorker(txt, conc, layers, SYSTEM_PROMPTS)
        self.worker.log_signal.connect(self.append_log)
        self.worker.progress_signal.connect(self.update_progress_bar)
        self.worker.timing_signal.connect(self.show_timing)
        self.worker.scores_signal.connect(self.display_scores)
        self.worker.finished_signal.connect(self.on_pipeline_finished)
        self.worker.total_time_signal.connect(lambda t: self.append_log(f"⏱ Total processing time: {t:.1f}s"))
        self.start_time = time.time()
        self.start_timer = threading.Timer(0.5, self.timer_tick)
        self.start_timer.start()
        self.worker.start()

    def timer_tick(self):
        self.update_timer()
        if self.worker and self.worker.isRunning():
            self.start_timer = threading.Timer(0.5, self.timer_tick)
            self.start_timer.start()

    def on_pipeline_finished(self, final_text):
        self.send_button.setEnabled(True)
        self.conc_spinner.setEnabled(True)
        self.layer_spinner.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.input_text.setEnabled(True)
        self.final_output_box.setPlainText(final_text)
        self.update_output_word_count(final_text)
        self.append_log("✅ Pipeline complete. Final output displayed.\n")
        ts = int(time.time())
        out_data = {"timestamp": ts, "output": final_text}
        out_path = os.path.join(USER_OUTPUTS_DIR, f"output_{ts}.json")
        with open(out_path, "w", encoding='utf-8') as f:
            json.dump(out_data, f, indent=2)
        self.append_log(f"💾 Saved final output to {out_path}\n")

    def show_log_window(self):
        self.log_window.show()
        self.log_window.raise_()
        self.log_window.activateWindow()

    def closeEvent(self, event):
        try:
            if self.start_timer:
                self.start_timer.cancel()
        except:
            pass
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()