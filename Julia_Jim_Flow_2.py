#!/usr/bin/env python3
"""
PyQt5-based multi-agent, layered summarization & refinement GUI for Ollama LLM,
with a dark-themed, two-pane layout, a pop-up “All System Prompts” dialog, a
pop-up log window (500×700) that auto-scrolls by default, a “JIM Agent” button
in the bottom-right, intelligent code‐block detection + splitting, and a full
JIM pipeline that analyzes the entire log in multiple stages.

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

# ────────────────────────────────────────────────────────────────
# Configuration & Defaults
# ────────────────────────────────────────────────────────────────
OLLAMA_HOST         = 'http://localhost:11434'
MODEL_NAME          = 'llama3.2:3b'

USER_INPUTS_DIR     = "user_inputs"
LOGS_DIR            = "julia_logs"
SYSTEM_PROMPTS_FILE = "system_prompts.json"

os.makedirs(USER_INPUTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Default prompts
DEFAULT_PROMPTS = {
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

# Load or create system prompts file
if not os.path.exists(SYSTEM_PROMPTS_FILE):
    with open(SYSTEM_PROMPTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(DEFAULT_PROMPTS, f, indent=2)
with open(SYSTEM_PROMPTS_FILE, 'r', encoding='utf-8') as f:
    SYSTEM_PROMPTS = json.load(f)


# ────────────────────────────────────────────────────────────────
# Separate Log Window (500×700)
# ────────────────────────────────────────────────────────────────
class LogWindow(QWidget):
    """
    A fixed 500×700 QWidget with a “Pause Auto‐Scroll” button and a QPlainTextEdit
    that shows the entire pipeline log. Auto‐scrolls by default, but can be paused.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Processing Log")
        self.setFixedSize(500, 700)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # Pause/Resume Auto-Scroll Button
        pause_layout = QHBoxLayout()
        self.pause_button = QPushButton("Pause Auto‐Scroll")
        self.pause_button.clicked.connect(self.toggle_autoscroll)
        pause_layout.addWidget(self.pause_button, alignment=Qt.AlignLeft)
        pause_layout.addStretch()
        layout.addLayout(pause_layout)

        # Log Output (read-only, dark style)
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet(
            "background-color: #2d2d44; color: #f0f0f0; border: 1px solid #3a3a5e;"
        )
        layout.addWidget(self.log_output)

        # Auto-scroll is on by default
        self.auto_scroll = True

    def append_log(self, text: str):
        """
        Append text to the log. If auto_scroll is True, move cursor to the end.
        """
        self.log_output.appendPlainText(text)
        if self.auto_scroll:
            cursor = self.log_output.textCursor()
            cursor.movePosition(cursor.End)
            self.log_output.setTextCursor(cursor)
            self.log_output.ensureCursorVisible()

    def toggle_autoscroll(self):
        """Toggle whether new log entries auto-scroll to the bottom."""
        self.auto_scroll = not self.auto_scroll
        if self.auto_scroll:
            self.pause_button.setText("Pause Auto‐Scroll")
            cursor = self.log_output.textCursor()
            cursor.movePosition(cursor.End)
            self.log_output.setTextCursor(cursor)
            self.log_output.ensureCursorVisible()
        else:
            self.pause_button.setText("Resume Auto‐Scroll")


# ────────────────────────────────────────────────────────────────
# System Prompts Management Dialog
# ────────────────────────────────────────────────────────────────
class PromptsWindow(QDialog):
    """
    Modal pop-up dialog listing all system prompts, each editable. Allows updating
    and adding new prompts. Saves to SYSTEM_PROMPTS_FILE.
    """
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

        # Close button at bottom
        btn_box = QDialogButtonBox(QDialogButtonBox.Close)
        btn_box.rejected.connect(self.close)
        layout.addWidget(btn_box)

    def load_prompts(self):
        """
        Load prompts from SYSTEM_PROMPTS and create UI entries.
        """
        # Clear existing
        for i in reversed(range(self.inner_layout.count())):
            item = self.inner_layout.itemAt(i)
            if item and item.widget():
                item.widget().deleteLater()

        # For each prompt: title label, editor, update button
        for key, content in SYSTEM_PROMPTS.items():
            group = QGroupBox(key.replace("_", " ").title())
            group_layout = QVBoxLayout()
            editor = QPlainTextEdit()
            editor.setPlainText(content)
            editor.setStyleSheet(
                "background-color: #2d2d44; color: #f0f0f0; border: 1px solid #3a3a5e;"
            )
            update_btn = QPushButton(f"Update '{key}'")
            update_btn.setStyleSheet("background-color: #30475e; color: #f0f0f0;")
            def make_update_fn(k, ed):
                def upd():
                    SYSTEM_PROMPTS[k] = ed.toPlainText()
                    with open(SYSTEM_PROMPTS_FILE, 'w', encoding='utf-8') as f:
                        json.dump(SYSTEM_PROMPTS, f, indent=2)
                    QMessageBox.information(self, "Prompt Updated", f"'{k}' has been updated.")
                return upd
            update_btn.clicked.connect(make_update_fn(key, editor))
            group_layout.addWidget(editor)
            group_layout.addWidget(update_btn, alignment=Qt.AlignRight)
            group.setLayout(group_layout)
            self.inner_layout.addWidget(group)

    def add_new_prompt(self):
        """
        Open a dialog to enter new prompt key and content, then save.
        """
        dialog = QDialog(self)
        dialog.setWindowTitle("Add New System Prompt")
        dlg_layout = QVBoxLayout(dialog)
        title_edit = QLineEdit()
        title_edit.setPlaceholderText("Enter prompt key (e.g. 'my_custom_prompt')")
        content_edit = QPlainTextEdit()
        content_edit.setPlaceholderText("Enter prompt content here...")
        dlg_layout.addWidget(QLabel("Prompt Key:"))
        dlg_layout.addWidget(title_edit)
        dlg_layout.addWidget(QLabel("Prompt Content:"))
        dlg_layout.addWidget(content_edit)

        btn_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)
        dlg_layout.addWidget(btn_box)

        if dialog.exec_() == QDialog.Accepted:
            key = title_edit.text().strip()
            content = content_edit.toPlainText().strip()
            if not key or not content:
                QMessageBox.warning(self, "Invalid Input", "Both key and content must be non-empty.")
                return
            if key in SYSTEM_PROMPTS:
                QMessageBox.warning(self, "Duplicate Key", f"'{key}' already exists.")
                return
            SYSTEM_PROMPTS[key] = content
            with open(SYSTEM_PROMPTS_FILE, 'w', encoding='utf-8') as f:
                json.dump(SYSTEM_PROMPTS, f, indent=2)
            self.load_prompts()


# ────────────────────────────────────────────────────────────────
# Worker Thread: Does the multi-agent pipeline
# ────────────────────────────────────────────────────────────────
class PipelineWorker(QThread):
    """
    QThread that runs the multi-agent, layered summarization & refinement pipeline.
    Signals:
      - log_signal (str): detailed logs for each step
      - finished_signal (str): final refined output
      - progress_signal (int): percentage progress (0-100)
      - scores_signal (list of (str, int)): complexity scores per layer+final
      - timing_signal (str, float): (layer_name, duration) for each layer
      - total_time_signal (float): total processing time in seconds
    """
    log_signal        = pyqtSignal(str)
    finished_signal   = pyqtSignal(str)
    progress_signal   = pyqtSignal(int)
    scores_signal     = pyqtSignal(list)
    timing_signal     = pyqtSignal(str, float)
    total_time_signal = pyqtSignal(float)

    def __init__(self, user_input, concurrency, num_layers, prompts, parent=None):
        super().__init__(parent)
        self.user_input   = user_input
        self.concurrency  = concurrency
        self.num_layers   = num_layers
        self.prompts      = prompts  # dict of prompts
        self.client       = Client(host=OLLAMA_HOST)
        self.lock         = threading.Lock()
        self.layer_texts  = {}
        # 2*(#layers + 1) steps = first pass (layers + consolidation) + refinement pass
        self.total_steps   = 2 * (self.num_layers + 1)
        self.current_step  = 0

    def split_input(self, text: str):
        """
        Splits the input into a list of “chunks,” where each chunk is either:
         - A standalone code block (lines starting with import/ from → up to its end),
         - Or ordinary prose split by periods.
        """
        lines = text.splitlines(keepends=True)
        chunks = []
        buffer = []

        def flush_buffer():
            """Flush the collected prose buffer into sentence‐based chunks."""
            if not buffer:
                return
            prose = ''.join(buffer).strip()
            buffer.clear()
            # Split on periods
            for sentence in [s.strip() for s in prose.split('.') if s.strip()]:
                chunks.append(sentence)

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.lstrip()
            # Detect the start of a python code block if line starts with import or from
            if stripped.startswith("import ") or stripped.startswith("from "):
                # Flush any accumulated prose first
                flush_buffer()

                # Collect entire code block: lines until a blank line or a line that does not belong to code
                code_block = []
                while i < len(lines):
                    curr = lines[i]
                    if curr.strip() == "":
                        code_block.append(curr)
                        i += 1
                        continue
                    # As long as the line is indented or starts with typical code keywords, keep including
                    if curr.startswith(" ") or curr.startswith("\t") or \
                       curr.lstrip().startswith(("import ", "from ", "class ", "def ", "#")):
                        code_block.append(curr)
                        i += 1
                    else:
                        break

                chunks.append("".join(code_block).rstrip())
            else:
                # Accumulate as prose
                buffer.append(line)
                i += 1

        # At end, flush any leftover prose
        flush_buffer()
        return chunks

    def split_code(self, code: str):
        """
        Given a code chunk, first split by top‐level classes (lines matching ^class <name>),
        then within each class block split into its methods (def ...), and also capture top‐level functions.
        Returns a list of smaller code sub‐chunks in this order:
          1) Each class definition (including its entire body) as one chunk
          2) Each standalone function (def ...) not inside a class as one chunk
        """
        # Regex patterns
        class_pattern = re.compile(r'^(class\s+\w+\(?.*?\)\s*:)', re.MULTILINE)
        func_pattern  = re.compile(r'^(def\s+\w+\(.*?\)\s*:)', re.MULTILINE)

        sub_chunks = []
        lines = code.splitlines(keepends=True)

        # First, find all class definitions and their block ranges
        class_ranges = []
        for m in class_pattern.finditer(code):
            start_idx = code[:m.start()].count("\n")
            # Find where this class ends by tracking indentation
            class_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
            end_idx = start_idx + 1
            while end_idx < len(lines):
                line = lines[end_idx]
                # If a non‐blank line has indentation <= class_indent, we've exited the class block
                if line.strip() and (len(line) - len(line.lstrip())) <= class_indent:
                    break
                end_idx += 1
            class_ranges.append((start_idx, end_idx))

        # Collect class chunks
        for (cs, ce) in class_ranges:
            sub_chunks.append("".join(lines[cs:ce]).rstrip())

        # Mark lines that were inside classes, so we don't re‐process them as top‐level
        covered = set()
        for cs, ce in class_ranges:
            covered.update(range(cs, ce))

        # Next, find standalone functions (not inside any class)
        for m in func_pattern.finditer(code):
            func_idx = code[:m.start()].count("\n")
            if func_idx in covered:
                continue
            # Find the end of this function by indentation
            func_indent = len(lines[func_idx]) - len(lines[func_idx].lstrip())
            end_idx = func_idx + 1
            while end_idx < len(lines):
                line = lines[end_idx]
                if line.strip() and (len(line) - len(line.lstrip())) <= func_indent:
                    break
                end_idx += 1
            sub_chunks.append("".join(lines[func_idx:end_idx]).rstrip())

        return sub_chunks

    def agent_call(self, chunk: str, system_prompt: str):
        """
        Call the LLM with a chunk plus a system prompt. Ensure output ≤ 1000 chars.
        """
        prompt = f"{system_prompt}\n\nUser Chunk:\n{chunk}"
        with self.lock:
            response = self.client.chat(
                model=MODEL_NAME,
                messages=[{'role': 'user', 'content': prompt}]
            )
        content = response.get('message', {}).get('content', '').strip()
        if len(content) > 1000:
            content = content[:1000].rstrip() + "..."
        return content

    def process_layer(self, chunks, concurrency, system_prompt, layer_name):
        """
        Process a list of `chunks` in parallel (up to `concurrency` threads).
        Emit a log entry per chunk. Return list of strings (agent outputs).
        If a chunk looks like code, we further split it and send each sub‐chunk individually.
        """
        results = []

        def task_wrapper(chunk_text):
            # If this chunk is a code block (heuristic: contains '\n' and lines starting with import/def/class),
            # we split it further.
            is_code = bool(re.search(r'^\s*(import |from |class |def )', chunk_text, re.MULTILINE))
            if is_code:
                # First, pass the entire code block as one call
                start = time.time()
                whole_response = self.agent_call(chunk_text, system_prompt)
                duration = time.time() - start
                self.log_signal.emit(
                    f"🔹 {layer_name} – code‐block processed (entire) (⏱ {duration:.1f}s):\n"
                    f"[CODE BLOCK START]\n{chunk_text}\n[CODE BLOCK END]\n→\n{whole_response}\n"
                )
                sub_results = [whole_response]

                # Then split the code into classes/functions and process each sub‐chunk
                sub_chunks = self.split_code(chunk_text)
                for sub in sub_chunks:
                    start2 = time.time()
                    sub_resp = self.agent_call(sub, system_prompt)
                    dur2   = time.time() - start2
                    label = "Class" if sub.strip().startswith("class ") else "Function"
                    self.log_signal.emit(
                        f"🔹 {layer_name} – {label} chunk processed (⏱ {dur2:.1f}s):\n"
                        f"{sub}\n→\n{sub_resp}\n"
                    )
                    sub_results.append(sub_resp)

                return "\n".join(sub_results)
            else:
                # Regular prose sentence chunk
                start = time.time()
                result = self.agent_call(chunk_text, system_prompt)
                duration = time.time() - start
                self.log_signal.emit(
                    f"🔹 {layer_name} – chunk processed (⏱ {duration:.1f}s):\n"
                    f"{chunk_text}\n→\n{result}\n"
                )
                return result

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(task_wrapper, c): c for c in chunks}
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        return results

    def update_progress(self):
        """Increment current_step, compute percentage, and emit it."""
        self.current_step += 1
        pct = int((self.current_step / self.total_steps) * 100)
        self.progress_signal.emit(pct)

    def layered_summary(self, text, concurrency, num_layers, final_prompt, pass_name):
        """
        Perform `num_layers` rounds of:
          1) splitting into chunks (via split_input),
          2) parallel agent calls per chunk (via process_layer),
          3) batch-level summaries when multiple outputs in one batch,
        then one final “consolidation” call per pass.
        Store each layer’s combined output in self.layer_texts[layer_name].
        Emit logs, timings, and progress updates.
        """
        # 1) Initial split into code/prose chunks
        layer_outputs = self.split_input(text)
        self.log_signal.emit(f"   • {pass_name}: Starting with {len(layer_outputs)} chunks.")

        for layer_idx in range(num_layers):
            layer_name   = f"{pass_name} – Layer {layer_idx+1}"
            start_time   = time.time()
            system_prompt = self.prompts["layer_system_prompt"]
            self.log_signal.emit(f"\n🔸 {layer_name}: processing {len(layer_outputs)} chunks...")

            new_outputs = []
            for i in range(0, len(layer_outputs), concurrency):
                batch = layer_outputs[i : i + concurrency]
                self.log_signal.emit(f"   ⏳ {layer_name} – batch {i//concurrency + 1}...")
                batch_outputs = self.process_layer(batch, concurrency, system_prompt, layer_name)

                # If more than one chunk in this batch, summarize at batch-level
                if len(batch_outputs) > 1:
                    combined = "\n".join(batch_outputs)
                    batch_summary = self.agent_call(
                        combined,
                        self.prompts["batch_summary_prompt"]
                    )
                    self.log_signal.emit(f"   🟡 {layer_name} batch summary:\n{batch_summary}\n")
                    new_outputs.append(batch_summary)
                else:
                    new_outputs.extend(batch_outputs)

            duration = time.time() - start_time
            combined_layer_output = "\n".join(new_outputs)
            self.layer_texts[layer_name] = combined_layer_output
            self.timing_signal.emit(layer_name, duration)
            self.log_signal.emit(
                f"✅ {layer_name} complete in {duration:.1f}s. {len(new_outputs)} chunks remain.\n"
            )
            layer_outputs = new_outputs
            self.update_progress()

        # 2) Consolidation for this pass
        cons_name = f"{pass_name} – Consolidation"
        start_time = time.time()
        final_input = "\n".join(layer_outputs)
        self.layer_texts[cons_name] = final_input
        self.log_signal.emit(f"🔷 {cons_name}: calling final agent with aggregated summaries…")
        final_output = self.agent_call(
            f"ORIGINAL INPUT:\n{text}\n\nAGGREGATED SUMMARIES:\n{final_input}",
            final_prompt
        )
        duration = time.time() - start_time
        self.timing_signal.emit(cons_name, duration)
        self.log_signal.emit(f"✅ {cons_name} complete in {duration:.1f}s.\n")
        self.update_progress()

        return final_output

    def score_text(self, text: str, label: str):
        """
        Ask the LLM to rate complexity of `text` on a scale from 1–10. Return an integer.
        Emit a log entry.
        """
        prompt = (
            "Rate the complexity of the following text on a scale from 1 (very simple) "
            "to 10 (very complex). Only respond with an integer (1–10).\n\n"
            f"Text:\n{text}"
        )
        with self.lock:
            response = self.client.chat(
                model=MODEL_NAME,
                messages=[{'role': 'user', 'content': prompt}]
            )
        cont = response.get('message', {}).get('content', '').strip()
        try:
            score = int(cont)
            if score < 1 or score > 10:
                score = 5
        except:
            score = 5
        self.log_signal.emit(f"🏷 Complexity score for {label}: {score}/10\n")
        return score

    def run(self):
        overall_start = time.time()

        # 1) First Pass
        self.log_signal.emit("🟢 Splitting input into chunks…")
        intermediate = self.layered_summary(
            text=self.user_input,
            concurrency=self.concurrency,
            num_layers=self.num_layers,
            final_prompt=self.prompts["final_system_prompt"],
            pass_name="First Pass"
        )
        self.log_signal.emit(f"\n🔵 Intermediate output from First Pass:\n{intermediate}\n")

        # 2) Refinement Pass
        self.log_signal.emit("🔄 Starting Refinement Pass on intermediate output…\n")
        refined = self.layered_summary(
            text=intermediate,
            concurrency=self.concurrency,
            num_layers=self.num_layers,
            final_prompt=self.prompts["final_system_prompt"],
            pass_name="Refinement Pass"
        )
        self.log_signal.emit("✅ Refinement Pass complete.\n")

        total_duration = time.time() - overall_start
        self.total_time_signal.emit(total_duration)

        # 3) Complexity Scores for each layer + final
        scores = []
        for label, txt in self.layer_texts.items():
            score = self.score_text(txt, label)
            scores.append((label, score))
        final_score = self.score_text(refined, "Final Output")
        scores.append(("Final Output", final_score))
        self.scores_signal.emit(scores)

        # 4) Emit final result
        self.finished_signal.emit(refined)


# ────────────────────────────────────────────────────────────────
# Worker Thread: The JIM Agent Pipeline
# ────────────────────────────────────────────────────────────────
class JimWorker(QThread):
    """
    QThread that runs the JIM Agent pipeline over the entire log text.
    - Splits into 1000-character chunks
    - For each chunk, asks LLM (separately) for each of the ten fields
    - Logs every answer
    - Then concatenates all field-answers → splits into 1500-character chunks → summarizes each to ≤1000 chars
      over a configurable number of layers & concurrency
    - Finally, concatenates all final summaries and asks LLM once more to produce all ten fields in one JSON.

    Signals:
      - log_signal (str): detailed logs for each step
      - progress_signal (int): percentage 0–100
      - finished_signal (str): the final JSON output (under 1000 chars)
    """
    log_signal      = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(str)

    def __init__(self, full_log_text, concurrency, num_layers, parent=None):
        super().__init__(parent)
        self.full_log_text = full_log_text
        self.concurrency   = concurrency
        self.num_layers    = num_layers
        self.client        = Client(host=OLLAMA_HOST)
        self.lock          = threading.Lock()

        # The ten fields JIM must answer per chunk
        self.fields = [
            "Key points",
            "Key people",
            "Key ideas",
            "Economic viability",
            "Questions",
            "Problems that could arise",
            "New ideas that complement key ideas",
            "Tone",
            "What is innovation",
            "Research that needs to be done"
        ]

    def chunk_by_chars(self, text: str, limit: int):
        """
        Split `text` into chunks of at most `limit` characters, breaking at the last
        whitespace before the limit. Returns a list of chunks.
        """
        chunks = []
        idx = 0
        length = len(text)
        while idx < length:
            if length - idx <= limit:
                chunks.append(text[idx:].strip())
                break
            # Otherwise, find last whitespace before idx+limit
            split_pos = idx + limit
            while split_pos > idx and not text[split_pos].isspace():
                split_pos -= 1
            if split_pos == idx:
                # No whitespace found, just force split at limit
                split_pos = idx + limit
            chunks.append(text[idx:split_pos].strip())
            idx = split_pos
        return chunks

    def agent_call(self, prompt: str):
        """
        Make a single LLM call with `prompt`. Return the text (truncate at 1000 chars
        if necessary).
        """
        with self.lock:
            response = self.client.chat(
                model=MODEL_NAME,
                messages=[{'role': 'user', 'content': prompt}]
            )
        content = response.get('message', {}).get('content', '').strip()
        if len(content) > 1000:
            content = content[:1000].rstrip() + "..."
        return content

    def run(self):
        """
        Execute the full JIM pipeline:
          1) Split `full_log_text` into 1000-char chunks
          2) For each chunk, for each of the ten fields, do a separate LLM call and log the answer
          3) Concatenate all those field-answers into one big string
          4) Over `num_layers` iterations:
             a) Split current combined text into 1500-char chunks
             b) For each chunk (in parallel, up to `concurrency`), ask LLM to “Summarize in ≤1000 chars”
             c) Concatenate those summaries into a new combined text (next layer’s input)
          5) Finally, take the fully-concatenated text from the last layer and ask LLM once more to produce all ten fields in one JSON
        """
        # 1) First, split full log into 1000-character chunks
        self.log_signal.emit("🦊 JIM: Splitting entire log into 1,000-char chunks…")
        chunks_1000 = self.chunk_by_chars(self.full_log_text, 1000)
        total_calls_first_phase = len(chunks_1000) * len(self.fields)
        completed_calls = 0

        # 2) For each chunk, for each field, call LLM
        all_field_answers = []
        for idx, chunk in enumerate(chunks_1000, start=1):
            self.log_signal.emit(f"\n🔹 Processing chunk {idx}/{len(chunks_1000)} (1,000-char) …\n")
            chunk_answers = {}
            for field in self.fields:
                prompt = f"{field}:\n\n{chunk}"
                answer = self.agent_call(prompt)
                chunk_answers[field] = answer
                self.log_signal.emit(f"• [{field}] →\n{answer}\n")
                completed_calls += 1
                pct = int((completed_calls / total_calls_first_phase) * 100 * 0.3)  # first phase = 30%
                self.progress_signal.emit(pct)
            # Combine this chunk’s field answers as one sub‐block
            block = "\n\n".join(f"{field}: {chunk_answers[field]}" for field in self.fields)
            all_field_answers.append(block)

        # 3) Concatenate all field‐answers
        combined_text = "\n\n".join(all_field_answers)
        self.log_signal.emit("\n🔷 JIM: Completed first phase (field extraction). Now concatenating all answers.\n")

        # 4) Summarization over layers
        current_text = combined_text
        for layer in range(1, self.num_layers + 1):
            self.log_signal.emit(f"\n🔸 JIM Summarization Layer {layer}: splitting into 1,500-char chunks…\n")
            chunks_1500 = self.chunk_by_chars(current_text, 1500)
            self.log_signal.emit(f"   • {len(chunks_1500)} chunk(s) at layer {layer}.\n")

            # Summarize each chunk in parallel (up to concurrency)
            layer_summaries = []

            def summarize_chunk(c_text):
                prompt = f"Summarize the following in 1000 characters or fewer:\n\n{c_text}"
                resp = self.agent_call(prompt)
                self.log_signal.emit(f"🔹 Layer {layer} chunk summary:\n{resp}\n")
                return resp

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency) as executor:
                futures = {executor.submit(summarize_chunk, c): c for c in chunks_1500}
                for future in concurrent.futures.as_completed(futures):
                    layer_summaries.append(future.result())

            current_text = "\n".join(layer_summaries)
            pct = 30 + int((layer / self.num_layers) * 50)  # next 50% of progress
            self.progress_signal.emit(pct)

        self.log_signal.emit("\n✅ JIM: Summarization layers complete.\n")

        # 5) Final single LLM call for all fields over the final combined text
        self.log_signal.emit("\n🔷 JIM: Final aggregation—asking for all fields in one JSON…\n")
        final_prompt = (
            "Using the text below, provide a JSON object with the following keys:\n"
            "- Key points\n"
            "- Key people\n"
            "- Key ideas\n"
            "- Economic viability\n"
            "- Questions\n"
            "- Problems that could arise\n"
            "- New ideas that complement key ideas\n"
            "- Tone\n"
            "- What is innovation\n"
            "- Research that needs to be done\n\n"
            "Ensure the JSON fits under 1000 characters. Text:\n\n"
            f"{current_text}"
        )
        final_answer = self.agent_call(final_prompt)
        self.log_signal.emit(f"🔹 Final JIM JSON Output:\n{final_answer}\n")
        self.progress_signal.emit(100)

        # Emit final
        self.finished_signal.emit(final_answer)


# ────────────────────────────────────────────────────────────────
# JIM Agent Dialog
# ────────────────────────────────────────────────────────────────
class JimDialog(QDialog):
    """
    Modal dialog for the JIM Agent. It shows:
      - A multi‐line text area pre‐filled with the entire log
      - Concurrency spinner, Layers spinner
      - Send button
      - Progress bar
      - A “JIM Log” area (QPlainTextEdit) where per‐chunk field answers & summaries appear
      - A “JIM Final Output” area (QTextEdit) with the final JSON
    """
    def __init__(self, parent=None, full_log_text=""):
        super().__init__(parent)
        self.setWindowTitle("JIM Agent")
        self.resize(800, 800)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # 1) Input Log (read-only)
        input_group = QGroupBox("🗂 Input Log (from main pipeline)")
        input_group.setStyleSheet("QGroupBox { color: #f0f0f0; }")
        input_layout = QVBoxLayout()
        self.input_area = QTextEdit()
        self.input_area.setPlainText(full_log_text)
        self.input_area.setReadOnly(True)
        self.input_area.setStyleSheet(
            "background-color: #2d2d44; color: #f0f0f0; border: 1px solid #3a3a5e;"
        )
        input_layout.addWidget(self.input_area)
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # 2) Parameters (Concurrency & Layers)
        param_group = QGroupBox("⚙️ JIM Parameters")
        param_group.setStyleSheet("QGroupBox { color: #f0f0f0; }")
        param_layout = QFormLayout()
        self.jim_conc_spinner  = QSpinBox()
        self.jim_conc_spinner.setRange(1, 10)
        self.jim_conc_spinner.setValue(3)
        self.jim_conc_spinner.setStyleSheet("background-color: #2d2d44; color: #f0f0f0;")
        self.jim_layer_spinner = QSpinBox()
        self.jim_layer_spinner.setRange(1, 5)
        self.jim_layer_spinner.setValue(1)
        self.jim_layer_spinner.setStyleSheet("background-color: #2d2d44; color: #f0f0f0;")
        param_layout.addRow("Parallel Calls:", self.jim_conc_spinner)
        param_layout.addRow("Summarization Layers:", self.jim_layer_spinner)
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # 3) Send Button & Progress Bar
        control_layout = QHBoxLayout()
        self.jim_send_button = QPushButton("🚀 Run JIM")
        self.jim_send_button.setStyleSheet("background-color: #30475e; color: #f0f0f0;")
        self.jim_send_button.clicked.connect(self.on_jim_send)
        self.jim_progress = QProgressBar()
        self.jim_progress.setStyleSheet(
            "QProgressBar { background-color: #2d2d44; color: #f0f0f0; border: 1px solid #3a3a5e; }"
            "QProgressBar::chunk { background-color: #ffaa00; }"
        )
        self.jim_progress.setValue(0)
        self.jim_progress.setFormat("%p%")
        control_layout.addWidget(self.jim_send_button)
        control_layout.addWidget(self.jim_progress)
        layout.addLayout(control_layout)

        # 4) JIM Log Output (where each field answer & summary is appended)
        jim_log_group = QGroupBox("📑 JIM Log")
        jim_log_group.setStyleSheet("QGroupBox { color: #f0f0f0; }")
        jim_log_layout = QVBoxLayout()
        self.jim_log_area = QPlainTextEdit()
        self.jim_log_area.setReadOnly(True)
        self.jim_log_area.setStyleSheet(
            "background-color: #2d2d44; color: #f0f0f0; border: 1px solid #3a3a5e;"
        )
        jim_log_layout.addWidget(self.jim_log_area)
        jim_log_group.setLayout(jim_log_layout)
        layout.addWidget(jim_log_group, stretch=2)

        # 5) JIM Final Output (JSON)
        final_group = QGroupBox("🎯 JIM Final Output (JSON)")
        final_group.setStyleSheet("QGroupBox { color: #f0f0f0; }")
        final_layout = QVBoxLayout()
        self.jim_final_output = QTextEdit()
        self.jim_final_output.setReadOnly(True)
        self.jim_final_output.setStyleSheet(
            "background-color: #2d2d44; color: #f0f0f0; border: 1px solid #3a3a5e;"
        )
        final_layout.addWidget(self.jim_final_output)
        final_group.setLayout(final_layout)
        layout.addWidget(final_group, stretch=1)

        # Worker placeholder
        self.worker = None

    def append_jim_log(self, text: str):
        """Append to the JIM log area (auto-scroll)."""
        self.jim_log_area.appendPlainText(text)
        cursor = self.jim_log_area.textCursor()
        cursor.movePosition(cursor.End)
        self.jim_log_area.setTextCursor(cursor)
        self.jim_log_area.ensureCursorVisible()

    def on_jim_send(self):
        """
        When “Run JIM” is clicked:
        1) Disable controls
        2) Clear JIM log & final output
        3) Start JimWorker with full log text
        """
        full_log = self.input_area.toPlainText().strip()
        if not full_log:
            QMessageBox.warning(self, "No Log Found", "There is no log to analyze.")
            return

        # Disable controls
        self.jim_send_button.setEnabled(False)
        self.jim_conc_spinner.setEnabled(False)
        self.jim_layer_spinner.setEnabled(False)

        # Clear JIM log & final output
        self.jim_log_area.clear()
        self.jim_final_output.clear()
        self.jim_progress.setValue(0)

        concurrency = self.jim_conc_spinner.value()
        layers      = self.jim_layer_spinner.value()

        # Start JimWorker
        self.worker = JimWorker(full_log, concurrency, layers)
        self.worker.log_signal.connect(self.append_jim_log)
        self.worker.progress_signal.connect(self.jim_progress.setValue)
        self.worker.finished_signal.connect(self.on_jim_finished)
        self.worker.start()

    def on_jim_finished(self, final_json: str):
        """Re-enable controls and display final JSON output."""
        self.jim_send_button.setEnabled(True)
        self.jim_conc_spinner.setEnabled(True)
        self.jim_layer_spinner.setEnabled(True)

        self.jim_final_output.setPlainText(final_json)
        self.append_jim_log("✅ JIM: Completed final JSON output.\n")


# ────────────────────────────────────────────────────────────────
# Main Window: Dark-themed, Two-Pane Layout
# ────────────────────────────────────────────────────────────────
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Agentic LLM Pipeline GUI")
        self.resize(950, 950)

        # Apply a dark palette globally
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor("#1e1e2f"))
        palette.setColor(QPalette.WindowText, QColor("#f0f0f0"))
        palette.setColor(QPalette.Base, QColor("#2d2d44"))
        palette.setColor(QPalette.Text, QColor("#f0f0f0"))
        palette.setColor(QPalette.Button, QColor("#252537"))
        palette.setColor(QPalette.ButtonText, QColor("#f0f0f0"))
        self.setPalette(palette)

        # Main layout: horizontal splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(4)

        # ─────────────────────────────────────────────────────────────
        # LEFT PANE: Input + Controls + “All System Prompts” button
        # ─────────────────────────────────────────────────────────────
        left_pane = QWidget()
        left_layout = QVBoxLayout(left_pane)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(8)

        # 1) User Input Group
        input_group = QGroupBox("📝 User Input")
        input_group.setStyleSheet("QGroupBox { color: #f0f0f0; }")
        input_layout = QVBoxLayout()
        self.input_text = QTextEdit()
        self.input_text.setStyleSheet(
            "background-color: #2d2d44; color: #f0f0f0; border: 1px solid #3a3a5e;"
        )
        self.input_text.setPlaceholderText("Paste your document here...")
        input_layout.addWidget(self.input_text)

        send_layout = QHBoxLayout()
        self.send_button = QPushButton("✉️ Send")
        self.send_button.setStyleSheet("background-color: #30475e; color: #f0f0f0;")
        self.send_button.clicked.connect(self.on_send_clicked)
        self.input_count_label = QLabel("Input Words: 0")
        self.input_count_label.setStyleSheet("color: #f0f0f0;")
        send_layout.addWidget(self.send_button, alignment=Qt.AlignLeft)
        send_layout.addWidget(self.input_count_label, alignment=Qt.AlignLeft)
        send_layout.addStretch()
        input_layout.addLayout(send_layout)

        input_group.setLayout(input_layout)
        left_layout.addWidget(input_group)

        # 2) Pipeline Parameters
        param_group = QGroupBox("⚙️ Parameters")
        param_group.setStyleSheet("QGroupBox { color: #f0f0f0; }")
        param_layout = QFormLayout()
        self.conc_spinner = QSpinBox()
        self.conc_spinner.setRange(1, 20)
        self.conc_spinner.setValue(5)
        self.conc_spinner.setStyleSheet("background-color: #2d2d44; color: #f0f0f0;")
        self.layer_spinner = QSpinBox()
        self.layer_spinner.setRange(1, 10)
        self.layer_spinner.setValue(2)
        self.layer_spinner.setStyleSheet("background-color: #2d2d44; color: #f0f0f0;")
        param_layout.addRow("Parallel Agents:", self.conc_spinner)
        param_layout.addRow("Summary Layers:", self.layer_spinner)
        param_group.setLayout(param_layout)
        left_layout.addWidget(param_group)

        # 3) Final Agent Prompt Editor
        final_group = QGroupBox("✏️ Final Agent Prompt")
        final_group.setStyleSheet("QGroupBox { color: #f0f0f0; }")
        final_layout = QVBoxLayout()
        self.final_prompt_editor = QPlainTextEdit()
        self.final_prompt_editor.setStyleSheet(
            "background-color: #2d2d44; color: #f0f0f0; border: 1px solid #3a3a5e;"
        )
        self.final_prompt_editor.setPlainText(SYSTEM_PROMPTS.get("final_system_prompt", ""))
        final_layout.addWidget(self.final_prompt_editor)
        self.reset_button = QPushButton("Return to Default")
        self.reset_button.setStyleSheet("background-color: #30475e; color: #f0f0f0;")
        self.reset_button.clicked.connect(self.reset_prompt)
        final_layout.addWidget(self.reset_button, alignment=Qt.AlignRight)
        final_group.setLayout(final_layout)
        left_layout.addWidget(final_group)

        # 4) “All System Prompts” Button
        prompts_button = QPushButton("All System Prompts 📜")
        prompts_button.setStyleSheet("background-color: #30475e; color: #f0f0f0;")
        prompts_button.clicked.connect(self.show_prompts_window)
        left_layout.addWidget(prompts_button, alignment=Qt.AlignLeft)

        # 5) Progress Bar & Timer
        progress_container = QWidget()
        progress_layout = QHBoxLayout(progress_container)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(
            "QProgressBar { background-color: #2d2d44; color: #f0f0f0; border: 1px solid #3a3a5e; }"
            "QProgressBar::chunk { background-color: #39A000; }"
        )
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.timer_label = QLabel("Elapsed Time: 0.0s")
        self.timer_label.setStyleSheet("color: #f0f0f0;")
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.timer_label)
        left_layout.addWidget(progress_container)

        splitter.addWidget(left_pane)

        # ─────────────────────────────────────────────────────────────
        # RIGHT PANE: “View Log” button + Scores + Final Output + JIM Button
        # ─────────────────────────────────────────────────────────────
        right_pane = QWidget()
        right_layout = QVBoxLayout(right_pane)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(8)

        # “View Log” Button (top-right)
        view_log_layout = QHBoxLayout()
        self.view_log_button = QPushButton("📄 View Log")
        self.view_log_button.setStyleSheet("background-color: #30475e; color: #f0f0f0;")
        self.view_log_button.clicked.connect(self.show_log_window)
        view_log_layout.addWidget(self.view_log_button, alignment=Qt.AlignRight)
        right_layout.addLayout(view_log_layout)

        # Layer & Final Complexity Scores
        score_group = QGroupBox("📊 Complexity Scores")
        score_group.setStyleSheet("QGroupBox { color: #f0f0f0; }")
        score_layout = QVBoxLayout()
        self.score_container = QVBoxLayout()
        score_layout.addLayout(self.score_container)
        score_group.setLayout(score_layout)
        right_layout.addWidget(score_group)

        # Final Output & Word Count
        output_group = QGroupBox("🎯 Final Output")
        output_group.setStyleSheet("QGroupBox { color: #f0f0f0; }")
        output_layout = QVBoxLayout()
        self.final_output_box = QTextEdit()
        self.final_output_box.setReadOnly(True)
        self.final_output_box.setStyleSheet(
            "background-color: #2d2d44; color: #f0f0f0; border: 1px solid #3a3a5e;"
        )
        output_layout.addWidget(self.final_output_box)
        self.output_count_label = QLabel("Output Words: 0")
        self.output_count_label.setStyleSheet("color: #f0f0f0;")
        output_layout.addWidget(self.output_count_label, alignment=Qt.AlignRight)
        output_group.setLayout(output_layout)
        right_layout.addWidget(output_group)

        # Spacer to push JIM button to bottom
        right_layout.addStretch()

        # JIM Agent Button (bottom-right)
        jim_layout = QHBoxLayout()
        jim_layout.addStretch()
        self.jim_button = QPushButton("🦊 JIM Agent")
        self.jim_button.setStyleSheet("background-color: #30475e; color: #f0f0f0;")
        self.jim_button.clicked.connect(self.open_jim_agent)
        jim_layout.addWidget(self.jim_button, alignment=Qt.AlignRight)
        right_layout.addLayout(jim_layout)

        splitter.addWidget(right_pane)

        # Left pane = 40% width, Right pane = 60% width
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        # Put splitter into main_window’s layout
        container = QVBoxLayout(self)
        container.setContentsMargins(0, 0, 0, 0)
        container.addWidget(splitter)

        # Create the Log Window and Prompts Dialog (hidden initially)
        self.log_window     = LogWindow()
        self.prompts_dialog = PromptsWindow(self)

        # Worker thread and timer handle
        self.worker      = None
        self.start_time  = None
        self.start_timer = None

    # ─────────────────────────────────────────────────────────────
    # Show prompts management dialog (modal pop-up)
    # ─────────────────────────────────────────────────────────────
    def show_prompts_window(self):
        """Open the All System Prompts pop‐up as a modal dialog."""
        # Refresh the final prompt editor with current final_system_prompt
        self.final_prompt_editor.setPlainText(SYSTEM_PROMPTS.get("final_system_prompt", ""))
        # Show dialog modally
        self.prompts_dialog.exec_()

    def reset_prompt(self):
        """Reset the final prompt editor to the default final_system_prompt."""
        default = DEFAULT_PROMPTS["final_system_prompt"]
        SYSTEM_PROMPTS["final_system_prompt"] = default
        with open(SYSTEM_PROMPTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(SYSTEM_PROMPTS, f, indent=2)
        self.final_prompt_editor.setPlainText(default)

    def update_input_word_count(self):
        """Count words in input and update the label."""
        text = self.input_text.toPlainText().strip()
        count = len(text.split()) if text else 0
        self.input_count_label.setText(f"Input Words: {count}")

    def update_output_word_count(self, text):
        """Count words in final output and update the label."""
        count = len(text.split()) if text else 0
        self.output_count_label.setText(f"Output Words: {count}")

    def append_log(self, text: str):
        """Forward log text to the Log Window’s append_log()."""
        self.log_window.append_log(text)

    def update_progress_bar(self, pct: int):
        """Update the progress bar to the given percentage."""
        self.progress_bar.setValue(pct)

    def update_timer(self):
        """Refresh the elapsed-time label every 0.5 seconds."""
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.timer_label.setText(f"Elapsed Time: {elapsed:.1f}s")

    def show_timing(self, layer_name: str, duration: float):
        """Log how long each layer or consolidation took."""
        self.append_log(f"⏱ {layer_name} took {duration:.1f} seconds.")

    def display_scores(self, scores: list):
        """
        Display complexity scores (1–10) as rows of ten small squares (green/gray)
        underneath “Complexity Scores.” Each row’s label (e.g., “First Pass – Layer 1:”)
        is followed by ten boxes.
        """
        # Clear any existing rows
        for i in reversed(range(self.score_container.count())):
            item = self.score_container.itemAt(i)
            if item:
                layout = item.layout()
                if layout:
                    while layout.count():
                        child = layout.takeAt(0)
                        if child.widget():
                            child.widget().deleteLater()
                    self.score_container.removeItem(item)

        # Add a new row per (label, score)
        for label, score in scores:
            row_layout = QHBoxLayout()
            lbl = QLabel(label + ":")
            lbl.setFixedWidth(200)
            lbl.setStyleSheet("color: #f0f0f0;")
            row_layout.addWidget(lbl)
            for i in range(1, 11):
                box = QFrame()
                box.setFixedSize(QSize(15, 15))
                if i <= score:
                    box.setStyleSheet("background-color: #39A000; border: 1px solid #000;")
                else:
                    box.setStyleSheet("background-color: #444466; border: 1px solid #000;")
                row_layout.addWidget(box)
            row_layout.addStretch()
            self.score_container.addLayout(row_layout)

    def on_send_clicked(self):
        """
        When “Send” is clicked:
        1) Log the raw user input at top of LogWindow
        2) Update input word count
        3) Auto-show LogWindow if hidden
        4) Disable UI controls, start PipelineWorker
        """
        user_text = self.input_text.toPlainText().strip()
        if not user_text:
            # If no input, log an error
            self.append_log("⚠️ Error: No input text provided.")
            return

        # 1) Update word count, log the input
        self.update_input_word_count()
        self.log_window.log_output.clear()
        self.log_window.append_log(f"📝 User Input:\n{user_text}\n")
        self.log_window.append_log("🚀 Starting pipeline...\n")

        # 2) Input parameters
        concurrency = self.conc_spinner.value()
        layers      = self.layer_spinner.value()
        # Save final_system_prompt from editor
        SYSTEM_PROMPTS["final_system_prompt"] = self.final_prompt_editor.toPlainText().strip()
        with open(SYSTEM_PROMPTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(SYSTEM_PROMPTS, f, indent=2)

        # 3) Clear final output + reset progress/timer/scores
        self.final_output_box.clear()
        self.update_output_word_count("")
        self.progress_bar.setValue(0)
        self.timer_label.setText("Elapsed Time: 0.0s")
        if not self.log_window.isVisible():
            self.log_window.show()

        # 4) Disable UI controls
        self.send_button.setEnabled(False)
        self.conc_spinner.setEnabled(False)
        self.layer_spinner.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.input_text.setEnabled(False)

        # 5) Start the background worker
        worker_prompts = SYSTEM_PROMPTS.copy()
        self.worker = PipelineWorker(user_text, concurrency, layers, worker_prompts)
        self.worker.log_signal.connect(self.append_log)
        self.worker.progress_signal.connect(self.update_progress_bar)
        self.worker.timing_signal.connect(self.show_timing)
        self.worker.scores_signal.connect(self.display_scores)
        self.worker.finished_signal.connect(self.on_pipeline_finished)
        self.worker.total_time_signal.connect(
            lambda t: self.append_log(f"⏱ Total processing time: {t:.1f} seconds\n")
        )

        self.start_time = time.time()
        # Kick off a timer to update the elapsed time every 0.5 s
        self.start_timer = threading.Timer(0.5, self.timer_tick)
        self.start_timer.start()

        self.worker.start()

    def timer_tick(self):
        """Called every 0.5 s to update the elapsed-time label."""
        self.update_timer()
        if self.worker and self.worker.isRunning():
            self.start_timer = threading.Timer(0.5, self.timer_tick)
            self.start_timer.start()

    def on_pipeline_finished(self, final_text: str):
        """Re-enable UI, display final output, and update word count."""
        self.send_button.setEnabled(True)
        self.conc_spinner.setEnabled(True)
        self.layer_spinner.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.input_text.setEnabled(True)

        self.final_output_box.setPlainText(final_text)
        self.update_output_word_count(final_text)
        self.append_log("✅ Pipeline complete. Final output displayed.\n")

    def show_log_window(self):
        """Show (or raise) the separate Log Window."""
        self.log_window.show()
        self.log_window.raise_()
        self.log_window.activateWindow()

    def open_jim_agent(self):
        """
        Open the JIM Agent dialog, passing the entire log as input.
        """
        full_log = self.log_window.log_output.toPlainText()
        jim_dialog = JimDialog(self, full_log)
        jim_dialog.exec_()

    def closeEvent(self, event):
        """Cancel the timer thread if the window closes."""
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
