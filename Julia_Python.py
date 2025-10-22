#!/usr/bin/env python3
"""
Julia_Python.py - A specialized agent for deep analysis of Python scripts.

This agent takes a Python script as input and performs a multi-pass analysis,
saving the final report, intermediate 'thinking' steps, and full runtime logs
into a structured directory layout.

Model: llama3.2:3b
"""

import sys
import os
import json
import threading
import time
import concurrent.futures
import re
import ast
import textwrap
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

# New, structured directory paths
FINAL_OUTPUTS_DIR = "julia_python_outputs"
JULIA_PYTHON_BASE_DIR = "julia_python"
THINKING_LOGS_DIR = os.path.join(JULIA_PYTHON_BASE_DIR, "thinking_logs")
LOGS_DIR = os.path.join(JULIA_PYTHON_BASE_DIR, "run_logs")
SYSTEM_PROMPTS_FILE = "julia_python_prompts.json"

# Create all necessary directories on startup
os.makedirs(FINAL_OUTPUTS_DIR, exist_ok=True)
os.makedirs(THINKING_LOGS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Default prompts specialized for Python script analysis
DEFAULT_PROMPTS = {
    "python_overall_summary_prompt": (
        "You are a senior software architect. Provide a high-level technical summary of the provided Python script. "
        "Explain its primary purpose, architecture, and what problem it solves. The response must be at least 2000 characters."
    ),
    "python_imports_prompt": (
        "You are a dependency analysis expert. Given the script's summary for context, analyze the following import statements. "
        "For each library, explain its role in this specific script and why it was likely chosen. The response must be a detailed, "
        "technical explanation of at least 2000 characters."
    ),
    "python_globals_prompt": (
        "You are a code reviewer. Given the script's summary for context, analyze the following global variables and constants. "
        "Explain their purpose, data types, and how they influence the script's execution. The response must be a detailed, "
        "technical explanation of at least 2000 characters."
    ),
    "python_function_prompt": (
        "You are a code documentation specialist. Given the script's summary for context, provide a detailed technical explanation "
        "of the following function. Describe its purpose, parameters, internal logic, and return values. "
        "The response must be at least 2000 characters."
    ),
    "python_class_prompt": (
        "You are an object-oriented design expert. Given the script's summary for context, provide a detailed technical analysis "
        "of the following class. Describe its role, attributes, and a breakdown of each method's purpose and logic. "
        "The response must be at least 2000 characters."
    ),
    "final_consolidation_prompt": (
        "You are a technical writer creating a final report. You have been given a series of detailed analyses for different parts of a Python script "
        "(overall summary, imports, functions, classes). Your task is to synthesize all of this information into a single, cohesive, and well-structured "
        "technical document. Do not just list the parts; integrate them into a readable and comprehensive explanation of the entire script."
    )
}

if not os.path.exists(SYSTEM_PROMPTS_FILE):
    with open(SYSTEM_PROMPTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(DEFAULT_PROMPTS, f, indent=2)
with open(SYSTEM_PROMPTS_FILE, 'r', encoding='utf-8') as f:
    SYSTEM_PROMPTS = json.load(f)
is_updated = False
for k, v in DEFAULT_PROMPTS.items():
    if k not in SYSTEM_PROMPTS:
        SYSTEM_PROMPTS[k] = v
        is_updated = True
if is_updated:
    with open(SYSTEM_PROMPTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(SYSTEM_PROMPTS, f, indent=2)


# ────────────────────────────────────────────────────────────────
# Helper Function to Validate LLM Responses
# ────────────────────────────────────────────────────────────────
def is_valid_response(text: str) -> bool:
    if not text or not text.strip(): return False
    lower_text = text.lower()
    refusal_phrases = ["i can't", "i cannot", "unable to", "as an ai"]
    if any(phrase in lower_text for phrase in refusal_phrases): return False
    return True


# ────────────────────────────────────────────────────────────────
# Log Window and PromptsWindow
# ────────────────────────────────────────────────────────────────
class LogWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Julia-Python Log")
        self.setFixedSize(500, 700)
        layout = QVBoxLayout(self)
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("background-color: #2d2d44; color: #f0f0f0;")
        layout.addWidget(self.log_output)

    def append_log(self, text: str):
        self.log_output.appendPlainText(text)
        self.log_output.verticalScrollBar().setValue(self.log_output.verticalScrollBar().maximum())


class PromptsWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Julia-Python System Prompts")
        self.resize(600, 500)
        layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        self.inner_layout = QVBoxLayout(inner)
        self.editors = {}
        for key, content in sorted(SYSTEM_PROMPTS.items()):
            group = QGroupBox(key.replace("_", " ").title())
            gl = QVBoxLayout()
            editor = QPlainTextEdit(content)
            self.editors[key] = editor
            gl.addWidget(editor)
            group.setLayout(gl)
            self.inner_layout.addWidget(group)
        scroll.setWidget(inner)
        layout.addWidget(scroll)

        btn_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Close)
        btn_box.accepted.connect(self.save_prompts)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def save_prompts(self):
        for key, editor in self.editors.items():
            SYSTEM_PROMPTS[key] = editor.toPlainText()
        with open(SYSTEM_PROMPTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(SYSTEM_PROMPTS, f, indent=2)
        QMessageBox.information(self, "Success", "Prompts have been saved.")
        self.accept()


# ────────────────────────────────────────────────────────────────
# Python Script Analysis Worker
# ────────────────────────────────────────────────────────────────
class PythonAnalysisWorker(QThread):
    # UPDATE: Signal now emits the final report and the intermediate steps
    finished_signal = pyqtSignal(str, list)
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)

    def __init__(self, script_content, prompts, parent=None):
        super().__init__(parent)
        self.script_content = script_content
        self.prompts = prompts
        self.client = Client(host=OLLAMA_HOST)

    def _agent_call(self, user_prompt, system_prompt):
        try:
            response = self.client.chat(
                model=MODEL_NAME,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ]
            )
            content = response.get('message', {}).get('content', '').strip()
            return content if is_valid_response(content) else None
        except Exception as e:
            self.log_signal.emit(f"⚠️ LLM call failed: {e}")
            return None

    def run(self):
        self.log_signal.emit("🐍 Starting Python script analysis...")

        try:
            tree = ast.parse(self.script_content)
        except SyntaxError as e:
            error_msg = f"ANALYSIS FAILED: The provided input is not a valid Python script.\n\nError: {e}"
            self.log_signal.emit(f"❌ Syntax Error: {e}")
            self.finished_signal.emit(error_msg, [])  # Emit empty list for thinking log
            return

        all_analyses = []

        self.log_signal.emit("   ► Pass 1: Generating high-level summary...")
        self.progress_signal.emit(10)
        overall_summary = self._agent_call(
            f"Provide a high-level summary of the following Python script:\n\n```python\n{self.script_content}\n```",
            self.prompts['python_overall_summary_prompt']
        )
        if not overall_summary:
            self.log_signal.emit("❌ Failed to generate overall summary. Aborting.")
            self.finished_signal.emit("Analysis failed: Could not generate a high-level summary.", [])
            return
        all_analyses.append(f"### Overall Script Purpose\n\n{overall_summary}")
        self.log_signal.emit("   ✔ Pass 1 complete.")

        components_to_analyze = []
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        if imports:
            components_to_analyze.append(('imports', imports))

        top_level_items = [node for node in tree.body]
        for item in top_level_items:
            if isinstance(item, ast.Assign):
                components_to_analyze.append(('global', item))
            elif isinstance(item, ast.FunctionDef):
                components_to_analyze.append(('function', item))
            elif isinstance(item, ast.ClassDef):
                components_to_analyze.append(('class', item))

        total_components = len(components_to_analyze)
        if total_components == 0:
            self.log_signal.emit("   ► No major components found to analyze.")

        for i, (comp_type, component) in enumerate(components_to_analyze):
            progress = 20 + int(((i + 1) / total_components) * 60) if total_components > 0 else 80
            self.progress_signal.emit(progress)

            context_prompt = f"For context, here is the overall summary of the script:\n'{overall_summary}'\n\n---\n\n"
            explanation = None
            title = ""

            if comp_type == 'imports':
                title = "Imports and Dependencies"
                self.log_signal.emit(f"   ► Analyzing {title}...")
                code_str = "\n".join([ast.unparse(n) for n in component])
                prompt = context_prompt + f"Analyze these imports:\n```python\n{code_str}\n```"
                explanation = self._agent_call(prompt, self.prompts['python_imports_prompt'])

            elif comp_type == 'global':
                try:
                    target_name = ast.unparse(component.targets[0])
                    title = f"Global Definition: `{target_name}`"
                    self.log_signal.emit(f"   ► Analyzing {title}...")
                    code_str = ast.unparse(component)
                    prompt = context_prompt + f"Analyze this global variable definition:\n```python\n{code_str}\n```"
                    explanation = self._agent_call(prompt, self.prompts['python_globals_prompt'])
                except Exception:
                    self.log_signal.emit(f"   ► Skipping complex global assignment...")

            elif comp_type == 'function':
                title = f"Function: `{component.name}`"
                self.log_signal.emit(f"   ► Analyzing {title}...")
                code_str = ast.unparse(component)
                prompt = context_prompt + f"Analyze this function:\n```python\n{code_str}\n```"
                explanation = self._agent_call(prompt, self.prompts['python_function_prompt'])

            elif comp_type == 'class':
                title = f"Class: `{component.name}`"
                self.log_signal.emit(f"   ► Analyzing {title}...")
                code_str = ast.unparse(component)
                prompt = context_prompt + f"Analyze this class and its methods:\n```python\n{code_str}\n```"
                explanation = self._agent_call(prompt, self.prompts['python_class_prompt'])

            if explanation:
                all_analyses.append(f"### {title}\n\n{explanation}")

        self.log_signal.emit("   ✔ All components analyzed.")

        self.log_signal.emit("   ► Final Pass: Consolidating all analyses into a single report...")
        self.progress_signal.emit(90)

        full_analysis_text = "\n\n---\n\n".join(all_analyses)
        final_report = self._agent_call(
            full_analysis_text,
            self.prompts['final_consolidation_prompt']
        )

        if not final_report:
            self.log_signal.emit("⚠️ Final consolidation failed. Using raw analysis as fallback.")
            final_report = full_analysis_text

        self.progress_signal.emit(100)
        self.log_signal.emit("✅ Analysis Complete.")
        # UPDATE: Emit both the final report and the intermediate steps
        self.finished_signal.emit(final_report, all_analyses)


# ────────────────────────────────────────────────────────────────
# Main Window
# ────────────────────────────────────────────────────────────────
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Julia-Python Agent")
        self.resize(950, 950)

        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(30, 30, 47))
        palette.setColor(QPalette.WindowText, QColor(240, 240, 240))
        palette.setColor(QPalette.Base, QColor(45, 45, 68))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, QColor(240, 240, 240))
        palette.setColor(QPalette.Button, QColor(48, 71, 94))
        palette.setColor(QPalette.ButtonText, QColor(240, 240, 240))
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        self.setPalette(palette)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(4)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(8, 8, 8, 8)

        ig = QGroupBox("🐍 Python Script Input")
        il = QVBoxLayout()
        self.input_text = QPlainTextEdit()
        self.input_text.setPlaceholderText("Paste your Python script here…")
        il.addWidget(self.input_text)
        ig.setLayout(il)
        ll.addWidget(ig, 1)  # Give stretch factor

        btn_layout = QHBoxLayout()
        self.send_button = QPushButton("🚀 Analyze Script")
        self.send_button.clicked.connect(self.on_send_clicked)
        btn_layout.addWidget(self.send_button)
        ll.addLayout(btn_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        ll.addWidget(self.progress_bar)
        splitter.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(8, 8, 8, 8)

        vl = QHBoxLayout()
        self.view_log_button = QPushButton("📄 View Log")
        self.view_log_button.clicked.connect(self.show_log_window)
        self.prompts_button = QPushButton("✏️ Edit Prompts")
        self.prompts_button.clicked.connect(self.show_prompts_window)
        vl.addWidget(self.view_log_button)
        vl.addWidget(self.prompts_button)
        rl.addLayout(vl)

        og = QGroupBox("🎯 Final Analysis Report")
        ol = QVBoxLayout()
        self.final_output_box = QTextEdit()
        self.final_output_box.setReadOnly(True)
        ol.addWidget(self.final_output_box)
        og.setLayout(ol)
        rl.addWidget(og, 1)  # Give stretch factor

        next_btn = QPushButton("Next agent Jim >")
        next_btn.clicked.connect(self.on_next_agent)
        rl.addWidget(next_btn)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(splitter)

        self.log_window = LogWindow()
        self.worker = None

    def on_send_clicked(self):
        script_content = self.input_text.toPlainText().strip()
        if not script_content:
            QMessageBox.warning(self, "No Input", "Please paste a Python script to analyze.")
            return

        self.send_button.setEnabled(False)
        self.final_output_box.clear()
        self.progress_bar.setValue(0)
        self.log_window.log_output.clear()
        self.log_window.append_log("🚀 Starting analysis...")
        self.log_window.show()

        self.worker = PythonAnalysisWorker(script_content, SYSTEM_PROMPTS)
        self.worker.log_signal.connect(self.log_window.append_log)
        self.worker.finished_signal.connect(self.on_pipeline_finished)
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.worker.start()

    def on_pipeline_finished(self, final_report, intermediate_analyses):
        self.send_button.setEnabled(True)
        self.final_output_box.setPlainText(final_report)

        ts = int(time.time())

        # 1. Save the final output
        self.log_window.append_log("\n✅ Analysis Complete. Saving files...")
        final_output_data = {
            "timestamp": ts,
            "agent": "Julia-Python",
            "report": final_report
        }
        final_output_path = os.path.join(FINAL_OUTPUTS_DIR, f"report_{ts}.json")
        try:
            with open(final_output_path, "w", encoding='utf-8') as f:
                json.dump(final_output_data, f, indent=4)
            self.log_window.append_log(f"💾 Saved final report to {final_output_path}")
        except Exception as e:
            self.log_window.append_log(f"⚠️ Failed to save final report: {e}")

        # 2. Save the thinking log
        thinking_data = {
            "timestamp": ts,
            "agent": "Julia-Python",
            "component_analyses": intermediate_analyses
        }
        thinking_log_path = os.path.join(THINKING_LOGS_DIR, f"thinking_{ts}.json")
        try:
            with open(thinking_log_path, "w", encoding='utf-8') as f:
                json.dump(thinking_data, f, indent=4)
            self.log_window.append_log(f"💾 Saved 'thinking' log to {thinking_log_path}")
        except Exception as e:
            self.log_window.append_log(f"⚠️ Failed to save thinking log: {e}")

        # 3. Save the full runtime log
        full_log_text = self.log_window.log_output.toPlainText()
        runtime_log_path = os.path.join(LOGS_DIR, f"runtime_log_{ts}.txt")
        try:
            with open(runtime_log_path, 'w', encoding='utf-8') as f:
                f.write(full_log_text)
            self.log_window.append_log(f"💾 Saved full runtime log to {runtime_log_path}\n")
        except Exception as e:
            self.log_window.append_log(f"⚠️ Failed to save runtime log: {e}")

    def on_next_agent(self):
        final_text = self.final_output_box.toPlainText().strip()
        if not final_text:
            QMessageBox.warning(self, "No Output", "There is no final analysis from Julia-Python to send.")
            return
        dialog = JimDialog(final_text, parent=self)
        dialog.exec_()

    def show_log_window(self):
        self.log_window.show()

    def show_prompts_window(self):
        dialog = PromptsWindow(self)
        dialog.exec_()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(30, 30, 47))
    dark_palette.setColor(QPalette.WindowText, QColor(240, 240, 240))
    dark_palette.setColor(QPalette.Base, QColor(45, 45, 68))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, QColor(240, 240, 240))
    dark_palette.setColor(QPalette.Button, QColor(48, 71, 94))
    dark_palette.setColor(QPalette.ButtonText, QColor(240, 240, 240))
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(dark_palette)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
