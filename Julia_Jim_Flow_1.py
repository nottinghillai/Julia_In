#!/usr/bin/env python3
"""
Standalone “JIM Agent” PyQt5 GUI for analyzing a full log. Expects a single
command-line argument: the path to a text file containing the entire log.
It splits that log into 1,000-character chunks, extracts twelve fields from each
chunk via Ollama LLM calls in parallel, then runs multi-layer summarization,
and finally asks the LLM for a JSON with all fields.

Usage:
    python3 Julia_Jim_Flow_1.py /path/to/jim_input.txt

Model: llama3.2:3b
"""

import sys
import os
import json
import re
import time
import threading
import concurrent.futures
from PyQt5.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QPushButton, QPlainTextEdit, QTextEdit, QSpinBox,
    QFormLayout, QProgressBar, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPalette, QColor
from ollama import Client

# ────────────────────────────────────────────────────────────────
# Configuration & Defaults for JIM
# ────────────────────────────────────────────────────────────────
OLLAMA_HOST = 'http://localhost:11434'
MODEL_NAME = 'llama3.2:3b'
JIM_PROMPTS_FILE = 'jim_prompts.json'

# The fields JIM must answer per chunk
JIM_FIELDS = [
    "Key points", "Key people", "Key ideas", "Economic viability", "Questions",
    "Problems that could arise", "New ideas that complement key ideas", "Tone",
    "What is the innovation", "Research that needs to be done",
    "The Personas profiles that relate to the input", "The types of people relating to this"
]

JIM_DEFAULT_PROMPTS = {
    "Key points": "Analyze the following text and provide a detailed, bulleted list of the most important takeaways and conclusions. Focus on actionable information and critical insights.",
    "Key people": "Identify all key individuals or groups mentioned in the text. For each, provide their name, their role or title, and a brief analysis of their significance or contribution to the topic.",
    "Key ideas": "Deconstruct the central arguments of the text. Provide a detailed breakdown of the core ideas, concepts, or conceptual pillars. Explain the reasoning behind each idea.",
    "Economic viability": "Perform a thorough analysis of the economic viability of the concepts in the text. Discuss potential revenue streams, cost factors, market fit, and any identifiable financial risks or opportunities.",
    "Questions": "Based on the text, formulate a list of insightful, open-ended questions that challenge the text's assumptions, explore its implications, or probe for deeper understanding. Avoid simple clarification questions.",
    "Problems that could arise": "Identify and list all potential problems, challenges, or risks associated with the ideas in the text. Consider implementation hurdles, unforeseen negative consequences, and any potential logical fallacies.",
    "New ideas that complement key ideas": "Brainstorm a list of actionable and innovative suggestions that build upon or complement the core ideas of the text. These new ideas should extend the original concepts in a meaningful way.",
    "Tone": "Analyze and describe the author's tone (e.g., analytical, optimistic, critical, speculative). Provide specific examples or word choices from the text to support your analysis.",
    "What is the innovation": "Pinpoint and describe the single most novel, disruptive, or innovative concept presented in the text. Explain what makes it a breakthrough compared to existing ideas or solutions.",
    "Research that needs to be done": "Based on the information and claims in the text, outline a list of specific, actionable research steps or experiments that should be conducted to validate, test, or expand upon the core ideas.",
    "The Personas profiles that relate to the input": "Develop detailed user persona profiles for the individuals who would use or benefit from the product/service described. Include their likely goals, pain points, motivations, and technical aptitude.",
    "The types of people relating to this": "Identify all stakeholders, groups, or communities that would be affected by or have a vested interest in the topics discussed. For each group, analyze their likely perspective, potential interest, or concerns."
}

if not os.path.exists(JIM_PROMPTS_FILE):
    with open(JIM_PROMPTS_FILE, 'w', encoding='utf-8') as pf:
        json.dump(JIM_DEFAULT_PROMPTS, pf, indent=2)

with open(JIM_PROMPTS_FILE, 'r', encoding='utf-8') as pf:
    JIM_PROMPTS = json.load(pf)

missing_fields = [f for f in JIM_FIELDS if f not in JIM_PROMPTS]
if missing_fields:
    for field in missing_fields:
        JIM_PROMPTS[field] = JIM_DEFAULT_PROMPTS.get(field, f"Extract the '{field}' from the following text.")
    with open(JIM_PROMPTS_FILE, 'w', encoding='utf-8') as pf:
        json.dump(JIM_PROMPTS, pf, indent=2)


# ────────────────────────────────────────────────────────────────
# Helper Function to Validate LLM Responses
# ────────────────────────────────────────────────────────────────
def is_valid_response(text: str) -> bool:
    """Checks if a response from the LLM is a valid summary or a refusal."""
    if not text or not text.strip():
        return False
    lower_text = text.lower()
    refusal_phrases = [
        "i can't", "i cannot", "unable to", "i'm not able", "i am not able",
        "ready to assist", "ready to help", "ready to summarize",
        "provide the text", "provide the chunk", "what is the chunk",
        "you haven't provided", "please provide", "as an ai", "i am an ai",
        "misunderstanding in my previous response"
    ]
    if any(phrase in lower_text for phrase in refusal_phrases):
        return False
    if len(text.split()) < 4 and "?" in text:
        return False
    return True


# ────────────────────────────────────────────────────────────────
# Worker Thread
# ────────────────────────────────────────────────────────────────
class JimWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(dict, dict)
    complexity_signal = pyqtSignal(int)

    def __init__(self, full_log_text, concurrency, num_layers, prompts, parent=None):
        super().__init__(parent)
        self.full_log_text = full_log_text
        self.concurrency = concurrency
        self.num_layers = num_layers
        self.prompts = {
            field: prompts.get(field, f"Extract the '{field}' from the following text.")
            for field in JIM_FIELDS
        }
        self.client = Client(host=OLLAMA_HOST)
        self.lock = threading.Lock()

    def chunk_by_chars(self, text: str, limit: int):
        chunks, idx, length = [], 0, len(text)
        while idx < length:
            if length - idx <= limit:
                chunks.append(text[idx:].strip())
                break
            sp = idx + limit
            while sp > idx and not text[sp].isspace():
                sp -= 1
            if sp == idx:
                sp = idx + limit
            chunks.append(text[idx:sp].strip())
            idx = sp
        return [c for c in chunks if c]

    def agent_call(self, prompt: str, system_prompt: str = None, json_format: bool = False):
        """Makes a call to the Ollama client, with an option to force JSON output."""
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': prompt})

        params = {
            'model': MODEL_NAME,
            'messages': messages
        }
        if json_format:
            params['format'] = 'json'

        with self.lock:
            resp = self.client.chat(**params)

        msg = resp.get('message', {})
        return msg.get('content', '').strip()

    def score_text(self, text: str):
        """Scores the text complexity using a dedicated system prompt for reliable output."""
        system_prompt = (
            "You are a text complexity analyst. Your sole task is to evaluate the provided text on a scale of 1 to 10 "
            "and respond with only a single integer. Do not provide any explanation, preamble, or summary. "
            "The scale is defined as follows:\n"
            "1: Very simple, basic vocabulary, short sentences.\n"
            "5: Average complexity, standard prose, like a news article.\n"
            "10: Very complex, dense, highly technical, or academic language with specialized jargon.\n"
            "Your response MUST be a single number (e.g., '7')."
        )
        user_prompt = f"Text to analyze:\n\n---\n\n{text}"

        try:
            res = self.agent_call(user_prompt, system_prompt=system_prompt)
            # Find the first number in the response. This is robust against accidental extra text.
            match = re.search(r'\d+', res)
            if match:
                s = max(1, min(10, int(match.group())))
            else:
                s = 5  # Fallback if no number is found
        except (ValueError, AttributeError):
            s = 5 # Fallback if parsing fails
        self.log_signal.emit(f"🏷 Complexity: {s}/10\n")
        return s

    def run(self):
        # STAGE 1: Initial Field Extraction from Chunks
        self.log_signal.emit("🦊 Stage 1: Splitting log and extracting initial fields…\n")
        chunks = self.chunk_by_chars(self.full_log_text, 1000)
        total_initial_tasks = len(chunks) * len(JIM_FIELDS)
        tasks_done = 0

        field_answers = {field: [] for field in JIM_FIELDS}

        for i, ch in enumerate(chunks, 1):
            self.log_signal.emit(f"🔹 Analyzing Chunk {i}/{len(chunks)}…\n")

            def fetch_field(field):
                prompt = f"{self.prompts[field]}\n\n{ch}"
                ans = self.agent_call(prompt)
                if not is_valid_response(ans):
                    self.log_signal.emit(f"• [{field}] → ⚠️ REFUSED or invalid, discarding.\n")
                    return field, None
                self.log_signal.emit(f"• [{field}] → {ans}\n")
                return field, ans

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency) as ex:
                futures = {ex.submit(fetch_field, f): f for f in JIM_FIELDS}
                for fut in concurrent.futures.as_completed(futures):
                    field, ans = fut.result()
                    if ans:
                        field_answers[field].append(ans)
                    tasks_done += 1
                    if total_initial_tasks > 0:
                        self.progress_signal.emit(int(tasks_done / total_initial_tasks * 70))

        self.log_signal.emit("\n✅ Stage 1 Complete: Initial field extraction finished.\n")

        # STAGE 2: Synthesize the findings for each field individually
        self.log_signal.emit("🧠 Stage 2: Synthesizing findings for each field…\n")
        synthesized_answers = {}

        total_synthesis_tasks = len([f for f in JIM_FIELDS if field_answers.get(f)])
        synthesis_tasks_done = 0

        for field in JIM_FIELDS:
            answers_list = field_answers.get(field)
            if not answers_list:
                synthesized_answers[field] = ""  # Ensure key exists
                continue

            self.log_signal.emit(f"🔹 Synthesizing '{field}'…\n")

            synthesis_context = "\n\n---\n\n".join(answers_list)
            synthesis_prompt = (
                f"You are a data synthesizer. Review the following observations related to '{field}' and create a single, consolidated, and comprehensive summary. "
                f"Focus on combining related points and removing redundancy. Your response should be a final, polished analysis for this single category.\n\n"
                f"OBSERVATIONS FOR '{field}':\n{synthesis_context}"
            )

            synthesized_ans = self.agent_call(synthesis_prompt)
            if is_valid_response(synthesized_ans):
                self.log_signal.emit(f"✓ Synthesized '{field}': {synthesized_ans}\n")
                synthesized_answers[field] = synthesized_ans
            else:
                self.log_signal.emit(f"⚠️ Synthesis for '{field}' failed, using raw data as fallback.\n")
                synthesized_answers[field] = synthesis_context  # Fallback to raw data

            synthesis_tasks_done += 1
            if total_synthesis_tasks > 0:
                self.progress_signal.emit(70 + int(synthesis_tasks_done / total_synthesis_tasks * 30))

        self.log_signal.emit("\n✅ Stage 2 Complete: Synthesis finished.\n")

        # STAGE 3: Assemble the final JSON object (no LLM call needed)
        self.log_signal.emit("📝 Stage 3: Assembling final JSON output…\n")

        final_json_obj = {field: synthesized_answers.get(field, "") for field in JIM_FIELDS}
        thinking_output_obj = field_answers

        self.progress_signal.emit(100)
        self.finished_signal.emit(final_json_obj, thinking_output_obj)

        # STAGE 4: Final Complexity Score (Separate LLM Call)
        self.log_signal.emit("📊 Stage 4: Calculating final complexity score…\n")
        final_output_string = json.dumps(final_json_obj, indent=2)
        if is_valid_response(final_output_string):
            score = self.score_text(final_output_string)
            self.complexity_signal.emit(score)


# ────────────────────────────────────────────────────────────────
# Main Dialog
# ────────────────────────────────────────────────────────────────
class JimDialog(QDialog):
    def __init__(self, full_log_text, parent=None):
        super().__init__(parent)
        self.setWindowTitle("JIM Agent")
        self.resize(900, 900)
        self.setModal(True)

        pal = QPalette()
        pal.setColor(QPalette.Window, QColor("#1e1e2f"))
        pal.setColor(QPalette.WindowText, QColor("#f0f0f0"))
        pal.setColor(QPalette.Base, QColor("#2d2d44"))
        pal.setColor(QPalette.Text, QColor("#f0f0f0"))
        pal.setColor(QPalette.Button, QColor("#252537"))
        pal.setColor(QPalette.ButtonText, QColor("#f0f0f0"))
        self.setPalette(pal)

        main = QVBoxLayout(self)
        main.setSpacing(8)
        main.setContentsMargins(8, 8, 8, 8)

        self.input_count = QLabel()
        self.input_count.setStyleSheet("color:#f0f0f0")
        self.update_input_count(full_log_text)
        main.addWidget(self.input_count)
        gi = QGroupBox("🗂 Input Log")
        gi.setStyleSheet("QGroupBox{color:#f0f0f0}")
        vi = QVBoxLayout(gi)
        self.input_area = QPlainTextEdit()
        self.input_area.setPlainText(full_log_text)
        self.input_area.setReadOnly(True)
        self.input_area.setStyleSheet("background:#2d2d44;color:#f0f0f0")
        vi.addWidget(self.input_area)
        main.addWidget(gi, stretch=2)
        be = QGroupBox("✏️ Field Prompts")
        be.setStyleSheet("QGroupBox{color:#f0f0f0}")
        se = QVBoxLayout(be)
        self.prompts_area = QPlainTextEdit()
        self.prompts_area.setPlainText(json.dumps(JIM_PROMPTS, indent=2))
        self.prompts_area.setStyleSheet("background:#2d2d44;color:#f0f0f0")
        se.addWidget(self.prompts_area)
        btn_save = QPushButton("Save Prompts")
        btn_save.clicked.connect(self.save_prompts)
        se.addWidget(btn_save)
        main.addWidget(be)
        gp = QGroupBox("⚙️ Parameters")
        gp.setStyleSheet("QGroupBox{color:#f0f0f0}")
        form = QFormLayout(gp)
        self.spin_c = QSpinBox();
        self.spin_c.setRange(1, 10);
        self.spin_c.setValue(3)
        self.spin_l = QSpinBox();
        self.spin_l.setRange(1, 5);
        self.spin_l.setValue(1)
        form.addRow("Parallel Calls:", self.spin_c)
        form.addRow("Summ Layers:", self.spin_l)
        main.addWidget(gp)
        hr = QHBoxLayout()
        self.btn_run = QPushButton("🚀 Run JIM")
        self.btn_run.setStyleSheet("background:#30475e;color:#f0f0f0")
        self.btn_run.clicked.connect(self.on_run)
        self.pb = QProgressBar()
        self.pb.setStyleSheet("background:#2d2d44;color:#f0f0f0")
        hr.addWidget(self.btn_run)
        hr.addWidget(self.pb)
        main.addLayout(hr)
        gl = QGroupBox("📑 JIM Log")
        gl.setStyleSheet("QGroupBox{color:#f0f0f0}")
        vl = QVBoxLayout(gl)
        self.log_area = QPlainTextEdit();
        self.log_area.setReadOnly(True)
        self.log_area.setStyleSheet("background:#2d2d44;color:#f0f0f0")
        vl.addWidget(self.log_area)
        main.addWidget(gl, stretch=2)
        go = QGroupBox("🎯 Final JSON")
        go.setStyleSheet("QGroupBox{color:#f0f0f0}")
        vo = QVBoxLayout(go)
        self.out_area = QTextEdit();
        self.out_area.setReadOnly(True)
        self.out_area.setStyleSheet("background:#2d2d44;color:#f0f0f0")
        vo.addWidget(self.out_area)
        self.output_count = QLabel();
        self.output_count.setStyleSheet("color:#f0f0f0")
        vo.addWidget(self.output_count)
        main.addWidget(go, stretch=2)
        gs = QGroupBox("📊 Complexity Score")
        gs.setStyleSheet("QGroupBox{color:#f0f0f0}")
        hs = QHBoxLayout(gs)
        self.score_boxes = []
        for _ in range(10):
            lb = QLabel();
            lb.setFixedSize(QSize(15, 15))
            lb.setStyleSheet("background:#444466;border:1px solid #000")
            hs.addWidget(lb)
            self.score_boxes.append(lb)
        hs.addStretch()
        main.addWidget(gs)
        self.worker = None

    def save_prompts(self):
        try:
            p = json.loads(self.prompts_area.toPlainText())
            with open(JIM_PROMPTS_FILE, 'w', encoding='utf-8') as pf:
                json.dump(p, pf, indent=2)
            QMessageBox.information(self, "Saved", "Prompts updated.")
            global JIM_PROMPTS
            JIM_PROMPTS = p
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

    def update_input_count(self, text):
        cnt = len(text.split()) if text else 0
        self.input_count.setText(f"Input Words: {cnt}")

    def update_output_count(self, text):
        cnt = len(text.split()) if text else 0
        self.output_count.setText(f"Output Words: {cnt}")

    def append_jim_log(self, txt):
        self.log_area.appendPlainText(txt)
        cur = self.log_area.textCursor()
        cur.movePosition(cur.End)
        self.log_area.setTextCursor(cur)

    def show_score(self, score):
        for i, lb in enumerate(self.score_boxes, start=1):
            color = "#39A000" if i <= score else "#444466"
            lb.setStyleSheet(f"background:{color};border:1px solid #000")

    def on_run(self):
        text = self.input_area.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "No Log", "No input log provided.")
            return
        self.btn_run.setEnabled(False)
        self.spin_c.setEnabled(False)
        self.spin_l.setEnabled(False)
        self.log_area.clear()
        self.out_area.clear()
        self.pb.setValue(0)
        self.show_score(0) # Reset score display
        conc = self.spin_c.value()
        layers = self.spin_l.value()
        self.worker = JimWorker(text, conc, layers, JIM_PROMPTS)
        self.worker.log_signal.connect(self.append_jim_log)
        self.worker.progress_signal.connect(self.pb.setValue)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.complexity_signal.connect(self.show_score)
        self.worker.start()

    def on_jim_send(self):
        self.on_run()

    def on_finished(self, final_obj, thinking_obj):
        self.btn_run.setEnabled(True)
        self.spin_c.setEnabled(True)
        self.spin_l.setEnabled(True)

        final_text = json.dumps(final_obj, indent=2)
        self.out_area.setPlainText(final_text)
        self.update_output_count(final_text)
        self.append_jim_log("✅ Processing complete.\n")

        od = os.path.join(os.getcwd(), "Jim")
        os.makedirs(od, exist_ok=True)
        ts = int(time.time())

        # Save the final JSON output
        final_pth = os.path.join(od, f"jim_output_{ts}.json")
        with open(final_pth, 'w', encoding='utf-8') as f:
            json.dump(final_obj, f, indent=2)
        self.append_jim_log(f"💾 Saved final output to {final_pth}\n")

        # Save the "thinking" (intermediate extractions)
        thinking_pth = os.path.join(od, f"jim_thinking_{ts}.json")
        with open(thinking_pth, 'w', encoding='utf-8') as f:
            json.dump(thinking_obj, f, indent=2)
        self.append_jim_log(f"💾 Saved 'thinking' log to {thinking_pth}\n")

        # Save the full raw log
        full_log_text = self.log_area.toPlainText()
        log_pth = os.path.join(od, f"jim_log_{ts}.txt")
        with open(log_pth, 'w', encoding='utf-8') as f:
            f.write(full_log_text)
        self.append_jim_log(f"💾 Saved full raw log to {log_pth}\n")


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 Julia_Jim_Flow_1.py /path/to/jim_input.txt")
        sys.exit(1)
    inp = sys.argv[1]
    if not os.path.exists(inp):
        print(f"Error: file not found: {inp}")
        sys.exit(1)
    with open(inp, 'r', encoding='utf-8') as f:
        text = f.read()
    app = QApplication(sys.argv)
    dlg = JimDialog(text)
    dlg.exec_()


if __name__ == "__main__":
    main()