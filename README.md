# Julia_In: Multi-Agent LLM Pipeline System

> **A modular, extensible framework for intelligent document analysis through layered summarization and specialized agent workflows**

## 🎯 Overview

Julia_In is a sophisticated multi-agent system that processes complex documents through **layered summarization**, **parallel agent processing**, and **specialized extraction workflows**. Built on Ollama LLM (`llama3.2:3b`), it breaks down large documents into manageable chunks, processes them in parallel across multiple AI agents, and synthesizes insights through multiple refinement passes.

**Key Innovation**: Rather than single-pass summarization, Julia uses a **multi-layer, multi-agent architecture** that progressively refines understanding, extracts key concepts, and delegates specialized analysis to dedicated sub-agents.

---

## 📁 System Architecture

### Core Files

#### 🔷 **Julia_In_2.py** (MAIN ENTRY POINT)
**Status**: ⭐ Primary production version

This is the **main application** you should use. It provides:

- **Two-Pane Dark-Themed GUI**
  - Left: Input area, parameters, prompt editor
  - Right: Complexity scores, final output, log viewer
  
- **Multi-Layer Summarization Pipeline**
  - Configurable parallel agents (1-20 concurrent)
  - Configurable summary layers (1-10 depth)
  - Two-pass system: First Pass → Refinement Pass
  
- **Intelligent Features**
  - **Subject Extraction**: Identifies core theme from document
  - **Concept Extraction**: Extracts key insights per layer
  - **Complexity Scoring**: Rates each layer's complexity (1-10)
  - **Code-Aware Chunking**: Detects and properly handles code blocks
  - **Batch Summarization**: Combines multiple agent outputs intelligently
  
- **JIM Agent Integration**
  - **"Next agent Jim >"** button passes final output to specialized analysis
  - Enables seamless hand-off to deep-dive extraction workflows

**When to Use**: Start here for any document analysis, summarization, or information extraction task.

---

#### 🦊 **Julia_Jim_Flow_1.py** (SPECIALIZED AGENT)
**Status**: 🔌 Modular plugin agent

JIM (Julia Intelligent Module) is a **specialized extraction agent** designed to perform **structured, field-based analysis** on processed documents.

**What JIM Does**:
1. Takes the entire output log from Julia_In_2
2. Splits into 1,000-character chunks
3. Extracts **12 specific fields** per chunk in parallel:
   - Key points
   - Key people
   - Key ideas
   - Economic viability
   - Questions to explore
   - Potential problems
   - Complementary new ideas
   - Tone analysis
   - Innovation identification
   - Research requirements
   - Persona profiles
   - Stakeholder types

4. **Synthesizes findings** per field across all chunks
5. Outputs structured JSON with all 12 fields

**When to Use**: After Julia_In_2 produces a summary, use JIM to extract detailed, structured insights for project planning, stakeholder analysis, or research scoping.

**Architecture Note**: JIM demonstrates the **extensibility** of Julia_In. You can create similar specialized agents for:
- Legal document analysis
- Medical report extraction
- Financial data synthesis
- Technical specification parsing

---

#### 🔗 **Julia_Jim_Flow_2.py** (INTEGRATED VERSION)
**Status**: 🧪 Experimental unified build

This version **embeds JIM directly** into the main Julia GUI as a modal dialog, accessed via a **"🦊 JIM Agent"** button in the bottom-right.

**Advantages**:
- Single application launch
- Seamless workflow (Julia → JIM in one window)
- Shared styling and configuration

**Trade-offs**:
- Less modular than Julia_In_2 + Julia_Jim_Flow_1
- Harder to customize JIM independently

---

### Legacy/Alternative Versions

#### 📄 **Julia_In_1.py**
Early version with basic layered summarization. Missing subject/concept extraction and JIM integration.

#### 📄 **Julia_In_2_legacy.py**
Backup of Julia_In_2 before major refactoring.

#### 📄 **Julia_In_3.py**
Experimental version exploring alternative chunking strategies.

#### 📄 **Julia_In_4.py**
Advanced version with additional refinement logic (may be more resource-intensive).

#### 📄 **Julia_Input_Box_Legacy.py**
Original input handling implementation before GUI redesign.

#### 📄 **Julia_Python.py**
Minimal CLI version for scripting/automation (no GUI).

---

## 🚀 Quick Start

### Prerequisites
```bash
# Install Python dependencies
pip install PyQt5 ollama

# Ensure Ollama is running with llama3.2:3b
ollama pull llama3.2:3b
ollama serve
```

### Basic Usage

**1. Run Julia for Document Summarization**
```bash
python3 Julia_In_2.py
```
- Paste your document in the input area
- Configure parallel agents (5 recommended) and layers (2 recommended)
- Click **"✉️ Send"**
- View live log via **"📄 View Log"**
- Final output appears in right pane with complexity scores

**2. Run JIM for Deep Analysis**
After Julia completes:
- Click **"Next agent Jim >"**
- JIM dialog opens with full log pre-loaded
- Set parallel calls (3 recommended) and summarization layers (1-2)
- Click **"🚀 Run JIM"**
- Structured JSON output appears with all 12 fields

**3. Standalone JIM**
```bash
python3 Julia_Jim_Flow_1.py /path/to/input.txt
```

---

## ⚙️ Configuration

### System Prompts
Customize AI behavior via **"All System Prompts 📜"** button:

- `subject_extraction_prompt`: How to identify core theme
- `concept_extraction_prompt`: How to extract key insights
- `layer_system_prompt`: Instructions for each summarization agent
- `batch_summary_prompt`: How to combine multiple agent outputs
- `final_system_prompt`: Instructions for final consolidation

Prompts are saved in `system_prompts.json` and persist across sessions.

### JIM Field Prompts
Customize JIM's extraction behavior by editing `jim_prompts.json`:
```json
{
  "Key points": "Analyze the following text and provide...",
  "Economic viability": "Perform a thorough analysis of...",
  ...
}
```

---

## 🧠 How It Works (High-Level)

### Julia's Multi-Agent Pipeline

```
User Input
    ↓
[Subject Extraction] ← Identifies core theme
    ↓
[First Pass]
    ├─ Split into chunks (code-aware)
    ├─ Layer 1: N parallel agents process chunks
    │   └─ Batch summary combines outputs
    ├─ Layer 2: N parallel agents process summaries
    │   └─ Batch summary combines outputs
    └─ Consolidation: Final agent synthesizes
    ↓
[Refinement Pass]
    ├─ Same multi-layer process on intermediate output
    └─ Produces polished final summary
    ↓
[Complexity Scoring] ← Rates each layer 1-10
    ↓
Final Output + Scores
```

### JIM's Specialized Extraction

```
Julia's Log
    ↓
[Chunk into 1,000-char pieces]
    ↓
[Parallel Field Extraction]
    ├─ Agent 1: Key points
    ├─ Agent 2: Key people
    ├─ Agent 3: Key ideas
    ├─ ... (12 fields total)
    └─ All in parallel per chunk
    ↓
[Synthesis Stage]
    ├─ Consolidate "Key points" from all chunks
    ├─ Consolidate "Key people" from all chunks
    └─ ... (12 consolidated fields)
    ↓
[JSON Assembly]
    └─ Structured output with all fields
```

---

## 🎨 Why This Architecture?

### Problem: Traditional Summarization Fails on Complex Documents
- Single-pass summarization loses nuance
- Large documents exceed context windows
- Important details get buried in noise

### Solution: Multi-Agent, Multi-Layer Processing
1. **Chunking**: Break document into manageable pieces
2. **Parallel Processing**: Multiple agents work simultaneously
3. **Layered Refinement**: Progressive distillation over N layers
4. **Concept Tracking**: Extract and carry forward key insights
5. **Specialized Delegation**: Hand off to domain-specific agents (JIM)

### Result: High-Fidelity, Structured Understanding
- Captures both details and big picture
- Scales to documents of any size
- Produces structured, actionable outputs
- Extensible to new analysis types

---

## 🔮 Future Extensions

This framework is designed to be **extended with new specialized agents**:

### Example: Legal Document Agent
```python
# Julia → Legal Agent
fields = [
    "Parties involved",
    "Key obligations",
    "Liability clauses",
    "Termination conditions",
    "Risk analysis"
]
```

### Example: Medical Report Agent
```python
# Julia → Medical Agent
fields = [
    "Diagnosis",
    "Symptoms",
    "Treatment plan",
    "Medications",
    "Follow-up requirements"
]
```

### Creating Your Own Agent
1. Copy `Julia_Jim_Flow_1.py`
2. Modify `JIM_FIELDS` list
3. Update field prompts in `JIM_DEFAULT_PROMPTS`
4. Customize synthesis logic if needed
5. Integrate into Julia via button handler

---

## 📊 Output Examples

### Julia Output
```
The document discusses a multi-agent AI system designed for 
layered summarization. Key innovations include parallel processing 
across 5 concurrent agents, progressive refinement over 2 layers, 
and intelligent code-aware chunking. The system achieves high-
fidelity understanding by extracting core concepts at each layer 
and synthesizing them through a final consolidation pass.
```

### JIM Output (JSON)
```json
{
  "Key points": "Multi-agent architecture, layered processing...",
  "Key people": "No specific individuals mentioned...",
  "Key ideas": "Progressive refinement, concept extraction...",
  "Economic viability": "Low operational cost using local LLM...",
  "Questions": "How does performance scale beyond 10 layers?...",
  "Problems that could arise": "Context window limitations...",
  "New ideas that complement key ideas": "Add dynamic layer...",
  "Tone": "Technical, analytical, innovative...",
  "What is the innovation": "Multi-layer concept tracking...",
  "Research that needs to be done": "Benchmark against GPT-4...",
  "The Personas profiles": "AI researchers, data scientists...",
  "The types of people relating to this": "ML engineers..."
}
```

---

## 🛠 Technical Stack

- **UI Framework**: PyQt5 (dark-themed, two-pane layout)
- **LLM Backend**: Ollama (llama3.2:3b)
- **Concurrency**: Python `concurrent.futures.ThreadPoolExecutor`
- **State Management**: JSON-based configuration files
- **Logging**: Real-time log window with auto-scroll

---

## 📝 Files Generated

### Runtime Outputs
- `user_outputs/output_<timestamp>.json` - Julia's final outputs
- `Jim/jim_output_<timestamp>.json` - JIM's structured JSON
- `Jim/jim_thinking_<timestamp>.json` - JIM's intermediate extractions
- `Jim/jim_log_<timestamp>.txt` - JIM's full processing log

### Configuration
- `system_prompts.json` - Julia's system prompts
- `jim_prompts.json` - JIM's field extraction prompts

---

## 🤝 Contributing

This is a **framework for building intelligent document processing pipelines**. To extend:

1. **Fork for your use case** (legal, medical, financial, etc.)
2. **Create specialized agents** following JIM's pattern
3. **Customize prompts** for your domain
4. **Share your agents** as modular plugins

---

## 📖 Further Reading

See **[ARCHITECTURE.md](./ARCHITECTURE.md)** for:
- Deep dive into layering and chunking algorithms
- Parallel agent coordination strategies
- Current limitations and improvement proposals
- Roadmap for "active reading" vs "passive scanning"

---

## 🏷️ Version History

- **v1legacy**: Initial multi-agent pipeline with basic layered summarization
- **v2**: Added JIM integration, subject/concept extraction, complexity scoring
- **v3**: Code-aware chunking, batch summarization, two-pass refinement
- **v4**: Experimental refinement enhancements
- **Current**: Production-ready multi-agent framework with modular agent support

---

## ⚡ Performance Notes

**Recommended Configuration**:
- **Parallel Agents**: 5 (balance between speed and resource usage)
- **Summary Layers**: 2 (diminishing returns beyond 3)
- **JIM Parallel Calls**: 3 (optimal for 12-field extraction)
- **JIM Layers**: 1 (sufficient for synthesis)

**Typical Processing Time**:
- 10-page document: 2-3 minutes (Julia)
- JIM deep analysis: 3-5 minutes
- Total: ~5-8 minutes for complete structured analysis

---

## 📧 Support

For questions about extending this framework or creating custom agents, refer to the code comments in `Julia_In_2.py` and `Julia_Jim_Flow_1.py` for detailed inline documentation.

---

**Ready to process complex documents intelligently? Run `python3 Julia_In_2.py` and start building your multi-agent pipeline!** 🚀

