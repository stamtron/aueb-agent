# AUEB Agent parallel Architecture

An advanced, agentic AI system implementing the **Parallel Research** pattern using Google's Agent Development Kit (ADK) and local Ollama models.

## Executive Summary

The AUEB Agent is a sophisticated multi-level AI system designed to handle complex research queries by leveraging a "Panel of Experts." It uses a root orchestrator to manage conversations and delegates deep inquiries to 7 distinct, specialized worker agents running concurrently via local Ollama models. A final verifier agent then synthesizes these 7 perspectives—fact-checking against the web if necessary—into a single, comprehensive, and well-reasoned response.

**Origins & Inspiration**: This project was developed specifically for students in the **MSc in Data Science** course at the **Athens University of Economics and Business (AUEB)**. Inspired by Andrej Karpathy's "LLM Council" concept, the architecture uses Ollama to run robust open-source models completely locally. This allows students to experiment with advanced multi-agent workflows and parallel generation techniques without the barrier of paying for expensive API keys.## Architecture

This project implements a multi-level agent architecture designed for deep research and verification:

1.  **Root Orchestrator**:
    *   The entry point for all user interactions.
    *   Handles casual conversation (greetings, chit-chat) directly.
    *   Delegates complex inquiries to the **Research Team**.

2.  **Research Team (Parallel System)**:
    *   **7 Parallel Workers**: A diverse panel of expert models (Logic, Coding, Creative, Vision, General) that analyze the query concurrently.
    *   **Models**: The system is configured to use specialized local models (via Ollama):
        *   `gpt-oss:20b-cloud` (General)
        *   `deepseek-v3.1:671b-cloud` (Reasoning)
        *   `qwen3-coder:480b-cloud` (Coding)
        *   `qwen3-vl:235b-cloud` (Multimodal)
        *   `minimax-m2:cloud` (Creative)
        *   `glm-4.6:cloud` (Bilingual/Academic)
        *   `gpt-oss:120b` (Large Scale)

3.  **Verifier Agent**:
    *   Synthesizes the 7 distinct responses into a single, comprehensive report.
    *   Equipped with **DuckDuckGo Search** to verifying conflicting facts or hallucinated data.

## Prerequisites

*   **Python 3.10+**
*   **[uv](https://docs.astral.sh/uv/)**: Fast Python package manager.
*   **[Ollama](https://ollama.com/)**: Running locally/remotely to serve the LLMs.

## Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/aueb-agent.git
    cd aueb-agent
    ```

2.  **Install dependencies**:
    ```bash
    make install
    ```

3.  **Configure Ollama Models**:
    Since the architecture asks for specific (and large) model IDs, we assume you have models mapped to these names. Run the provided helper script to alias your existing local models (e.g., `llama3`, `mistral`) to the architecture's required names:
    ```bash
    chmod +x setup_ollama_models.sh
    ./setup_ollama_models.sh
    ```
    *(Note: Edit this script first if you want to map different source models)*

## Running the Agent

### Web Interface (Playground)
Launch the full ADK Web UI to chat with the Root Orchestrator:
```bash
make playground
```
*   Access at: `http://localhost:8801` (Port changed to 8801)
*   Say "Hello" to test the Orchestrator.
*   Ask "What is the importance of attention in LLMs?" to trigger the full parallel research team.

### CLI Verification Script
To test the pipeline programmatically without the UI:
```bash
uv run verify_parallel.py
```

## Key Features & Implementations

*   **Ollama Compatibility Layer**: Includes a custom `OllamaLiteLlm` wrapper (`app/ollama_fix.py`) that transparently fixes JSON payload compatibility issues between LiteLLM and current Ollama API versions.
*   **Parallel Execution**: Uses `ParallelAgent` to run multiple LLM calls concurrently, significantly speeding up the "Panel of Experts" approach.
*   **Tool Integration**: Wraps the entire multi-agent system as a single tool (`parallel_verifier_system`) callable by the Orchestrator.

## Project Structure

```
aueb-agent/
├── app/
│   ├── agent.py            # Root Orchestrator definition
│   ├── agent_parallel.py   # Parallel Workers & Verifier system
│   ├── ollama_fix.py       # Compatibility patch for LiteLLM/Ollama
│   └── ...
├── verify_parallel.py      # CLI script to test the full flow
├── setup_ollama_models.sh  # Helper to alias local Ollama models
├── Makefile                # Command shortcuts
└── README.md               # This file
```
