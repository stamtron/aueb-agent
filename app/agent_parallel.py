
from google.adk.agents import Agent, ParallelAgent, SequentialAgent
# from google.adk.models.lite_llm import LiteLlm # Replaced with custom fix
from app.ollama_fix import OllamaLiteLlm as LiteLlm # Alias it to minimize code changes
from google.genai import types as genai_types
from ddgs import DDGS
import os
from app.ollama_cloud_model import OllamaCloudLlm


CLOUD_OLLAMA_BASE = "https://ollama.com"

MODEL_ROUTER = {
    "gpt-oss:20b-cloud": {
        "model": "ollama/gpt-oss:20b-cloud",
        "api_base": CLOUD_OLLAMA_BASE,
    },
    "gpt-oss:120b-cloud": {   # ← THIS WAS MISSING
        "model": "ollama/gpt-oss:120b-cloud",
        "api_base": CLOUD_OLLAMA_BASE,
    },
    "deepseek-v3.1:671b-cloud": {
        "model": "ollama/deepseek-v3.1:671b-cloud",
        "api_base": CLOUD_OLLAMA_BASE,
    },
    "qwen3-coder:480b-cloud": {
        "model": "ollama/qwen3-coder:480b-cloud",
        "api_base": CLOUD_OLLAMA_BASE,
    },
    "qwen3-vl:235b-cloud": {
        "model": "ollama/qwen3-vl:235b-cloud",
        "api_base": CLOUD_OLLAMA_BASE,
    },
    "minimax-m2:cloud": {
        "model": "ollama/minimax-m2:cloud",
        "api_base": CLOUD_OLLAMA_BASE,
    },
    "glm-4.6:cloud": {
        "model": "ollama/glm-4.6:cloud",
        "api_base": CLOUD_OLLAMA_BASE,
    },
}


# --- Tools ---

def duckduckgo_search_tool(query: str) -> str:
    """
    Performs a web search using DuckDuckGo.
    """
    try:
        results = DDGS().text(keywords=query, max_results=3)
        if not results:
            return "No results found."
        return str(results)
    except Exception as e:
        return f"Search failed: {e}"

# --- Worker Agents (Ollama) ---

# Helper to create workers easily
def create_worker(name, model_id, focus):
    if model_id.endswith("-cloud") or model_id.endswith(":cloud"):
        model = OllamaCloudLlm(model_id)
    else:
        model = LiteLlm(model=f"ollama_chat/{model_id}")

    return Agent(
        name=name,
        model=model,
        instruction=f"""
        You are an expert AI model specialized in your architecture.
        Your model ID is {model_id}.
        Focus on answering the user's question from your unique perspective: {focus}.
        """,
        output_key=f"{name}_response",
    )



# 1. llama3.2:latest — General Purpose
worker_llama = create_worker(
    "worker_llama",
    "llama3.2:latest",
    "General purpose reasoning and open-source alignment"
)

# 2. deepseek-coder:1.3b — Coding & Reasoning
worker_deepseek = create_worker(
    "worker_deepseek",
    "deepseek-coder:1.3b",
    "Deep technical reasoning and coding"
)

# 3. mistral:latest — Synthesis & Large-scale Knowledge
worker_mistral = create_worker(
    "worker_mistral",
    "mistral:latest",
    "Large-scale knowledge synthesis"
)

# --- Parallel Orchestration ---

parallel_workers = ParallelAgent(
    name="parallel_workers",
    sub_agents=[
        worker_llama,
        worker_deepseek,
        worker_mistral,
    ],
    description="Consults 3 different open source models in parallel."
)

# --- Verifier/Summarizer Agent ---

verifier_instruction = """
You are a Lead Researcher and Verifier.
Your task is to synthesize the answers provided by a panel of 3 expert AI models.

The experts have provided their responses in the session state.
Synthesize their perspectives into a single, comprehensive, and verified answer.
IF there are conflicting facts, use the `duckduckgo_search_tool` tool to verify.

Check the following keys in state for their inputs:
- worker_llama_response
- worker_deepseek_response
- worker_mistral_response

Provide a final, verified response to the user.
"""

verifier_agent = Agent(
    name="verifier_agent",
    model=LiteLlm(model="ollama_chat/llama3.2:latest"),
    instruction=verifier_instruction,
    tools=[duckduckgo_search_tool],
    output_key="final_verified_response"
)

# --- Main Pipeline ---

agent_system = SequentialAgent(
    name="parallel_verifier_system",
    sub_agents=[parallel_workers, verifier_agent],
    description="A system that consults 3 distinct OSS models in parallel and synthesizes their answers."
)

