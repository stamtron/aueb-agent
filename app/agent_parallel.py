
from google.adk.agents import Agent, ParallelAgent, SequentialAgent
# from google.adk.models.lite_llm import LiteLlm # Replaced with custom fix
from app.ollama_fix import OllamaLiteLlm as LiteLlm # Alias it to minimize code changes
from google.genai import types as genai_types
from duckduckgo_search import DDGS

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
    return Agent(
        name=name,
        model=LiteLlm(model=f"ollama_chat/{model_id}"),
        instruction=f"""
        You are an expert AI model specialized in your architecture.
        Your model ID is {model_id}.
        Focus on answering the user's question from your unique perspective: {focus}.
        """,
        output_key=f"{name}_response",
    )

# 1. gpt-oss:20b-cloud
worker_gpt_oss_20b = create_worker(
    "worker_gpt_oss_20b",
    "gpt-oss:20b-cloud",
    "General Purpose, Open Source alignment"
)

# 2. deepseek-v3.1:671b-cloud
worker_deepseek = create_worker(
    "worker_deepseek",
    "deepseek-v3.1:671b-cloud",
    "Deep technical reasoning and coding"
)

# 3. qwen3-coder:480b-cloud
worker_qwen_coder = create_worker(
    "worker_qwen_coder",
    "qwen3-coder:480b-cloud",
    "Software Engineering and Algorithms"
)

# 4. qwen3-vl:235b-cloud
worker_qwen_vl = create_worker(
    "worker_qwen_vl",
    "qwen3-vl:235b-cloud",
    "Visual understanding and multimodal context"
)

# 5. minimax-m2:cloud
worker_minimax = create_worker(
    "worker_minimax",
    "minimax-m2:cloud",
    "Creative writing and storytelling"
)

# 6. glm-4.6:cloud
worker_glm = create_worker(
    "worker_glm",
    "glm-4.6:cloud",
    "Bilingual (English/Chinese) and academic knowledge"
)

# 7. gpt-oss:120b
worker_gpt_oss_120b = create_worker(
    "worker_gpt_oss_120b",
    "gpt-oss:120b",
    "Large-scale knowledge synthesis"
)

# --- Parallel Orchestration ---

parallel_workers = ParallelAgent(
    name="parallel_workers",
    sub_agents=[
        worker_gpt_oss_20b,
        worker_deepseek,
        worker_qwen_coder,
        worker_qwen_vl,
        worker_minimax,
        worker_glm,
        worker_gpt_oss_120b
    ],
    description="Consults 7 different open source models in parallel."
)

# --- Verifier/Summarizer Agent ---

# Uses gpt-oss:20b-cloud as requested for the summary/verification
verifier_instruction = """
You are a Lead Researcher and Verifier.
Your task is to synthesize the answers provided by a panel of 7 expert AI models.

The experts have provided their responses in the session state.
Synthesize their perspectives into a single, comprehensive, and verified answer.
IF there are conflicting facts, use the `duckduckgo_search_tool` tool to verify.

Check the following keys in state for their inputs:
- worker_gpt_oss_20b_response
- worker_deepseek_response
- worker_qwen_coder_response
- worker_qwen_vl_response
- worker_minimax_response
- worker_glm_response
- worker_gpt_oss_120b_response

Provide a final, verified response to the user.
"""

verifier_agent = Agent(
    name="verifier_agent",
    model=LiteLlm(model="ollama_chat/gpt-oss:20b-cloud"),
    instruction=verifier_instruction,
    tools=[duckduckgo_search_tool],
    output_key="final_verified_response"
)

# --- Main Pipeline ---

agent_system = SequentialAgent(
    name="parallel_verifier_system",
    sub_agents=[parallel_workers, verifier_agent],
    description="A system that consults 7 distinct OSS models in parallel and synthesizes their answers."
)
