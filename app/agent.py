# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import os
from zoneinfo import ZoneInfo

import google.auth
from google.adk.agents import Agent
from google.adk.apps.app import App
from google.adk.models.lite_llm import LiteLlm
import warnings

# Suppress Pydantic UserWarnings (serialization issues from LiteLLI integration/Ollama)
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

_, project_id = google.auth.default()
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")


def get_weather(query: str) -> str:
    """Simulates a web search. Use it get information on weather.

    Args:
        query: A string containing the location to get weather information for.

    Returns:
        A string with the simulated weather information for the queried location.
    """
    if "sf" in query.lower() or "san francisco" in query.lower():
        return "It's 60 degrees and foggy."
    return "It's 90 degrees and sunny."

def addition(a: int, b: int) -> int:
    """Returns the sum of two integers."""
    return a + b

def get_current_time(query: str) -> str:
    """Simulates getting the current time for a city.

    Args:
        city: The name of the city to get the current time for.

    Returns:
        A string with the current time information.
    """
    if "sf" in query.lower() or "san francisco" in query.lower():
        tz_identifier = "America/Los_Angeles"
    else:
        return f"Sorry, I don't have timezone information for query: {query}."

    tz = ZoneInfo(tz_identifier)
    now = datetime.datetime.now(tz)

# --- Switch to Parallel Agent with Orchestrator ---
from .agent_parallel import agent_system
from google.adk.tools import AgentTool
from app.ollama_fix import OllamaLiteLlm as LiteLlm

# Wrap the Parallel System as a Tool
research_team_tool = AgentTool(agent_system)

# Define the Root Orchestrator
orchestrator_agent = Agent(
    name="root_orchestrator",
    model=LiteLlm(model="ollama_chat/gpt-oss:20b-cloud"),
    instruction="""
    You are the Root Orchestrator Agent.
    
    1. **Interaction**: If the user says hello or greets you, reply politely, explain that you are an orchestrator managing a team of 7 expert AI models, and ask what they need help with. Do NOT call any tools for greetings.
    
    2. **Delegation**: If the user asks a question or request information, you MUST delegate it to your `parallel_verifier_system` tool (the Research Team).
    
    3. **Presentation**: After the Research Team returns with an answer, you must present it to the user clearly. "The team has found..."
    """,
    tools=[research_team_tool]
)

root_agent = orchestrator_agent

app = App(root_agent=root_agent, name="app")


