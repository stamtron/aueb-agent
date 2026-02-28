import asyncio
import warnings
import logging

# Suppress Pydantic UserWarnings (serialization issues from LiteLLM integration)
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Suppress the ADK Runner's cosmetic "App name mismatch" warning.
# This fires because inspect.getmodule() on the base Agent class resolves to
# google.adk.agents (inferred name: "agents") instead of the local app/ module.
logging.getLogger("google.adk.runners").setLevel(logging.ERROR)
from app.agent import app  # Import the App instance
from google.adk import Runner
from google.genai import types as genai_types
from google.adk.sessions import InMemorySessionService, Session, State

async def main():
    print("Initializing Runner with Root Orchestrator...")
    session_service = InMemorySessionService()
    runner = Runner(app=app, session_service=session_service)

    # Create session
    await session_service.create_session(app_name=app.name, user_id="test_user", session_id="test_session", state={})

    # helper to run and print
    async def run_turn(text):
        print(f"\nUser: {text}")
        events = runner.run_async(
            user_id="test_user",
            session_id="test_session",
            new_message=genai_types.Content(parts=[genai_types.Part(text=text)])
        )
        
        print("Agent Response(s):")
        async for event in events:
            if hasattr(event, "content") and event.content:
                text_content = " ".join(part.text for part in event.content.parts if part.text)
                print(f" > {text_content.strip()}")
            
            if hasattr(event, "tool_use") and event.tool_use:
                print(f" [System] Orchestrator is calling tool: {event.tool_use.name} (Please wait, this involves 7 agents...)")

            # Check for tool calls or specific debug info if needed
        
        # After run, we can inspect session state if available via service
        session = await runner.session_service.get_session(session_id="test_session", user_id="test_user", app_name=app.name)
        state = session.state
        
        # Check specific parallel outputs if they exist
        # Only print if we are likely in a deep research turn
        if "worker_gpt_oss_20b_response" in state:
            # Check if verifier ran
            vf = state.get("final_verified_response", "")
            if vf:
                 # Use a simple check to see if we haven't printed this before? 
                 # Actually, for this script, we just inspect state at end of turn.
                 pass

    print("\n--- Test 1: Greeting (Should be handled by Orchestrator) ---")
    await run_turn("Hello, who are you?")

    print("\n--- Test 2: Complex Question (Should be delegated) ---")
    await run_turn("What is the importance of attention in LLMs?")

    # Final State Inspection
    print("\n[Final State Inspection of Workers]")
    session = await runner.session_service.get_session(session_id="test_session", user_id="test_user", app_name=app.name)
    state = session.state
    workers = [
        "worker_llama", "worker_deepseek", "worker_mistral"
    ]
    for w in workers:
        resp = state.get(f"{w}_response", "N/A")
        snippet = resp[:50] + "..." if isinstance(resp, str) else str(resp)
        print(f"{w}: {snippet}")


if __name__ == "__main__":
    asyncio.run(main())
