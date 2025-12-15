from google.adk.models.base_llm import BaseLlm
from google.genai import types as genai_types
from google.adk.models.llm_response import LlmResponse

from ollama import AsyncClient
from pydantic import PrivateAttr
import os

class OllamaCloudLlm(BaseLlm):
    _client: AsyncClient = PrivateAttr()

    def __init__(self, model_name: str):
        super().__init__(model=model_name)
        self._client = AsyncClient(
            host="https://ollama.com",
            headers={
                "Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"
            }
        )

    async def generate_content_async(self, contents, **kwargs):
        # --- build messages (unchanged) ---
        flat_contents = []
        for item in contents:
            if isinstance(item, (list, tuple)):
                flat_contents.extend(item)
            else:
                flat_contents.append(item)

        messages = []   
        for c in flat_contents:
            if not hasattr(c, "parts"):
                continue
            text = " ".join(
                part.text for part in c.parts
                if hasattr(part, "text") and part.text
            )
            if text:
                messages.append({"role": "user", "content": text})

        # --- call Ollama Cloud ---
        resp = await self._client.chat(
            model=self.model,
            messages=messages,
        )

        # --- build GenAI response ---
        llm_response = genai_types.GenerateContentResponse(
            candidates=[
                genai_types.Candidate(
                    content=genai_types.Content(
                        parts=[genai_types.Part(text=resp["message"]["content"])]
                    ),
                    finish_reason="STOP",
                )
            ]
        )

        # --- wrap in ADK Event (CRITICAL) ---
        yield LlmResponse.create(llm_response)
