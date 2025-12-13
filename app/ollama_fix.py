
from google.adk.models.lite_llm import LiteLlm
import litellm

def normalize_messages_for_ollama(messages):
    """
    Flattens message content lists (used for multimodal) into strings
    because current Ollama/LiteLLM integration fails with array content.
    """
    normalized = []
    for m in messages:
        if isinstance(m.get("content"), list):
            # extract text parts only
            text = " ".join(
                part.get("text", "")
                for part in m["content"]
                if isinstance(part, dict) and part.get("type") == "text"
            )
            # Fallback if no text parts found (e.g. only images)
            if not text:
               # If there are strings in the list (some formats), join them
               text = " ".join(p for p in m["content"] if isinstance(p, str))
            
            normalized.append({**m, "content": text})
        else:
            normalized.append(m)
    return normalized

class OllamaClientWrapper:
    def __init__(self, original_client):
        self.original_client = original_client

    async def acompletion(self, **kwargs):
        if "messages" in kwargs:
            kwargs["messages"] = normalize_messages_for_ollama(kwargs["messages"])
        return await self.original_client.acompletion(**kwargs)
    
    def completion(self, **kwargs):
        if "messages" in kwargs:
            kwargs["messages"] = normalize_messages_for_ollama(kwargs["messages"])
        return self.original_client.completion(**kwargs)
    
    # Delegate other attribute accesses to original client
    def __getattr__(self, name):
        return getattr(self.original_client, name)


class OllamaLiteLlm(LiteLlm):
    """
    Subclass of LiteLlm that wraps the internal client to normalize messages
    for Ollama compatibility.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Wrap the llm_client to intercept calls
        self.llm_client = OllamaClientWrapper(self.llm_client)

