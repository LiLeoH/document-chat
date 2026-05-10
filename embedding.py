import os
import json
import httpx
import asyncio
from abc import ABC, abstractmethod
from openai import AsyncOpenAI


class DocumentEmbedding(ABC):
    """Abstract base class for embeddings."""

    def __init__(self, model_name: str = ""):
        self.model_name = model_name or os.environ.get(
            "EMBEDDING_MODEL", "qwen3-embedding-8b"
        )

    @abstractmethod
    async def get_embedding(self, text: str) -> list[float]:
        """Get embedding for a single text string asynchronously."""
        pass

    async def get_field_embedding(self, field_data) -> list[float]:
        """
        Convert various data types (str, list, dict) to a string and get its embedding.
        """
        if isinstance(field_data, str):
            text = field_data
        elif isinstance(field_data, list):
            text = " ".join([str(item) for item in field_data])
        elif isinstance(field_data, dict):
            text = json.dumps(field_data, ensure_ascii=False)
        else:
            text = str(field_data)

        return await self.get_embedding(text)


class QianfanEmbedding(DocumentEmbedding):
    """Embedding implementation using Qianfan platform (via httpx)."""

    def __init__(self, model_name: str = ""):
        super().__init__(model_name)
        self.url = os.environ.get("QIANFAN_EMBEDDING_BASE_URL")
        self.api_key = os.environ.get("EMBEDDING_API_KEY")

    async def get_embedding(self, text: str) -> list[float]:
        """Get embedding for a single text string asynchronously using httpx."""
        if not text.strip():
            return [0.0] * 4096  # Default dimension for Qwen3

        max_context = int(os.environ.get("EMBEDDING_CONTEXT_LEN", 500))
        if len(text) > max_context:
            text = text[:max_context]
        payload = json.dumps({"model": self.model_name, "input": [text]})
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(self.url, headers=headers, content=payload)
            if response.status_code == 200:
                data = response.json()
                if "data" in data and len(data["data"]) > 0:
                    return data["data"][0]["embedding"]
                else:
                    raise Exception(f"Unexpected response format: {data}")
            else:
                if response.status_code == 429:
                    await asyncio.sleep(1)
                    return await self.get_embedding(text)
                raise Exception(f"Failed to fetch embedding from API: {response.text}")


class OpenAIEmbedding(DocumentEmbedding):
    """Embedding implementation using the official OpenAI SDK."""

    def __init__(self, model_name: str = ""):
        super().__init__(model_name)
        self.client = AsyncOpenAI(
            api_key=os.environ.get("EMBEDDING_API_KEY"),
            base_url=os.environ.get("OPENAI_EMBEDDING_BASE_URL"),
        )

    async def get_embedding(self, text: str) -> list[float]:
        """Get embedding for a single text string asynchronously using OpenAI SDK."""
        if not text.strip():
            return [0.0] * 4096

        max_context = int(os.environ.get("EMBEDDING_CONTEXT_LEN", 500))
        if len(text) > max_context:
            text = text[:max_context]

        try:
            response = await self.client.embeddings.create(
                model=self.model_name, input=text, encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            # Simple retry for rate limit
            if "429" in str(e):
                await asyncio.sleep(1)
                return await self.get_embedding(text)
            raise e


# Singleton instance
_embedder_instance = None


def get_embedder() -> DocumentEmbedding:
    global _embedder_instance
    if _embedder_instance is None:
        if os.environ.get("OPENAI_EMBEDDING_BASE_URL"):
            print("Initializing OpenAI Embedding API Client (SDK)...")
            _embedder_instance = OpenAIEmbedding()
        elif os.environ.get("QIANFAN_EMBEDDING_BASE_URL"):
            print("Initializing Qianfan Embedding API Client (httpx)...")
            _embedder_instance = QianfanEmbedding()
        else:
            # Default fallback
            print("Initializing OpenAI Embedding API Client (SDK)...")
            _embedder_instance = OpenAIEmbedding()
    return _embedder_instance
