import os
import json
import httpx
import asyncio


class DocumentEmbedding:
    def __init__(self, model_name: str = None):
        """
        Initialize the embedding model using Baidu Qianfan API.
        """
        self.model_name = model_name or os.environ.get(
            "EMBEDDING_MODEL", "qwen3-embedding-8b"
        )
        self.url = "https://qianfan.baidubce.com/v2/embeddings"
        # API Key (Token)
        self.api_key = os.environ.get("QIANFAN_API_KEY")

    async def get_embedding(self, text: str) -> list[float]:
        """Get embedding for a single text string asynchronously."""
        if not text.strip():
            return [0.0] * 4096  # Return zero vector for empty text

        if len(text) > 8000:
            text = text[:8000]
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
                # Handle rate limits or other errors
                if response.status_code == 429:
                    # Simple retry once for rate limit
                    await asyncio.sleep(1)
                    return await self.get_embedding(text)
                raise Exception(
                    f"Failed to fetch embedding from Qianfan API: {response.text}"
                )

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


# Singleton instance
_embedder_instance = None


def get_embedder() -> DocumentEmbedding:
    global _embedder_instance
    if _embedder_instance is None:
        print("Initializing Qianfan Embedding API Client (Async)...")
        _embedder_instance = DocumentEmbedding()
    return _embedder_instance
