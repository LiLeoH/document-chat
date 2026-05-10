import os
import json
import httpx
import asyncio
from abc import ABC, abstractmethod
from dotenv import load_dotenv


class DocumentReranker(ABC):
    """Abstract base class for rerankers."""

    def __init__(self, model_name: str = ""):
        self.model_name = model_name or os.environ.get(
            "RERANK_MODEL", "qwen3-reranker-8b"
        )

    @abstractmethod
    async def rerank(self, query: str, documents: list[str]) -> list[dict]:
        """
        Rerank a list of documents based on a query asynchronously.
        Returns a list of results with relevance scores.
        """
        pass


class QianfanReranker(DocumentReranker):
    """Reranker implementation using Qianfan platform (via httpx)."""

    def __init__(self, model_name: str = ""):
        super().__init__(model_name)
        self.url = os.environ.get("QIANFAN_RERANK_BASE_URL")
        self.api_key = os.environ.get("RERANK_API_KEY")

    async def rerank(self, query: str, documents: list[str]) -> list[dict]:
        if not documents:
            return []

        payload = json.dumps(
            {"model": self.model_name, "query": query, "documents": documents}
        )
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(self.url, headers=headers, content=payload)
            if response.status_code == 200:
                data = response.json()
                if "results" in data:
                    return data["results"]
                else:
                    raise Exception(f"Unexpected response format: {data}")
            else:
                if response.status_code == 429:
                    await asyncio.sleep(1)
                    return await self.rerank(query, documents)
                raise Exception(
                    f"Failed to fetch rerank results from API: {response.text}"
                )


class DashScopeReranker(DocumentReranker):
    """Reranker implementation using DashScope platform."""

    def __init__(self, model_name: str = ""):
        super().__init__(model_name)
        self.url = os.environ.get("DASHSCOPE_RERANK_BASE_URL")
        self.api_key = os.environ.get("RERANK_API_KEY")

    async def rerank(self, query: str, documents: list[str]) -> list[dict]:
        if not documents:
            return []

        payload = json.dumps({
            "model": self.model_name,
            "input": {
                "query": query,
                "documents": documents
            },
            "parameters": {
                "return_documents": True,
                "top_n": len(documents)
            }
        })
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(self.url, headers=headers, content=payload)
            if response.status_code == 200:
                data = response.json()
                if "output" in data and "results" in data["output"]:
                    return data["output"]["results"]
                else:
                    raise Exception(f"Unexpected response format: {data}")
            else:
                if response.status_code == 429:
                    await asyncio.sleep(1)
                    return await self.rerank(query, documents)
                raise Exception(
                    f"Failed to fetch rerank results from DashScope: {response.text}"
                )


# Singleton instance
_reranker_instance = None


def get_reranker() -> DocumentReranker | None:
    global _reranker_instance
    if _reranker_instance is None:
        if os.environ.get("DASHSCOPE_RERANK_BASE_URL"):
            print("Initializing DashScope Rerank API Client...")
            _reranker_instance = DashScopeReranker()
        elif os.environ.get("QIANFAN_RERANK_BASE_URL"):
            print("Initializing Qianfan Rerank API Client (httpx)...")
            _reranker_instance = QianfanReranker()
        else:
            print("No reranker configured.")
    return _reranker_instance


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # Test block
    async def test():
        reranker = get_reranker()
        query = "上海天气"
        docs = ["上海气候", "北京美食"]
        try:
            results = await reranker.rerank(query, docs)
            print(json.dumps(results, indent=4, ensure_ascii=False))
        except Exception as e:
            print(f"Error: {e}")

    asyncio.run(test())
