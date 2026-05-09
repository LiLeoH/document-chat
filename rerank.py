import os
import json
import httpx
import asyncio
from dotenv import load_dotenv


class DocumentReranker:
    def __init__(self, model_name: str = None):
        """
        Initialize the rerank model using Baidu Qianfan API.
        """
        self.model_name = model_name or os.environ.get(
            "RERANK_MODEL", "qwen3-reranker-8b"
        )
        self.url = "https://qianfan.baidubce.com/v2/rerank"
        # API Key (Token)
        self.api_key = os.environ.get("QIANFAN_API_KEY")

    async def rerank(self, query: str, documents: list[str]) -> list[dict]:
        """
        Rerank a list of documents based on a query asynchronously.
        Returns a list of results with relevance scores.
        """
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
                # Handle rate limits or other errors
                if response.status_code == 429:
                    # Simple retry once for rate limit
                    await asyncio.sleep(1)
                    return await self.rerank(query, documents)
                raise Exception(
                    f"Failed to fetch rerank results from Qianfan API: {response.text}"
                )


# Singleton instance
_reranker_instance = None


def get_reranker() -> DocumentReranker:
    global _reranker_instance
    if _reranker_instance is None:
        print("Initializing Qianfan Rerank API Client (Async)...")
        _reranker_instance = DocumentReranker()
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
