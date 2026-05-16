from PIL import ImageChops
import json
import asyncio
import os
from pathlib import Path
from typing import Union, List
from pymilvus import (
    MilvusClient,
    DataType,
    Function,
    FunctionType,
    AnnSearchRequest,
    RRFRanker,
)
from embedding import get_embedder

DB_NAME = "milvus_local.db"
COLLECTION_PREFIX = "coll_"
DIMENSION = 4096  # Dimension for qwen3-embedding-8b


class MilvusOperator:
    def __init__(self, workspace: str = ".", db_name: str = DB_NAME):
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.db_path = str(self.workspace / db_name)

        # Use the workspace directory name as part of the collection name
        # Prepend a prefix because Milvus collection names shouldn't start with numbers
        self.collection_name = f"{COLLECTION_PREFIX}{self.workspace.name}"

        remote_uri = os.environ.get("MILVUS_URI")
        if remote_uri:
            self.client = MilvusClient(uri=remote_uri)
        else:
            self.client = MilvusClient(self.db_path)
        self.embedder = get_embedder()
        self._init_collection()

    def _init_collection(self):
        """Initialize collection if it doesn't exist."""
        if self.client.has_collection(self.collection_name):
            return

        print(f"Creating Milvus collection: {self.collection_name}...")

        schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)

        # Add scalar fields
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(
            field_name="source", datatype=DataType.VARCHAR, max_length=1024
        )
        schema.add_field(
            field_name="summary",
            datatype=DataType.VARCHAR,
            max_length=65000,
            enable_analyzer=True,
        )
        schema.add_field(
            field_name="takeaways",
            datatype=DataType.VARCHAR,
            max_length=65000,
            enable_analyzer=True,
        )
        schema.add_field(
            field_name="concepts",
            datatype=DataType.VARCHAR,
            max_length=65000,
            enable_analyzer=True,
        )
        schema.add_field(
            field_name="entities",
            datatype=DataType.VARCHAR,
            max_length=65000,
            enable_analyzer=True,
        )
        schema.add_field(
            field_name="combined_content",
            datatype=DataType.VARCHAR,
            max_length=65000,
            enable_analyzer=True,
        )

        # Add one combined Sparse Vector field for BM25
        schema.add_field(
            field_name="combined_sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR
        )

        # Map a single combined scalar field to a sparse vector field using a BM25 function
        bm25_func = Function(
            name="combined_bm25_func",
            input_field_names=["combined_content"],
            output_field_names=["combined_sparse_vector"],
            function_type=FunctionType.BM25,
        )
        schema.add_function(bm25_func)

        schema.add_field(
            field_name="combined_dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=DIMENSION,
        )
        schema.add_field(
            field_name="chunk_dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=DIMENSION,
        )

        # Create indices for the vector fields to enable search
        index_params = self.client.prepare_index_params()

        # Dense vector indices
        # Dense vector index
        index_params.add_index(
            field_name="combined_dense_vector",
            metric_type="COSINE",
            index_type="FLAT",
        )
        index_params.add_index(
            field_name="chunk_dense_vector",
            metric_type="COSINE",
            index_type="FLAT",
        )

        # Sparse vector index for BM25
        index_params.add_index(
            field_name="combined_sparse_vector",
            metric_type="BM25",
            index_type="SPARSE_INVERTED_INDEX",
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )
        print("Collection created successfully.")

    def close(self):
        self.client.close()

    async def insert_chunk(self, parsed_dict: dict):
        """
        Embeds the required fields and inserts the document metadata and embeddings into Milvus.
        """
        summary = parsed_dict.get("summary", "")
        takeaways = parsed_dict.get("takeaways", [])
        concepts = parsed_dict.get("concepts", [])
        entities = parsed_dict.get("entities", {})

        # Generate a single embedding for the combined text of all attributes
        combined_text = f"{summary}\n{json.dumps(takeaways, ensure_ascii=False)}\n{json.dumps(concepts, ensure_ascii=False)}\n{json.dumps(entities, ensure_ascii=False)}"
        raw_content = parsed_dict.get("raw", "")

        # Generate embeddings in parallel
        print("Starting parallel embedding generation...")
        combined_vec, chunk_vec = await asyncio.gather(
            self.embedder.get_embedding(combined_text),
            self.embedder.get_embedding(raw_content),
        )

        data = {
            "source": parsed_dict.get("source", ""),
            "summary": summary if isinstance(summary, str) else str(summary),
            "takeaways": json.dumps(takeaways, ensure_ascii=False),
            "concepts": json.dumps(concepts, ensure_ascii=False),
            "entities": json.dumps(entities, ensure_ascii=False),
            "combined_content": combined_text,
            "combined_dense_vector": combined_vec,
            "chunk_dense_vector": chunk_vec,
        }

        # Milvus Lite is synchronous, use to_thread to avoid blocking the event loop
        print("start insert into Milvus")
        # res = self.client.insert(collection_name=self.collection_name, data=[data])
        res = await asyncio.to_thread(
            self.client.insert, collection_name=self.collection_name, data=[data]
        )
        self.client.flush(self.collection_name)
        return res

    async def search(
        self, query: str, search_field: str = "combined_dense_vector", limit: int = 3
    ):
        """
        Search within the combined dense vector field using ANN search asynchronously.
        """
        query_vec = await self.embedder.get_embedding(query)
        res = await asyncio.to_thread(
            self.client.search,
            collection_name=self.collection_name,
            data=[query_vec],
            anns_field=search_field,
            search_params={"metric_type": "COSINE"},
            limit=limit,
            output_fields=[
                "source",
                "summary",
                "takeaways",
                "concepts",
                "entities",
            ],
        )
        return res

    async def search_keyword(
        self, query: str, search_field: str = "combined_sparse_vector", limit: int = 3
    ):
        """
        Search within the combined sparse vector field using BM25 keyword search asynchronously.
        """
        res = await asyncio.to_thread(
            self.client.search,
            collection_name=self.collection_name,
            data=[query],
            anns_field=search_field,
            limit=limit,
            output_fields=[
                "source",
                "summary",
                "takeaways",
                "concepts",
                "entities",
            ],
        )
        return res

    async def search_hybrid_all(self, query: Union[str, List[str]], limit: int = 3):
        """
        Hybrid search (Dense + Sparse BM25) using combined fields.
        Supports single query string or a list of query strings for multi-vector search.
        """
        if isinstance(query, str):
            queries = [query]
        else:
            queries = query

        reqs = []
        for q in queries:
            query_vec = await self.embedder.get_embedding(q)
            reqs.append(
                AnnSearchRequest(
                    data=[query_vec],
                    anns_field="combined_dense_vector",
                    param={"metric_type": "COSINE"},
                    limit=limit,
                )
            )
            reqs.append(
                AnnSearchRequest(
                    data=[query_vec],
                    anns_field="chunk_dense_vector",
                    param={"metric_type": "COSINE"},
                    limit=limit,
                )
            )
            reqs.append(
                AnnSearchRequest(
                    data=[q],
                    anns_field="combined_sparse_vector",
                    param={"metric_type": "BM25"},
                    limit=limit,
                )
            )

        res = await asyncio.to_thread(
            self.client.hybrid_search,
            collection_name=self.collection_name,
            reqs=reqs,
            ranker=RRFRanker(),
            limit=limit,
            output_fields=[
                "source",
                "summary",
                "takeaways",
                "concepts",
                "entities",
            ],
        )
        return res


async def test_main():
    from pdf_processor import parse_document

    # Basic verification test
    op = MilvusOperator()
    print("\n--- Testing Insertion ---")
    mock_chunk = "# Test Chunk\nThis is a mock chunk."
    parsed = parse_document(mock_chunk)

    # Inject mock data since the default is empty
    parsed["summary"] = "This is a brief summary of the mock document." # type: ignore
    parsed["takeaways"] = [ # type: ignore
        "Takeaway 1: Milvus is fast",
        "Takeaway 2: Multi-vector rocks",
    ]
    parsed["concepts"] = ["Vector Database", "Embedding"] # type: ignore
    parsed["entities"] = {"Database": "Milvus", "Company": "Zilliz"} # type: ignore
    parsed["source"] = "mock_chunk.md" # type: ignore

    res = await op.insert_chunk(parsed) # type: ignore
    print(f"Inserted 1 record. ID: {res}")

    print("\n--- Testing Search (Combined Dense) ---")
    search_res = await op.search("Database technology")

    if search_res and len(search_res[0]) > 0:
        top_match = search_res[0][0]
        print(f"Matched ID: {top_match['id']}, Distance: {top_match['distance']}")
        print(f"Entities found: {top_match['entity']['entities']}")
    else:
        print("No matches found.")

    print("\n--- Testing Keyword Search (Combined Sparse BM25) ---")
    bm25_res = await op.search_keyword("brief summary")

    if bm25_res and len(bm25_res[0]) > 0:
        top_match = bm25_res[0][0]
        print(f"Matched ID: {top_match['id']}, Distance: {top_match['distance']}")
        print(f"Summary found: {top_match['entity']['summary']}")
    else:
        print("No matches found.")

    print("\n--- Testing Hybrid Search (All Fields) ---")
    hybrid_res = await op.search_hybrid_all("Database technology")

    if hybrid_res and len(hybrid_res[0]) > 0:
        top_match = hybrid_res[0][0]
        print(
            f"Matched ID: {top_match['id']}, Distance (RRF score): {top_match['distance']}"
        )
        print(f"Summary found: {top_match['entity']['summary']}")
    else:
        print("No matches found.")


if __name__ == "__main__":
    asyncio.run(test_main())
