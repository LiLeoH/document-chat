import asyncio
from pathlib import Path
from pymilvus import MilvusClient
import yaml
from milvus_operator import MilvusOperator


def test_milvus_info():
    workspace = "4e5a1ea2"
    collection_name = f"coll_{workspace}"
    db_path = f"workspace/{workspace}/milvus_local.db"

    client = MilvusClient(db_path)

    # 1. 查看当前库里有哪些 Collection
    collections = client.list_collections()
    print(f"Collections: {collections}")

    schema = client.describe_collection(collection_name)
    for field in schema["fields"]:
        print(f"字段名: {field['name']}, 类型: {field['type']}")

    # 2. 获取某个 Collection 的数据总数
    count = client.get_collection_stats(collection_name=collection_name)
    print(f"数据量: {count}")

    # 3. 查看具体数据 (Query 接口)
    # filter="" 表示不筛选
    data = client.query(
        collection_name=collection_name,
        filter="",
        output_fields=["*"],  # 返回所有字段
        # output_fields=[
        #     "id", "chunk_text", "concepts", "entities", "summary", "takeaways",
        #     "summary_vector", "takeaways_vector", "concepts_vector", "entities_vector",
        #     "summary_bm25", "takeaways_bm25", "concepts_bm25", "entities_bm25"
        # ],
        limit=100,
    )

    for item in data:
        # 向量字段太长进行截断输出
        if "summary_vector" in item:
            item["summary_vector"] = item["summary_vector"][:2]
        if "takeaways_vector" in item:
            item["takeaways_vector"] = item["takeaways_vector"][:2]
        if "concepts_vector" in item:
            item["concepts_vector"] = item["concepts_vector"][:2]
        if "entities_vector" in item:
            item["entities_vector"] = item["entities_vector"][:2]
        print(item)


def test_milvus_insert():
    workspace = "4e5a1ea2"
    workspace_path = Path(f"workspace/{workspace}")
    chunks_dir = workspace_path / "chunks"
    milvus_op = MilvusOperator(workspace=str(workspace_path))
    chunk_data_list = []  # 改为列表更方便
    print(f"Loading chunks from {chunks_dir}...")
    # 1. 准备数据
    for chunk_file in sorted(chunks_dir.glob("*.md")):
        with open(chunk_file, "r", encoding="utf-8") as f:
            content = f.read()
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter_str = parts[1]
                raw_content = parts[2].strip()
                chunk_data = yaml.safe_load(frontmatter_str)
                chunk_data["raw"] = raw_content
                chunk_data_list.append(chunk_data)

    # 2. 顺序执行插入任务，避免触发 API 并发限制
    async def run_all_inserts():
        for data in chunk_data_list:
            await milvus_op.insert_chunk(data)
            await asyncio.sleep(0.1)
            print(f"  - {data.get('source', 'Unknown')} inserted into Milvus.")

    # 3. 在同步测试函数中只调用一次 asyncio.run
    if chunk_data_list:
        print(f"Inserting {len(chunk_data_list)} chunks sequentially...")
        asyncio.run(run_all_inserts())
    print("Test data inserted into Milvus.")
