from rerank import get_reranker
import re
from typing import List
import os
import pandas as pd
import io
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from milvus_operator import MilvusOperator
from langchain.agents import create_agent


# Cache for Milvus operators and agents per workspace
_operators = {}
_agents = {}


async def init_retriever(workspace: str, op: MilvusOperator):
    """
    Initialize the retriever with the Milvus operator.
    """
    if workspace not in _operators:
        _operators[workspace] = op
    else:
        _operators[workspace].close()
        _operators[workspace] = op
    return _operators[workspace]


async def retrieve_context(
    query: List[str], raw_query: str, workspace: str, limit: int = 3
) -> str:
    """
    Search Milvus for context matching the query.
    """
    if workspace not in _operators:
        _operators[workspace] = MilvusOperator(workspace=workspace)

    op = _operators[workspace]

    print(f"Searching for: '{query}' in workspace: {workspace}...")

    # Use hybrid search (Dense + BM25) for best results
    search_res = await op.search_hybrid_all(query, limit=limit)

    context_parts = []
    source_list = []
    if search_res and len(search_res[0]) > 0:
        for i, hit in enumerate(search_res[0]):
            entity = hit["entity"]
            source = entity.get("source", "")
            source_list.append(source)

            # Read actual text from the chunk file
            chunk_file_path = os.path.join(workspace, "chunks", source)
            try:
                with open(chunk_file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Skip YAML frontmatter if present
                    if content.startswith("---"):
                        parts = content.split("---", 2)
                        text = parts[2].strip() if len(parts) > 2 else content
                    else:
                        text = content
            except Exception as e:
                text = f"[无法读取文件 {source}: {str(e)}]"

            context_parts.append(
                f"<data_fragment source='{source}'>\n{text}\n</data_fragment>"
            )

    print(f"DEBUG: Found {len(context_parts)} context parts")
    print(f"DEBUG: Sources: {source_list}")

    if not context_parts:
        return "数据库中未找到相关参考资料。"

    # rerank
    reranker = get_reranker()
    if reranker:
        raw_rerank_results = await reranker.rerank(raw_query, context_parts)

        # Sort results by relevance_score and extract document content
        sorted_results = sorted(
            raw_rerank_results, key=lambda x: x["relevance_score"], reverse=True
        )
        rerank_results = [res["document"] for res in sorted_results]

        # Apply rerank limit
        rerank_results = rerank_results[: int(os.environ.get("RERANK_LIMIT", 3))]
        # 从rerank_results中每一个的内容的’<data_fragment source='chunk_2.md'>‘提取出source，再打印
        for i, hit in enumerate(rerank_results):
            source = hit.split("<data_fragment source=")[1].split(">")[0]
            print(f"DEBUG: Rerank source {i}: {source}")
    else:
        print("DEBUG: No reranker configured, skipping rerank step.")
        rerank_results = context_parts

    return "\n\n".join(rerank_results)


def get_agent_executor(workspace: str):
    """
    Initialize a LangGraph ReAct agent with a retrieval tool.
    """
    llm = ChatOpenAI(
        model=os.environ.get("CHAT_MODEL", "qwen3.5-122b-a10b"),
        api_key=os.environ.get("OPENAI_API_KEY"), # type: ignore
        base_url=os.environ.get("OPENAI_BASE_URL"),
        temperature=0.5,
        extra_body={
            "thinking": {"type": "enabled"},
            # "thinking": {"type": "disabled"},
        },
        streaming=True,
    )

    frontmatter_list = []
    for chunk_file in os.listdir(os.path.join(workspace, "chunks")):
        if chunk_file.endswith(".md"):
            chunk_file_path = os.path.join(workspace, "chunks", chunk_file)
            with open(chunk_file_path, "r", encoding="utf-8") as f:
                content = f.read()
                # get YAML frontmatter
                if content.startswith("---"):
                    parts = content.split("---", 2)
                    yaml_content = parts[1]
                    frontmatter_list.append(yaml_content)
                    # frontmatter_list.append("```yaml\n" + yaml_content + "\n```")

    @tool
    async def search_documents(query: List[str], raw_query: str):
        """
        根据用户的问题，在向量数据库中搜索相关的文档资料。
        如果你需要回答关于项目概况、建设规模、建设内容等具体问题，请务必使用此工具。
        该工具会返回多个片段的摘要和来源文件名。

        Args:
            query: 搜索关键词或短语列表。建议提取用户问题中的核心实体或概念。
            raw_query: 用户原始问题。
        """
        return await retrieve_context(
            query, raw_query, workspace, limit=int(os.environ.get("RETRIEVE_LIMIT", 5))
        )

    @tool
    async def read_document_file(source: str):
        """
        读取指定的文档章节文件内容。
        当 search_documents 返回了来源列表，而你需要查看某个特定章节的详细全文或进行更深入分析时，请使用此工具。
        参数 source 必须是 search_documents 返回结果中的文件名或文档资料摘要的source字段（如 'chunk_1.md'）。
        """
        print(f"DEBUG: Calling read_document_file with source: '{source}'")
        chunk_file_path = os.path.join(workspace, "chunks", source)
        try:
            with open(chunk_file_path, "r", encoding="utf-8") as f:
                content = f.read()
                # Skip YAML frontmatter if present
                if content.startswith("---"):
                    parts = content.split("---", 2)
                    return parts[2].strip() if len(parts) > 2 else content
                return content
        except Exception as e:
            return f"错误: 无法读取文件 {source}。原因: {str(e)}"

    @tool
    async def parse_markdown_table(md_table: str, value: str):
        """
        解析 Markdown 格式的表格，并获取包含指定value的数据行。
        参数:
        - md_table: 原始 Markdown 格式的表格文本。以 '|' 分隔。
        - value: 需要查找的值（采用值相当判断，必须和单元格内容完全一致）
        """
        try:
            # 1. 预处理：去掉开头和结尾的 '|'，避免 Pandas 生成多余的空列
            cleaned_lines = []
            for line in md_table.strip().split("\n"):
                line = line.strip()
                if line.startswith("|"):
                    line = line[1:]
                if line.endswith("|"):
                    line = line[:-1]
                if line:
                    cleaned_lines.append(line)
            value = value.strip()  # 确保value也没有多余空格

            # 2. 重构为 CSV 格式，Pandas 解析更准确
            # 跳过第二行（--- 分隔符）
            csv_data = "\n".join([line for line in cleaned_lines if "---" not in line])

            df = pd.read_csv(io.StringIO(csv_data), sep="|", skipinitialspace=True)

            # 3. 清理列名（去除可能的空格）
            df.columns = df.columns.str.strip()
            # 清理所有单元格数据（去除两侧空格）
            df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

            data = df.to_dict(orient="records")

            # 4. 筛选逻辑
            results = [
                row
                for row in data
                if any(str(value) in str(cell_value) for cell_value in row.values())
            ]

            print("DEBUG: parse_markdown_table md_table: ", md_table)
            print("DEBUG: parse_markdown_table value: ", value)
            print("DEBUG: parse_markdown_table results: ", results)
            return results
        except Exception as e:
            return f"解析表格出错: {str(e)}"

    memory = MemorySaver()

    prompt = (
        """
# 角色与目标
你是一名资深的电力工程设计专家。你的唯一任务是基于提供的文档资料，专业、客观、准确地回答用户问题。

# 工具使用规范（严格按条件触发）
1. [索引工具]: 必须优先使用 `search_documents` 工具查找相关资料。
2. [阅读工具]: 下方提供的【摘要】仅作为“目录”参考。一旦在摘要中发现可能相关的分块，**必须**使用 `read_document_file` 工具读取该分块的详细原文，绝对不能直接基于【摘要】回答。
3. [表格解析]: 当需要从原文的复杂表格中提取数据时，必须使用 `parse_markdown_table` 工具确保数据准确。

# 核心约束与规则
- **绝对忠于原文**：仅根据 `read_document_file` 获取的原文或 `search_documents` 的检索结果作答，严禁捏造事实或使用外部知识。
- **信息缺失处理**：如果资料中未提及用户询问的信息，请直接回答：“根据提供的资料，我无法找到关于该问题的相关信息。”
- **冲突处理原则**：若检索到的原文之间存在冲突或不一致，你必须：1) 列出冲突的内容；2) 给出专业的逻辑分析；3) 得出一个你认为最合理的正确结论。
- **强制引用规范**：回答必须包含引用。引用的原文必须来自使用工具读取的“正文”，**绝对不能引用【摘要】中的文本**。

# 回答输出模板（请严格按此格式输出）
📝 **分析与回答**：
[在此处给出你的专业回答，如果存在资料冲突，在此处进行分析对比并得出结论]

📖 **参考资料**：
- **来源章节**：[明确指出是哪个分块/章节]
- **原文引用**："[在此处严格摘录原文句子]"

--- 

以下是文档各分块的【摘要】(yaml格式，仅作为目录索引，禁止直接用于引用)：

       """
        f"{'\n\n --- \n\n'.join(frontmatter_list)}"
    )
    # Create the ReAct agent
    agent_executor = create_agent(
        llm,
        tools=[search_documents, read_document_file, parse_markdown_table],
        checkpointer=memory,
        system_prompt=prompt,
    )
    print(f"DEBUG: Agent created with prompt: {prompt}")
    return agent_executor


def reset_agent(workspace: str):
    """
    Reset the agent for a workspace, effectively clearing its memory.
    """
    if workspace in _agents:
        print(f"Resetting agent for workspace: {workspace}")
        del _agents[workspace]


async def rag_flow(query: str, workspace: str, thread_id: str = "default_session"):
    """
    Full RAG flow implemented using LangGraph Agent.
    """
    if workspace not in _agents:
        print(f"Initializing new agent for workspace: {workspace}")
        _agents[workspace] = get_agent_executor(workspace)

    agent = _agents[workspace]

    # Configuration for memory (persistence)
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Run the agent asynchronously
        print(f"Agent processing query: {query}")

        async for msg, metadata in agent.astream(
            {"messages": [("user", query)]},
            stream_mode="messages",
            config=config,
        ):
            if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                print(
                    f"DEBUG: Agent stream response_metadata: {msg.response_metadata} usage_metadata: {msg.usage_metadata}"
                )
            if metadata.get("langgraph_node") == "model":
                content = msg.content
                if isinstance(content, str) and content:
                    yield content
                elif isinstance(content, list):
                    # Handle content blocks if necessary
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            yield block.get("text", "")
                        elif isinstance(block, str):
                            yield block

    except Exception as e:
        yield f"\nAgent Error: {str(e)}"


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    # Test block
    async def test():
        workspace = "workspace/469a9e97"
        if os.path.exists(workspace):
            print(f"\nFinal Answer:\n", end="")
            async for chunk in rag_flow("工程总装机容量是多少？", workspace):
                print(chunk, end="", flush=True)
            print()
        else:
            print(f"Workspace {workspace} not found. Please run main.py first.")

    asyncio.run(test())
