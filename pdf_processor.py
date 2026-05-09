import json
import yaml
import pymupdf4llm
from pathlib import Path
import fitz
import re
import uuid
import os
from typing import List, Dict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


def pdf_to_markdown(pdf_path: str, output_md_path: str = None) -> tuple[str, str]:
    """
    Reads a PDF file and converts it to Markdown format.

    Args:
        pdf_path: Path to the input PDF file.
        output_md_path: Optional path to save the Markdown output.
                        If None, saves with the same name but .md extension inside a random subfolder.

    Returns:
        A tuple of (extracted_markdown_text, output_md_path).
    """
    if output_md_path is None:
        pdf_file = Path(pdf_path)
        random_dir = Path(f"workspace/{uuid.uuid4().hex[:8]}")
        random_dir.mkdir(parents=True, exist_ok=True)
        output_md_path = str(random_dir / pdf_file.with_suffix(".md").name)
    else:
        Path(output_md_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"Converting '{pdf_path}' to Markdown...")

    # Extract TOC (bookmarks) using fitz
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()

    # Organize TOC by page number
    toc_by_page = {}
    for level, title, page in toc:
        if page not in toc_by_page:
            toc_by_page[page] = []
        toc_by_page[page].append((level, title.strip()))

    # Extract Markdown chunks per page
    md_chunks = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)

    final_md = []
    for i, chunk in enumerate(md_chunks):
        page_num = i + 1
        page_text = chunk["text"]

        # Strip all existing markdown headings to remove fake ones added by pymupdf4llm
        page_text = re.sub(r"^#+\s+", "", page_text, flags=re.MULTILINE)

        # Inject Markdown headings based on TOC
        if page_num in toc_by_page:
            for level, title in toc_by_page[page_num]:
                escaped_title = re.escape(title)
                # Strict match: Title alone on a line, possibly with # or ** markers
                pattern = re.compile(
                    rf"^(#*\s*\**\s*{escaped_title}\s*\**\s*)$", re.MULTILINE
                )
                header_prefix = "#" * level
                replacement = f"{header_prefix} {title}"

                if pattern.search(page_text):
                    page_text = pattern.sub(replacement, page_text)
                else:
                    # Fuzzy match: remove whitespaces
                    title_no_space = re.sub(r"\s+", "", title)
                    lines = page_text.split("\n")
                    for j, line in enumerate(lines):
                        line_no_space = re.sub(r"[\s#*]+", "", line)
                        if title_no_space and title_no_space in line_no_space:
                            # If the matched line is roughly just the title length, replace it
                            if len(line_no_space) <= len(title_no_space) + 10:
                                lines[j] = replacement
                                break
                    page_text = "\n".join(lines)

        final_md.append(page_text)

    # Combine all pages
    md_text = "\n\n".join(final_md)

    # Save the Markdown text to a file
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    print(f"Successfully saved Markdown to '{output_md_path}'")
    return md_text, output_md_path


def split_markdown_by_chapters(md_text: str, max_length: int = 30000) -> list[str]:
    """
    Splits a markdown text into chunks.
    Rules:
    1. Each chunk should not exceed max_length characters.
    2. Major chapters (lines starting with '# ') should not be split across chunks.
    3. If a major chapter exceeds max_length, it is kept as a single chunk.
    4. If a chunk is smaller than max_length/10, it is merged with adjacent chunks.
    """
    # Split text at the start of lines beginning with '# '
    sections = re.split(r"(?m)^(?=# )", md_text)

    # Clean up empty strings from the split
    sections = [s for s in sections if s.strip()]

    if not sections:
        return []

    # Identify title from the first SECTION
    title = ""
    if len(sections[0]) < 50 and len(sections) > 1:
        title = sections.pop(0).strip()

    chunks = []
    current_chunk = ""

    for section in sections:
        # If adding this section exceeds max_length and we already have content in current_chunk
        if len(current_chunk) + len(section) > max_length and current_chunk:
            chunks.append(current_chunk)
            current_chunk = section
        else:
            current_chunk += section

    if current_chunk:
        chunks.append(current_chunk)

    # Prepend title to all chunks if found
    if title:
        chunks = [f"{title}\n\n{c}" for c in chunks]

    # Merge small chunks (less than 1/10 of max_length)
    if len(chunks) > 1:
        min_length = max_length // 10
        i = 0
        while i < len(chunks):
            if len(chunks[i]) < min_length:
                if i < len(chunks) - 1:
                    # Merge current small chunk into the next one
                    chunks[i + 1] = chunks[i] + chunks[i + 1]
                    chunks.pop(i)
                    # Don't increment i, check the new combined chunk at this position
                elif i > 0:
                    # It's the last chunk and it's small, merge it into the previous one
                    chunks[i - 1] = chunks[i - 1] + chunks[i]
                    chunks.pop(i)
                    # No more chunks to check
                    break
            else:
                i += 1

    return chunks


class DocumentMetadata(BaseModel):
    """提取的文档元数据结构"""

    summary: str = Field(description="文档块的简要摘要")
    takeaways: List[str] = Field(description="文档块的关键点/要点列表")
    concepts: List[str] = Field(description="文档块中提到的核心概念列表")
    entities: Dict[str, str] = Field(
        description="文档块中提到的实体及其简要说明（键值对）"
    )


async def parse_document(chunk: str, extra_info: str = "") -> dict:
    """
    使用 LangChain 从大模型获取结构化解析结果。
    """
    llm = ChatOpenAI(
        model=os.environ.get("SUMMARY_MODEL", "qwen3.5-122b-a10b"),
        api_key=os.environ.get("QIANFAN_API_KEY"),
        base_url=os.environ.get("QIANFAN_BASE_URL", "https://qianfan.baidubce.com/v2"),
        temperature=0.1,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一名电力工程设计专家。"),
            (
                "user",
                """
下面是一份电力工程设计说明书的部分片段，请分析该文档并**严格**按下面的json格式输出内容：

# 输出格式
```json
{{"summary":"", "takeaways": [""], "concepts": [""], "entities":{{"$key":"$value"}}}}
```

## 字段说明
`summary`为该文档的摘要内容，同时需要满足以下要求：
- 不超过200字
- 不需要有细节描述，但不能遗漏内容点

`takeaways`为该文档的要点：
- 如果有表格，需要包含表格内容的描述

`concepts`为该文档中的抽象概念、主题等：
- 抽取最重要的 3 ~ 8 个

`entities`为该文档中的人、公司、地点、工具和其他具体实体：
- 抽取最重要的 3 ~ 8 个
- 其中`key`为该实体的类型，比如人物，公司，项目名等
- 如果多项`key`相关，合并到一项中，用逗号分割

# 其它约束条件
要对所有字段内容做向量数据库入库专用描述，要求： 
1. 仅保留对语义检索有价值的信息；
2. 禁止自然语言叙述；
3. 禁止完整句子；
4. 使用短语、标签、术语枚举；
5. 删除空话、套话、管理口号、流程性描述；
6. 删除“提高效率”“安全可靠”“满足要求”等低信息密度内容；
7. 优先保留：
   * 电压等级
   * 系统名称
   * 设备名称
   * 设备型号
   * 专业术语
   * 标准编号
   * 通信协议
   * 拓扑结构
   * 功能逻辑
   * 保护逻辑
   * 故障类型
   * 工程场景
8. 可用补充：
   * 项目名称
   * 行业别名
   * 常用简称
   * 英文缩写
   * 相关设备
   * 上下游系统
   * 关联专业
{extra_info_block}
9. 保持术语原貌不得改写。
10. 输出高密度术语集合；
11. 避免冗余重复；
12. 所有内容仅服务向量检索召回。

---

待解析的技术文档：
{input}
        """,
            ),
        ]
    )

    structured_llm = llm.with_structured_output(DocumentMetadata)
    chain = prompt | structured_llm

    # 准备输入数据
    input_data = {
        "input": chunk,
        "extra_info_block": (
            "\n"
            + "\n".join(
                f"   * {item.strip()}"
                for item in extra_info.splitlines()
                if item.strip()
            )
            if extra_info and extra_info.strip()
            else ""
        ),
    }

    try:
        # 打印实际 Prompt
        formatted_prompt = prompt.invoke(input_data)
        print(
            f"\n--- [DEBUG] Actual Prompt ---\n{formatted_prompt.to_string()}\n-----------------------------\n"
        )

        # 执行 LLM 调用
        metadata_obj = await chain.ainvoke(input_data)

        # 转换为字典并添加原始文本
        result = metadata_obj.model_dump()
        result["raw"] = chunk
        return result
    except Exception as e:
        print(f"解析文档块出错: {e}")
        # 出错时返回默认结构
        return {
            "summary": "解析失败",
            "takeaways": [],
            "concepts": [],
            "entities": {},
            "raw": chunk,
        }


def main():
    print("PDF to Markdown Converter initialized.")
    print("You can use the pdf_to_markdown() function to convert your PDFs.")
    # Example usage:
    pdf_path = "xxxxx-考试文档.pdf"
    if Path(pdf_path).exists():
        md_text, output_md_path = pdf_to_markdown(pdf_path)

        # 演示分割功能
        chunks = split_markdown_by_chapters(md_text, max_length=500)
        print(f"Document split into {len(chunks)} chunks.")

        import asyncio

        async def _run_parse():
            tasks = [parse_document(c) for c in chunks]
            return await asyncio.gather(*tasks)

        parsed_chunks = asyncio.run(_run_parse())

        # 将分割后的块也保存到同一个文件夹的chunks文件夹下
        output_dir = Path(output_md_path).parent / "chunks"
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, chunk_data in enumerate(parsed_chunks):
            chunk_file = output_dir / f"chunk_{i + 1}.md"
            raw_content = chunk_data.pop("raw", "")
            # Reorder fields: source first, summary second
            chunk_data["source"] = chunk_file.name
            ordered_data = {
                "source": chunk_data.pop("source"),
                "summary": chunk_data.pop("summary"),
            }
            ordered_data.update(chunk_data)

            frontmatter = yaml.dump(
                ordered_data,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            )
            with open(chunk_file, "w", encoding="utf-8") as f:
                f.write(f"---\n{frontmatter}---\n\n{raw_content}")
            print(
                f"  - Chunk {i + 1}: {len(raw_content)} characters, saved to {chunk_file}"
            )


if __name__ == "__main__":
    main()
