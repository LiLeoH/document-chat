import asyncio
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
from prompts import get_parse_document_prompt


def pdf_to_markdown(pdf_path: str, output_md_path: str | None = None) -> tuple[str, str]:
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


async def parse_document(chunk: str, extra_info: str = "", retry_count: int = 3) -> dict:
    """
    使用 LangChain 从大模型获取结构化解析结果。
    """
    llm = ChatOpenAI(
        model=os.environ.get("SUMMARY_MODEL"),
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
        temperature=0.1,
        streaming=True,
    )

    # 确保 extra_info 是字符串
    if extra_info is None:
        extra_info = ""

    prompt = get_parse_document_prompt(extra_info)

    structured_llm = llm.with_structured_output(DocumentMetadata)
    chain = prompt | structured_llm

    # 准备输入数据
    input_data = {
        "input": chunk,
    }

    try:
        # 打印实际 Prompt
        # formatted_prompt = prompt.invoke(input_data)
        # print(
        #     f"\n--- [DEBUG] Actual Prompt ---\n{formatted_prompt.to_string()}\n-----------------------------\n"
        # )

        # 执行 LLM 调用
        metadata_obj = await chain.ainvoke(input_data)

        if metadata_obj is None:
            raise ValueError("LLM 返回了空结果，解析失败。")

        # 转换为字典并添加原始文本
        result = metadata_obj.model_dump()
        result["raw"] = chunk
        return result
    except Exception as e:
        if retry_count > 0:
            print(f"解析文档块出错: {e}，1秒后重试 (剩余重试次数: {retry_count})...")
            await asyncio.sleep(1)
            return await parse_document(chunk, extra_info, retry_count - 1)
        
        print(f"解析文档块最终失败: {e}")
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
