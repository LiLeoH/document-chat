import asyncio
from pathlib import Path

from dotenv import load_dotenv
import yaml
from pdf_processor import parse_document, pdf_to_markdown, split_markdown_by_chapters


def test_pdf_to_markdown():
    """简单的测试方法，直接参考 main() 中使用实际 PDF 进行测试"""
    test_pdf = "/Users/leo/Downloads/doc/xxxxx-考试文档.pdf"

    # 执行转换
    md_text, output_md_path = pdf_to_markdown(test_pdf)

    # 验证返回值不为空
    assert md_text is not None
    assert len(md_text) > 0
    assert output_md_path is not None

    # 验证 Markdown 文件确实被生成了
    assert Path(output_md_path).exists()

def test_split_markdown_by_chapters():
    """测试 split_markdown_by_chapters 函数的基本功能"""
    workspace = "4e5a1ea2"
    md_path = f"workspace/{workspace}/xxxxx-考试文档.md"
    workspace_dir = f"./workspace/{workspace}"

    # 读取markdown文件内容
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()  
    # 执行分割
    chunks = split_markdown_by_chapters(md_text, max_length=1000)
    # 写入workspace目录下的测试文件
    output_chunks_dir = Path(workspace_dir) / "chunks"
    output_chunks_dir.mkdir(parents=True, exist_ok=True)
    # 验证分割结果
    assert len(chunks) > 0

    load_dotenv()
    async def process_single_chunk(i, chunk_text):
        """Helper to process and insert a single chunk."""
        # Parse chunk for database (summary, entities, etc.)
        chunk_data = await parse_document(chunk_text)

        # Save chunk to Markdown file with YAML Frontmatter
        file_name = f"chunk_{i + 1}.md"
        chunk_file = output_chunks_dir / file_name

        # Add source to chunk metadata
        chunk_data["source"] = file_name

        # Create a copy for file saving
        save_data = chunk_data.copy()
        raw_content = save_data.pop("raw", "")

        # Reorder fields: source first, summary second
        ordered_save_data = {
            "source": save_data.pop("source"),
            "summary": save_data.pop("summary"),
        }
        ordered_save_data.update(save_data)

        frontmatter = yaml.dump(
            ordered_save_data,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )

        # File I/O is blocking, use to_thread
        await asyncio.to_thread(
            self_write_file, chunk_file, f"---\n{frontmatter}---\n\n{raw_content}"
        )

        return chunk_text, chunk_data

    def self_write_file(path, content):
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    async def run_all():
        semaphore = asyncio.Semaphore(5)

        async def sem_process_chunk(i, text):
            async with semaphore:
                return await process_single_chunk(i, text)

        tasks = [sem_process_chunk(i, text) for i, text in enumerate(chunks)]
        return await asyncio.gather(*tasks)

    results = asyncio.run(run_all())
