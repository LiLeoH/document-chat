from retrieve import init_retriever
import os
import yaml
import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from pdf_processor import pdf_to_markdown, split_markdown_by_chapters, parse_document
from milvus_operator import MilvusOperator

# Load environment variables
load_dotenv()


async def process_pdf_to_milvus(
    pdf_path: str, extra_info: str = ""
) -> tuple[str, dict]:
    """
    Full pipeline: PDF -> Markdown -> Chunks -> Milvus (Asynchronous & Concurrent)
    """
    if not Path(pdf_path).exists():
        print(f"Error: File '{pdf_path}' not found.")
        return

    # Convert PDF to Markdown (This part is typically synchronous/blocking)
    md_text, output_md_path = await asyncio.to_thread(pdf_to_markdown, pdf_path)
    workspace_dir = Path(output_md_path).parent

    # Split Markdown into logical chunks
    chunks = split_markdown_by_chapters(
        md_text, max_length=int(os.environ.get("CHUNK_MAX_LENGTH", 3000))
    )
    print(f"Document split into {len(chunks)} chunks.")

    # Prepare output directory
    output_chunks_dir = workspace_dir / "chunks"
    output_chunks_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing and inserting {len(chunks)} chunks concurrently...")
    chunk_data_list = {}

    async def process_single_chunk(i, chunk_text):
        """Helper to process and insert a single chunk."""
        # Parse chunk for database (summary, entities, etc.)
        chunk_data = await parse_document(chunk_text, extra_info=extra_info)

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

        chunk_data_list[file_name] = chunk_data
        return chunk_text, chunk_data

    def self_write_file(path, content):
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    # Process all chunks concurrently using asyncio.gather
    semaphore = asyncio.Semaphore(5)

    async def sem_process_chunk(i, text):
        async with semaphore:
            return await process_single_chunk(i, text)

    tasks = [sem_process_chunk(i, text) for i, text in enumerate(chunks)]
    results = await asyncio.gather(*tasks)

    # Initialize Milvus Operator within the same workspace
    milvus_op = MilvusOperator(workspace=str(workspace_dir))
    print(f"All chunks processed. Now inserting {len(results)} chunks into Milvus...")
    for chunk_text, chunk_data in results:
        await milvus_op.insert_chunk(chunk_data)
        print(f"  - {chunk_data['source']} inserted into Milvus.")

    print("\nPipeline completed successfully!")
    print(f"Markdown outputs: {workspace_dir}")
    print(f"Milvus Database: {milvus_op.db_path}")
    await init_retriever(str(workspace_dir), milvus_op)
    return workspace_dir, chunk_data_list


async def load_from_workspace(workspace_dir: str) -> tuple[Path, dict]:
    """
    Load existing chunks and initialize retriever from a workspace.
    """
    workspace_path = Path(workspace_dir)
    chunks_dir = workspace_path / "chunks"

    if not chunks_dir.exists():
        raise FileNotFoundError(f"Workspace chunks directory not found: {chunks_dir}")

    chunk_data_list = {}
    print(f"Loading chunks from {chunks_dir}...")

    # Read all .md files in the chunks directory
    for chunk_file in sorted(chunks_dir.glob("*.md")):
        with open(chunk_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse YAML frontmatter
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter_str = parts[1]
                raw_content = parts[2].strip()

                chunk_data = yaml.safe_load(frontmatter_str)
                chunk_data["raw"] = raw_content
                chunk_data_list[chunk_file.name] = chunk_data

    # Initialize Milvus Operator
    milvus_op = MilvusOperator(workspace=str(workspace_path))

    # Initialize Retriever
    await init_retriever(str(workspace_path), milvus_op)

    print(f"Loaded {len(chunk_data_list)} chunks. Workspace ready.")
    return workspace_path, chunk_data_list


if __name__ == "__main__":
    pdf_file = "xxxxx-考试文档.pdf"
    try:
        asyncio.run(process_pdf_to_milvus(pdf_file))
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
