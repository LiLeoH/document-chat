import gradio as gr
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from main import process_pdf_to_milvus, load_from_workspace
from retrieve import rag_flow, reset_agent

# Load environment variables from .env file
load_dotenv()

# Store the active workspace in a global-ish state (or gr.State)
# For simplicity in this demo, we can use gr.State to hold the workspace path per session.


async def upload_file(file, extra_info: str):
    if file is None:
        return "请先上传 PDF 文件。", None

    try:
        # process_pdf_to_milvus returns chunk_data_list, but we need the workspace path.
        # Let's modify main.py slightly or just extract the path here.
        # Actually, main.py already prints the workspace.
        # I'll update main.py to return (workspace_path, chunk_data_list)

        print(f"正在处理文件: {file.name}")
        # Gradio provides a temp file path in file.name
        # We need to make sure we pass a string path
        chunk_data = await process_pdf_to_milvus(file.name, extra_info=extra_info)

        # Since process_pdf_to_milvus doesn't return the workspace path directly yet,
        # but it's generated inside. I'll need to update main.py to return it.
        # For now, I'll assume I'll update main.py in the next step.

        # Let's peek into the workspace directory to find the latest one if needed,
        # but it's better to be explicit.

        return (
            f"文件 '{Path(file.name).name}' 处理完成！现在可以开始提问了。",
            chunk_data,
        )
    except Exception as e:
        return f"处理出错: {str(e)}", None


def list_workspaces():
    workspace_root = Path("workspace")
    if not workspace_root.exists():
        return []

    choices = []
    # 遍历 workspace 目录下的子目录
    for d in workspace_root.iterdir():
        if d.is_dir():
            # 在子目录下查找 .md 文件（排除 chunks 目录下的）
            md_files = [f for f in d.glob("*.md") if f.is_file()]
            if md_files:
                # 取第一个 md 文件名作为显示标签
                label = md_files[0].stem
                choices.append((label, d.name))
            else:
                # 兜底：如果没有 md 文件，显示文件夹名
                choices.append((d.name, d.name))

    # 按标签排序
    return sorted(choices, key=lambda x: x[0], reverse=True)


async def load_existing_workspace(workspace_name):
    if not workspace_name:
        return "请先选择一个项目。", None

    try:
        workspace_path = Path("workspace") / workspace_name
        print(f"正在加载项目: {workspace_name}")
        chunk_data = await load_from_workspace(str(workspace_path))
        return f"项目 '{workspace_name}' 加载成功！现在可以开始提问了。", chunk_data
    except Exception as e:
        return f"加载失败: {str(e)}", None


def user(message, history):
    if not message:
        return "", history
    return "", history + [{"role": "user", "content": message}]


async def bot(history, chunk_data_list):
    if not history or history[-1]["role"] != "user":
        yield history
        return

    message = history[-1]["content"]

    if not chunk_data_list:
        history.append({"role": "assistant", "content": "请先在左侧上传或选择一个项目。"})
        yield history
        return

    workspace_dir, _ = chunk_data_list

    try:
        # Add a jumping dots placeholder while waiting for the first token
        history.append({
            "role": "assistant", 
            "content": '<div class="thinking-dots"><span></span><span></span><span></span></div>'
        })
        yield history

        has_content = False
        # Generate response using RAG flow
        async for chunk in rag_flow(message, str(workspace_dir)):
            if not has_content:
                # Clear the placeholder when the first chunk of text arrives
                history[-1]["content"] = ""
                has_content = True
            history[-1]["content"] += chunk
            yield history
    except Exception as e:
        if history[-1]["content"] == "" or 'thinking-dots' in history[-1]["content"]:
            history[-1]["content"] = f"检索出错: {str(e)}"
        else:
            history[-1]["content"] += f"\n\n检索出错: {str(e)}"
        yield history


def clear_chat(workspace_state):
    if workspace_state:
        workspace_dir, _ = workspace_state
        reset_agent(str(workspace_dir))
    return []


CUSTOM_CSS = """
.main-header {
    text-align: center;
    padding: 1.5rem 0 0.5rem;
}
.main-header h1 {
    font-size: 2rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.25rem;
}
.main-header p {
    color: #6b7280;
    font-size: 0.95rem;
    max-width: 640px;
    margin: 0 auto;
}
.status-bar textarea {
    border-left: 3px solid #667eea !important;
    background: #f8f9ff !important;
    font-weight: 500;
}
.chat-input textarea {
    border-radius: 12px !important;
    padding: 12px 16px !important;
}
.send-btn {
    border-radius: 12px !important;
    height: 46px !important;
    font-size: 1rem !important;
}
.clear-btn {
    border-radius: 12px !important;
    height: 46px !important;
}
.thinking-dots {
    display: flex;
    gap: 4px;
    padding: 8px 0;
}
.thinking-dots span {
    width: 6px;
    height: 6px;
    background-color: #667eea;
    border-radius: 50%;
    display: inline-block;
    animation: dot-bounce 1.4s infinite ease-in-out both;
}
.thinking-dots span:nth-child(1) { animation-delay: -0.32s; }
.thinking-dots span:nth-child(2) { animation-delay: -0.16s; }

@keyframes dot-bounce {
    0%, 80%, 100% { transform: scale(0); opacity: 0.3; }
    40% { transform: scale(1.0); opacity: 1; }
}
"""


APP_THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.purple,
    neutral_hue=gr.themes.colors.gray,
    font=gr.themes.GoogleFont("Inter"),
)


def build_app():
    with gr.Blocks(title="项目文档智能助手") as demo:
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>📄 项目文档智能问答助手</h1>
            <p>上传 PDF 文件或加载已解析的项目，系统自动完成解析与向量化，随后即可针对文档内容智能问答</p>
        </div>
        """)

        workspace_state = gr.State(None)

        with gr.Row(equal_height=False):
            # ── Left Panel ──
            with gr.Column(scale=1, min_width=360):
                with gr.Tabs(selected="upload"):
                    # Tab 1: Load existing
                    with gr.Tab("📂 加载已有项目", id="load"):
                        workspace_dropdown = gr.Dropdown(
                            choices=list_workspaces(),
                            label="选择已解析的文档",
                            interactive=True,
                        )
                        with gr.Row():
                            load_workspace_btn = gr.Button(
                                "🚀 加载", variant="primary", scale=3
                            )
                            refresh_btn = gr.Button("🔄 刷新", scale=1)

                    # Tab 2: Process new
                    with gr.Tab("📤 处理新文档", id="upload"):
                        file_input = gr.File(
                            label="上传 PDF",
                            file_types=[".pdf"],
                            file_count="single",
                        )
                        extra_info_input = gr.Textbox(
                            label="补充提取关键信息（可选）",
                            placeholder="每行一项，例如：\n工程名称\n技术方案\n预算",
                            lines=4,
                            info="这些关键词将动态注入到 LLM 中，优化对内容的提取",
                        )
                        upload_button = gr.Button(
                            "✨ 开始处理", variant="primary"
                        )

                status_output = gr.Textbox(
                    label="📋 状态",
                    interactive=False,
                    elem_classes=["status-bar"],
                )

            # ── Right Panel ──
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="💬 智能问答",
                    height=560,
                    placeholder="请先在左侧加载或上传文档，然后开始提问 💡",
                )
                with gr.Row():
                    query_input = gr.Textbox(
                        show_label=False,
                        placeholder="输入您的问题，按 Enter 发送...",
                        scale=5,
                        container=False,
                        elem_classes=["chat-input"],
                    )
                    send_button = gr.Button(
                        "发送", variant="primary", scale=1, elem_classes=["send-btn"]
                    )
                    clear_button = gr.Button(
                        "🗑️ 清空", scale=1, elem_classes=["clear-btn"]
                    )

        # ── Event handlers ──
        refresh_btn.click(
            fn=lambda: gr.update(choices=list_workspaces()),
            inputs=[],
            outputs=[workspace_dropdown],
        )

        load_workspace_btn.click(
            fn=load_existing_workspace,
            inputs=[workspace_dropdown],
            outputs=[status_output, workspace_state],
        )

        upload_button.click(
            fn=upload_file,
            inputs=[file_input, extra_info_input],
            outputs=[status_output, workspace_state],
        )

        query_input.submit(
            fn=user,
            inputs=[query_input, chatbot],
            outputs=[query_input, chatbot],
            queue=False,
        ).then(
            fn=bot,
            inputs=[chatbot, workspace_state],
            outputs=[chatbot],
        )

        send_button.click(
            fn=user,
            inputs=[query_input, chatbot],
            outputs=[query_input, chatbot],
            queue=False,
        ).then(
            fn=bot,
            inputs=[chatbot, workspace_state],
            outputs=[chatbot],
        )

        clear_button.click(
            fn=clear_chat,
            inputs=[workspace_state],
            outputs=[chatbot],
            queue=False,
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=16789, theme=APP_THEME, css=CUSTOM_CSS)
