"""
RAG 测试界面 — 通过 HTTP 调用后端 FastAPI 接口。
启动: python test/test_rag_gradio.py
前提: 后端 uvicorn src.main:app 已启动
"""

from __future__ import annotations
import json
from pathlib import Path

import gradio as gr
import httpx

# 后端地址
API_BASE = "http://localhost:8000"


# ── Tab1: RAG 检索 ─────────────────────────────────────────────────────

async def rag_search(
    question: str,
    channel: str,
    doc_type: str,
    use_hyde: bool,
) -> tuple[str, str]:
    if not question.strip():
        return "请输入问题", ""

    async with httpx.AsyncClient(timeout=120) as client:
        try:
            resp = await client.post(f"{API_BASE}/api/v1/knowledge/search", json={
                "question": question,
                "channel": channel,
                "doc_type": doc_type or "",
                "use_hyde": use_hyde,
            })
            resp.raise_for_status()
            data = resp.json()

            answer = data["answer"]
            elapsed = data.get("elapsed_ms", 0)
            answer += f"\n\n---\n⏱ 耗时: {elapsed/1000:.1f}s | 通道: {channel}"

            # 格式化原始命中
            raw_display = ""
            raw_hits = data.get("raw_hits", [])
            raw_graph = data.get("raw_graph", [])
            if raw_hits:
                raw_display += f"═══ 文档检索 ({len(raw_hits)} 条) ═══\n"
                raw_display += _format_hits(raw_hits)
            if raw_graph:
                if raw_display:
                    raw_display += "\n\n"
                raw_display += f"═══ 图谱检索 ({len(raw_graph)} 条) ═══\n"
                raw_display += json.dumps(raw_graph, ensure_ascii=False, indent=2)
            if not raw_display and channel == "prescription":
                raw_display = "（处方审核走完整管线：解析 → 剂量/配伍/过敏/重复并行校验 → 报告生成）"

            return answer, raw_display

        except httpx.HTTPError as e:
            return f"请求失败: {e}", ""


def _format_hits(hits: list[dict]) -> str:
    if not hits:
        return "无检索命中"
    lines = []
    for i, h in enumerate(hits, 1):
        score = h.get("score", 0)
        lines.append(
            f"[{i}] score={score:.4f} | {h.get('doc_name', '?')} "
            f"| page={h.get('page_number', '?')} | chunk={h.get('chunk_index', '?')}\n"
            f"    text: {h.get('text', '')[:200]}..."
        )
    return "\n".join(lines)


# ── Tab2: 文档上传 ──────────────────────────────────────────────────────

async def upload_document(file, doc_type: str, category: str):
    if file is None:
        return "请选择文件"

    file_path = Path(file)
    ext = file_path.suffix.lower()
    allowed = {".pdf", ".docx", ".doc", ".txt", ".md"}
    if ext not in allowed:
        return f"不支持的文件格式: {ext}，支持: {allowed}"

    async with httpx.AsyncClient(timeout=300) as client:
        try:
            content = file_path.read_bytes()
            resp = await client.post(
                f"{API_BASE}/api/v1/knowledge/upload",
                files={"file": (file_path.name, content)},
                data={"doc_type": doc_type, "category": category},
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", f"上传成功，{data.get('chunks', 0)} 个分块")
        except httpx.HTTPError as e:
            return f"上传失败: {e}"


# ── Tab3: 文档管理 ──────────────────────────────────────────────────────

async def list_documents():
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.get(f"{API_BASE}/api/v1/knowledge/docs")
            resp.raise_for_status()
            data = resp.json()
            docs = data.get("documents", [])
            if not docs:
                return "知识库为空"
            lines = []
            for r in docs:
                lines.append(
                    f"[{r.get('doc_type', '?')}] {r.get('doc_name', '?')} "
                    f"— {r.get('category', '?')}"
                )
            return "\n".join(lines)
        except httpx.HTTPError as e:
            return f"查询失败: {e}"


async def delete_document(doc_name: str):
    if not doc_name.strip():
        return "请输入文档名"
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.delete(
                f"{API_BASE}/api/v1/knowledge/docs/{doc_name}"
            )
            resp.raise_for_status()
            return resp.json().get("message", "删除成功")
        except httpx.HTTPError as e:
            return f"删除失败: {e}"


# ── UI ───────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(title="Tiangong RAG 测试") as demo:
        gr.Markdown("# Tiangong Medical RAG 测试平台")

        with gr.Tabs():
            with gr.Tab("RAG 检索"):
                with gr.Row():
                    with gr.Column(scale=1):
                        channel = gr.Radio(
                            choices=["doc_rag", "graph_rag", "multi", "prescription"],
                            value="doc_rag",
                            label="检索通道",
                        )
                        doc_type = gr.Dropdown(
                            choices=["", "guideline", "drug_instruction", "sop", "literature"],
                            value="",
                            label="文档类型过滤 (仅 doc_rag 有效)",
                        )
                        use_hyde = gr.Checkbox(value=True, label="启用 HyDE 增强")
                        gr.Markdown("---\n**通道说明**\n"
                            "- **doc_rag**: 文档向量检索 (HyDE + Milvus + Reranker)\n"
                            "- **graph_rag**: 知识图谱检索 (实体提取 → NL2Cypher)\n"
                            "- **multi**: 多通道并行检索 (doc + graph)\n"
                            "- **prescription**: 处方审核 (剂量/配伍/过敏/重复)")

                    with gr.Column(scale=2):
                        question = gr.Textbox(
                            label="问题",
                            placeholder="输入医学问题，如：高血压患者日常生活中需要注意什么？",
                            lines=3,
                        )
                        submit_btn = gr.Button("检索", variant="primary")
                        answer = gr.Textbox(label="生成回答", lines=12)

                with gr.Accordion("原始检索结果", open=False):
                    raw_output = gr.Textbox(
                        label="Hit Details",
                        lines=20,
                        elem_classes=["raw-output"],
                    )

                submit_btn.click(
                    fn=rag_search,
                    inputs=[question, channel, doc_type, use_hyde],
                    outputs=[answer, raw_output],
                )

            with gr.Tab("文档上传"):
                with gr.Row():
                    with gr.Column():
                        file_input = gr.File(
                            label="选择文档 (PDF/Word/TXT/MD)",
                            file_types=[".pdf", ".docx", ".doc", ".txt", ".md"],
                        )
                        upload_doc_type = gr.Dropdown(
                            choices=["guideline", "drug_instruction", "sop", "literature"],
                            value="guideline",
                            label="文档类型",
                        )
                        upload_category = gr.Textbox(value="通用", label="分类")
                        upload_btn = gr.Button("上传到知识库", variant="primary")
                    with gr.Column():
                        upload_result = gr.Textbox(label="上传结果", lines=5)

                upload_btn.click(
                    fn=upload_document,
                    inputs=[file_input, upload_doc_type, upload_category],
                    outputs=[upload_result],
                )

            with gr.Tab("文档管理"):
                with gr.Row():
                    with gr.Column():
                        refresh_btn = gr.Button("刷新文档列表", variant="primary")
                        doc_list = gr.Textbox(label="知识库文档", lines=15)
                    with gr.Column():
                        delete_name = gr.Textbox(label="要删除的文档名（完整文件名）")
                        delete_btn = gr.Button("删除文档", variant="stop")
                        delete_result = gr.Textbox(label="删除结果")

                refresh_btn.click(fn=list_documents, inputs=[], outputs=[doc_list])
                delete_btn.click(fn=delete_document, inputs=[delete_name], outputs=[delete_result])

        return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
        css="""
        .raw-output textarea { font-size: 12px !important; font-family: monospace !important; }
        footer { display: none !important; }
        """,
    )
