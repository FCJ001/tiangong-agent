"""消息修复中间件：清理 SummarizationMiddleware 可能产生的孤立 tool_calls/tool_response。

DeepSeek API（以及其他 OpenAI 兼容 API）严格要求：
AIMessage(tool_calls) 后必须紧跟对应的 ToolMessage(tool_call_id)，
否则返回 400 错误。

SummarizationMiddleware 在压缩对话历史时可能破坏这种配对关系，
本中间件在其后运行，确保发送给模型的消息列表满足配对要求。
"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ResponseT
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langgraph.runtime import Runtime


class ToolCallRepairMiddleware(AgentMiddleware[AgentState[Any], Any, ResponseT]):
    """修复 SummarizationMiddleware 可能导致的 tool call/response 配对断裂。

    在 before_model 中扫描消息列表：
    1. 移除无法匹配到 AIMessage(tool_calls) 的 ToolMessage（孤儿响应）
    2. 清理 AIMessage 中无法匹配到 ToolMessage 的 tool_calls 条目（孤儿调用）
    3. 如果 AIMessage 的所有 tool_calls 都是孤儿，移除 tool_calls 字段
    """

    def _repair_messages(self, messages: list[AnyMessage]) -> list[AnyMessage]:
        """扫描并修复消息列表中的 tool call/response 配对问题。"""

        # 第一遍：收集所有有效的 tool_call_id
        ai_tool_call_ids: set[str] = set()
        tool_response_ids: set[str] = set()

        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    tc_id = tc.get("id")
                    if tc_id:
                        ai_tool_call_ids.add(tc_id)
            elif isinstance(msg, ToolMessage):
                if msg.tool_call_id:
                    tool_response_ids.add(msg.tool_call_id)

        # 找出孤儿
        orphan_tool_response_ids = tool_response_ids - ai_tool_call_ids
        orphan_tool_call_ids = ai_tool_call_ids - tool_response_ids

        if not orphan_tool_response_ids and not orphan_tool_call_ids:
            return messages  # 无需修复

        # 第二遍：构建修复后的消息列表
        repaired: list[AnyMessage] = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                # 移除无法匹配到 AIMessage 的 ToolMessage
                if msg.tool_call_id and msg.tool_call_id in orphan_tool_response_ids:
                    continue

            elif isinstance(msg, AIMessage) and msg.tool_calls:
                # 过滤掉孤立的 tool_calls 条目
                surviving_tool_calls = [
                    tc for tc in msg.tool_calls
                    if tc.get("id") and tc["id"] not in orphan_tool_call_ids
                ]
                if surviving_tool_calls:
                    # 部分 tool_calls 有效 → 替换
                    repaired.append(
                        AIMessage(
                            content=msg.content,
                            tool_calls=surviving_tool_calls,
                            id=msg.id,
                            additional_kwargs=msg.additional_kwargs,
                            response_metadata=msg.response_metadata,
                        )
                    )
                    continue
                else:
                    # 所有 tool_calls 都是孤儿 → 保留消息但去除 tool_calls
                    repaired.append(
                        AIMessage(
                            content=msg.content or "",
                            id=msg.id,
                            additional_kwargs={
                                k: v for k, v in msg.additional_kwargs.items()
                                if k != "tool_calls"
                            },
                            response_metadata=msg.response_metadata,
                        )
                    )
                    continue

            repaired.append(msg)

        return repaired

    def before_model(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        messages = state.get("messages", [])
        repaired = self._repair_messages(messages)
        if repaired is not messages:
            return {"messages": repaired}
        return None

    async def abefore_model(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        return self.before_model(state, runtime)
