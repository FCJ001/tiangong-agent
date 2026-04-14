import uuid

import pytest

from agents.supervisor_agent import chat_endpoint

# @pytest.mark.asyncio
async def test_supervisor_agent():
    result = await chat_endpoint("1234","AEEEEA", "你好，我是雷丰阳")
    print(result)


async def test_agent_memory():
    """验证短期记忆：同一 thread_id 下第二轮能记住第一轮的内容"""
    # 第一轮：自我介绍
    reply1 = await chat_endpoint("1123", "ATDAAS", "你好，我叫雷丰阳")
    print(f"\n第一轮回复：{reply1}")

    # 第二轮：考察记忆
    reply2 = await chat_endpoint("1123", "ATDAAS", "我是谁？")
    print(f"第二轮回复：{reply2}")

    # 断言：第二轮回复应包含名字
    assert "雷丰阳" in reply2, f"Agent 应该记住用户名字，实际回复：{reply2}"

    # 第三轮：新会话
    reply3 = await chat_endpoint("1123", "7783AAA", "我是谁？")
    print(f"第三轮回复：{reply3}")


def new_session() -> str:
    """每次测试生成唯一 session_id，防止不同测试/不同执行轮次的 checkpoint 互相污染。"""
    return uuid.uuid4().hex[:8]


async def test_agent_store():
    """验证长期记忆：跨会话记忆
    第一个 session 写入病史 → 第二个全新 session 能检索到
    """
    # ── 第一轮：用新会话写入病史 ──────────────────────────────────────────
    session_1 = new_session()
    resp1 = await chat_endpoint("2244", session_1, "你好，我叫张三，有新冠病史")
    print(f"\n[写入] 回复：{resp1}")
    resp1 = await chat_endpoint("2244", session_1, "我对樱桃过敏")
    print(f"\n[写入] 回复：{resp1}")
    resp1 = await chat_endpoint("2244", session_1, "我对樱桃过敏")

    # ── 第二轮：换一个全新会话，查询病史（跨会话长期记忆） ───────────────
    session_2 = new_session()  # 与 session_1 不同，短期记忆里没有上文
    resp2 = await chat_endpoint("2244", session_2, "我有什么病史和过敏史？")
    print(f"[检索] 回复：{resp2}")

    assert "糖尿病" in resp2, f"长期记忆应能跨会话检索到糖尿病史，实际回复：{resp2}"
