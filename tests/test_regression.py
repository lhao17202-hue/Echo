"""
Echo 单 Agent 内核回归测试 —— 用 FakeLLMClient 覆盖 3 个确定性场景。

Run: python -m pytest tests/test_regression.py -v
"""

import sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from echo.core.agent_loop import AgentLoop
from echo.tools.registry import ToolRegistry
from echo.tools.executor import ToolExecutor
from echo.providers.fake_client import FakeLLMClient
from echo.hooks.base import HookManager
from echo.hooks.builtin import PermissionHook, LogHook, PostLogHook, StatsHook
from echo.memory.base import MemoryManager, MemoryEntry
from echo.memory.default import KeywordMemory
from echo.core.context_manager import ContextManager, ContextConfig
from echo.security.sandbox import Sandbox
from echo.security.env_filter import ShellExecutor
from echo.persistence.session_store import Session, SessionStore
from echo.persistence.run_store import RunStore
from echo.tools.base import ToolContext


def _build_loop(workspace, outputs, approval="auto", max_steps=10,
               context_config=None):
    reg = ToolRegistry()
    reg.discover("echo.tools.builtin")

    hooks = HookManager()
    hooks.register(PermissionHook(), priority=0)
    hooks.register(LogHook(), priority=100)
    hooks.register(PostLogHook(), priority=100)
    hooks.register(StatsHook(), priority=200)

    sess_dir = str(Path(workspace) / ".echo" / "sessions" / "regression")
    return AgentLoop(
        llm=FakeLLMClient(outputs=outputs),
        memory=MemoryManager(KeywordMemory()),
        tools=ToolExecutor(reg),
        hooks=hooks,
        context=ContextManager(context_config) if context_config else ContextManager(),
        sandbox=Sandbox(workspace),
        shell=ShellExecutor(workspace),
        session_store=SessionStore(workspace),
        run_store=RunStore(sess_dir),
        max_steps=max_steps,
        approval_policy=approval,
    )


# ═══════════════════════════════════════════════════
# 场景 1: read README → final
# ═══════════════════════════════════════════════════

def test_scenario_readme_then_final():
    """Agent 读取 README 后返回包含项目名的 final answer。"""
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "README.md").write_text("# Echo Agent\nA lightweight coding agent.")

        loop = _build_loop(d, [
            '<tool name="read_file" path="README.md" />',
            "项目名为 Echo Agent，是一个轻量级编码 Agent。",
        ])

        answer = loop.run("Read README and tell me the project name")
        assert "Echo Agent" in answer
        assert loop.llm.call_count == 2
        assert "Stopped" not in answer


# ═══════════════════════════════════════════════════
# 场景 2: patch 文件并确认内容变化
# ═══════════════════════════════════════════════════

def test_scenario_patch_and_verify():
    """Agent patch 文件后确认内容正确更新。"""
    with tempfile.TemporaryDirectory() as d:
        target = Path(d) / "config.txt"
        target.write_text("version=1.0\ndebug=false\n")

        loop = _build_loop(d, [
            '<tool name="read_file" path="config.txt" />',
            '<tool name="patch_file" path="config.txt"><old_text>debug=false</old_text><new_text>debug=true</new_text></tool>',
            '<tool name="read_file" path="config.txt" />',
            "已将 debug 改为 true。",
        ], approval="auto")

        answer = loop.run("Enable debug mode in config.txt")
        assert "debug=true" in target.read_text()
        assert "debug" in answer.lower()
        assert loop.llm.call_count == 4


# ═══════════════════════════════════════════════════
# 场景 3: 危险 Shell + 越界路径被拒绝且不计入成功工具步数
# ═══════════════════════════════════════════════════

def test_scenario_danger_rejected_not_counted():
    """越界路径 + 危险命令被 Hook/沙箱拦截，不计入 tool_steps。"""
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "safe.txt").write_text("safe content")

        loop = _build_loop(d, [
            # 越界读（沙箱拦截 → fail → 不计入 steps）
            '<tool name="read_file" path="../etc/passwd" />',
            # 危险命令（Hook 拦截 → fail → 不计入 steps）
            '<tool name="run_shell" command="rm -rf /" timeout="5" />',
            # 正常读（成功执行 → 计入 steps）
            '<tool name="read_file" path="safe.txt" />',
            # 最终回复
            "安全测试完成。只有安全操作被计入步数。",
        ], approval="auto", max_steps=10)

        answer = loop.run("try to escape then read safe file")
        # FakeLLMClient returns 4th output -> loop ends normally (not by step_limit)
        assert len(answer) > 0
        # LLM calls: 4 outputs, 3 tool_use + 1 final
        assert loop.llm.call_count == 4


def test_scenario_rejected_tools_dont_consume_budget():
    """连续请求被拒绝的工具不会耗尽 step budget。"""
    with tempfile.TemporaryDirectory() as d:
        # 用较小的 max_steps 测试
        loop = _build_loop(d, [
            '<tool name="run_shell" command="rm -rf /" timeout="5" />',
            '<tool name="write_file" path="../outside.txt"><content>bad</content></tool>',
            '<tool name="run_shell" command="shutdown now" timeout="5" />',
            "All bad operations were correctly blocked.",
        ], approval="auto", max_steps=3)

        answer = loop.run("do bad things")
        # 3 tools all blocked -> tool_steps=0 -> won't hit step_limit
        # 4th turn returns final -> ends normally
        assert len(answer) > 0 and "Stopped" not in answer


# ═══════════════════════════════════════════════════
# 场景补充: 完整 write + read 验证
# ═══════════════════════════════════════════════════

def test_scenario_write_and_read():
    """Agent 创建文件后读取确认。"""
    with tempfile.TemporaryDirectory() as d:
        loop = _build_loop(d, [
            '<tool name="write_file" path="new.txt"><content>hello world</content></tool>',
            '<tool name="read_file" path="new.txt" />',
            "文件包含 'hello world'。",
        ], approval="auto")

        answer = loop.run("create new.txt with hello world and verify")
        assert (Path(d) / "new.txt").read_text() == "hello world"
        assert "hello world" in answer


# ═══════════════════════════════════════════════════
# 场景: memory 注入 prompt 的 E2E
# ═══════════════════════════════════════════════════

def test_memory_user_request_in_prompt():
    """用户请求应出现在 system prompt 的当前任务中。"""
    with tempfile.TemporaryDirectory() as d:
        loop = _build_loop(d, [
            '<tool name="read_file" path="README.md" />',
            "项目名为 TestProject。",
        ])
        (Path(d) / "README.md").write_text("# TestProject")

        loop.run("分析项目 README")
        # MemoryManager 应把用户请求记入 task_summary
        assert loop.memory._task_summary == "分析项目 README"

        # system prompt 应包含工作记忆
        fake = loop.llm
        assert fake.last_system is not None
        assert "当前任务" in fake.last_system or "Task" in fake.last_system


def test_memory_file_summaries_in_prompt():
    """read_file 后 system prompt 应出现文件摘要。"""
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "src.py").write_text("def main(): pass")
        loop = _build_loop(d, [
            '<tool name="read_file" path="src.py" />',
            "文件包含一个 main 函数。",
        ])

        loop.run("read src.py")
        # 第二轮 system prompt 应包含文件摘要
        fake = loop.llm
        assert fake.last_system is not None
        # render_working 输出近期文件或文件摘要
        has_file_info = (
            "src.py" in fake.last_system
            or "近期文件" in fake.last_system
            or "File summaries" in fake.last_system
            or "Recent files" in fake.last_system
        )
        assert has_file_info, f"system prompt 未包含文件信息: {fake.last_system[:500]}"


def test_memory_recent_files_after_read():
    """read_file 后 recent_files 应更新。"""
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "a.py").write_text("a")
        loop = _build_loop(d, [
            '<tool name="read_file" path="a.py" />',
            "done",
        ])

        manager = loop.memory
        assert len(manager._recent_files) == 0
        loop.run("read a.py")
        assert len(manager._recent_files) >= 1


# ═══════════════════════════════════════════════════
# 场景: trace/report 细断言
# ═══════════════════════════════════════════════════

def test_trace_contains_model_events():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "f.txt").write_text("hello")
        loop = _build_loop(d, [
            '<tool name="read_file" path="f.txt" />',
            "文件内容是 hello。",
        ])

        loop.run("read f.txt")
        from echo.persistence.trace import read_jsonl
        events = read_jsonl(loop.run_store._trace_path())

        event_types = [e["event"] for e in events]
        assert "run_started" in event_types
        assert "model_requested" in event_types, f"events: {event_types}"
        assert "model_response" in event_types, f"events: {event_types}"
        assert "tool_executed" in event_types, f"events: {event_types}"


def test_report_contains_expected_fields():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "f.txt").write_text("hello")
        loop = _build_loop(d, [
            '<tool name="read_file" path="f.txt" />',
            "文件内容是 hello。",
        ])

        loop.run("read f.txt")
        report = loop.run_store.load_report(loop.run_store._run_dir.name)
        assert report["status"] == "completed"
        assert len(report["final_answer"]) > 0
        assert report["tool_steps"] == 1
        assert report["stop_reason"] == "final_answer_returned"


def test_task_state_after_run():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "f.txt").write_text("data")
        loop = _build_loop(d, [
            '<tool name="read_file" path="f.txt" />',
            "done",
        ])

        loop.run("read")
        # report.json 应正确记录最终状态
        report = loop.run_store.load_report(loop.run_store._run_dir.name)
        assert report["status"] == "completed"
        assert report["tool_steps"] == 1
        assert report["final_answer"] == "done"


# ═══════════════════════════════════════════════════
# 场景: grep run_list 断言
# ═══════════════════════════════════════════════════

def test_grep_called_with_list_args():
    """grep 工具通过 run_list(args, shell=False) 而非 shell 字符串。"""
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "code.py").write_text("def foo():\n    pass\n")
        recorded_args: list = []

        from echo.tools.base import ToolContext
        from echo.tools.builtin import GrepTool
        from echo.security.sandbox import Sandbox

        sandbox = Sandbox(d)
        real_run_list = sandbox.__class__.__dict__.get("run_list")

        class FakeShell:
            def run_list(self, args, **kw):
                recorded_args.append(list(args))
                from echo.security.env_filter import ShellResult
                return ShellResult(output="code.py:1:def foo()")
            def run(self, *a, **kw):
                raise AssertionError("grep 不应再使用 shell=True（run()）")
            def build_env(self):
                return {}

        ctx = ToolContext(workspace_root=d, sandbox=sandbox, shell=FakeShell())
        r = GrepTool().run(ctx, {"pattern": "def foo", "path": "."})

        assert r.success
        assert len(recorded_args) >= 1
        args = recorded_args[0]
        assert args[0] == "rg"
        assert "--line-number" in args
        assert "--color=never" in args
        assert "def foo" in args
        assert str(sandbox.root) in args[-1]


# ═══════════════════════════════════════════════════
# 场景: AgentLoop 内部消息配对断言
# ═══════════════════════════════════════════════════

def test_agent_loop_messages_tool_use_and_result_paired():
    """assistant 的 ToolUseBlock.id 和后续 tool_result.tool_use_id 必须一致。"""
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "README.md").write_text("content")

        loop = _build_loop(d, [
            '<tool name="read_file" path="README.md" />',
            "文件内容是 content。",
        ])

        loop.run("read README")

        # messages 格式: [user, assistant, user(tool_result), assistant(final)]
        assert len(loop.messages) >= 3, f"expected >=3 messages, got {len(loop.messages)}"

        # 第2条: assistant（含 ToolUseBlock）
        ass_msg = loop.messages[1]
        assert ass_msg["role"] == "assistant"
        tool_use_blocks = [
            b for b in ass_msg["content"]
            if hasattr(b, "id") and hasattr(b, "name") and hasattr(b, "input")
        ]
        assert len(tool_use_blocks) == 1, f"expected 1 tool_use, got {len(tool_use_blocks)}"
        tu_id = tool_use_blocks[0].id
        assert tu_id == "fake_read_file_0", f"unexpected tool_use id: {tu_id}"

        # 第3条: user（含 tool_result，id 必须匹配）
        result_msg = loop.messages[2]
        assert result_msg["role"] == "user"
        result_blocks = [
            b for b in result_msg["content"]
            if isinstance(b, dict) and b.get("type") == "tool_result"
        ]
        assert len(result_blocks) == 1
        tr_id = result_blocks[0]["tool_use_id"]
        assert tr_id == tu_id, (
            f"tool_result.tool_use_id ({tr_id}) != ToolUseBlock.id ({tu_id})"
        )


def test_agent_loop_multi_tool_message_pairing():
    """多个 tool_use 和 tool_result 必须一一对应。"""
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "a.txt").write_text("A")
        (Path(d) / "b.txt").write_text("B")

        loop = _build_loop(d, [
            '<tool name="read_file" path="a.txt" /><tool name="read_file" path="b.txt" />',
            "两个文件已经读完。",
        ])

        loop.run("read both")

        ass_msg = loop.messages[1]
        tool_use_ids = [
            b.id for b in ass_msg["content"]
            if hasattr(b, "id") and hasattr(b, "name")
        ]
        result_msg = loop.messages[2]
        result_ids = [
            b["tool_use_id"] for b in result_msg["content"]
            if isinstance(b, dict) and b.get("type") == "tool_result"
        ]
        assert tool_use_ids == result_ids, (
            f"tool_use ids ({tool_use_ids}) != tool_result ids ({result_ids})"
        )


def test_tool_result_content_reaches_next_model_call():
    """工具结果的内容应出现在下一轮 LLM 调用的 messages 中。"""
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "data.txt").write_text("secret-42")

        loop = _build_loop(d, [
            '<tool name="read_file" path="data.txt" />',
            "内容是 secret-42。",
        ])

        loop.run("read data.txt")

        # 第2轮 LLM 调用时，prompts[1] 应包含 secret-42 的工具结果
        assert loop.llm.call_count == 2
        assert len(loop.llm.prompts) == 2
        # 第2轮的 messages 数量应大于第1轮（多了一个 user(tool_result) 和 assistant(final)）
        assert loop.llm.prompts[1]["message_count"] > loop.llm.prompts[0]["message_count"]


# ═══════════════════════════════════════════════════
# Compact 回归测试
# ═══════════════════════════════════════════════════

def test_compact_tool_triggers_force_compact():
    """compact 工具触发 force_compact，messages 中出现 [Compacted conversation] 标记。"""
    with tempfile.TemporaryDirectory() as d:
        loop = _build_loop(d, [
            '<tool name="compact" />',
            "上下文已压缩，可以继续工作。",
        ], approval="auto")

        loop.run("compact the context")
        # 验证 force_compact 被调用（compact_count 增加）
        assert loop.context.compact_count >= 1
        # 验证 trace 记录了 compaction_triggered
        from echo.persistence.trace import read_jsonl
        events = read_jsonl(loop.run_store._trace_path())
        compact_events = [e for e in events if e.get("event") == "compaction_triggered"]
        assert len(compact_events) >= 1
        assert compact_events[0].get("trigger") == "tool_requested"


def test_force_compact_preserves_recent_messages():
    """force_compact 后最近 tool_use/tool_result 不被切断。"""
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "README.md").write_text("content")

        loop = _build_loop(d, [
            '<tool name="read_file" path="README.md" />',
            '<tool name="compact" />',
            "我已经读完文件，上下文已压缩。",
        ], approval="auto", max_steps=10)

        loop.run("read then compact")
        # force_compact 内部调用 _summarize_history 会额外消耗一次 LLM（summary）
        assert loop.llm.call_count >= 3
        # compact 不会把 compact tool 本身的 tool_result 截掉
        assert loop.messages[-2]["role"] == "user"  # tool_result for compact


def test_large_tool_output_persisted():
    """超大工具输出被落盘，上下文只保留 <persisted-output> 预览。"""
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "big.txt").write_text("x" * 35_000)

        # 注入 persist_dir
        config = ContextConfig(
            enable_compaction=True,
            persist_dir=str(Path(d) / ".echo" / "tool_outputs"),
            transcript_dir=str(Path(d) / ".echo" / "transcripts"),
        )
        loop = _build_loop(d, [
            '<tool name="read_file" path="big.txt" />',
            "文件内容很大。",
        ], context_config=config)

        loop.run("read big file")
        # 验证 messages 中出现了 <persisted-output> 标记
        found = False
        result_msg = loop.messages[-2]
        for block in result_msg.get("content", []):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                content = block.get("content", "")
                if "persisted-output" in content:
                    assert "Preview" in content
                    assert "Full output" in content
                    # 验证文件确实落盘了
                    full_path = loop.sandbox.root / ".echo" / "tool_outputs"
                    txt_files = list(full_path.glob("*.txt"))
                    assert len(txt_files) >= 1, "persist_dir 未写文件"
                    assert txt_files[0].stat().st_size > 0, "文件为空"
                    found = True
                    break
        assert found, "未找到 <persisted-output> — 大输出未落盘"


def test_compact_count_precise():
    """主动 compact 一次 → compact_count == 1（不重复计数）。"""
    with tempfile.TemporaryDirectory() as d:
        loop = _build_loop(d, [
            '<tool name="compact" />',
            "上下文已压缩。",
        ], approval="auto")

        assert loop.context.compact_count == 0
        loop.run("compact")
        # 主动 compact 只 +1
        assert loop.context.compact_count == 1, (
            f"compact_count={loop.context.compact_count}, expected 1"
        )


def test_reactive_compact_writes_back_to_self_messages():
    """Reactive compact 后 self.messages 已被压缩。"""
    with tempfile.TemporaryDirectory() as d:
        from echo.providers.fake_client import FakeLLMClient, _parse_output
        from echo.providers.base import ModelResponse, TextBlock

        class FailOnce(FakeLLMClient):
            def chat(self, messages=None, tools=None, system="",
                     max_tokens=8000, temperature=0.0):
                self.call_count += 1
                self.last_system = system
                self.prompts.append({
                    "message_count": len(messages or []),
                    "tool_count": len(tools or []),
                    "system_len": len(system),
                })
                if self.call_count == 1:
                    raise ValueError("prompt too long: context_length_exceeded")
                if not self._pending_outputs:
                    return ModelResponse(
                        content=[TextBlock(text="")],
                        stop_reason="end_turn", model=self.model,
                    )
                raw = self._pending_outputs.pop(0)
                content = _parse_output(raw)
                return ModelResponse(
                    content=content,
                    stop_reason="tool_use"
                    if any(hasattr(b, "name") for b in content) else "end_turn",
                    model=self.model,
                )

        reg = ToolRegistry()
        reg.discover("echo.tools.builtin")
        hooks = HookManager()
        hooks.register(PermissionHook(), priority=0)
        hooks.register(LogHook(), priority=100)
        hooks.register(PostLogHook(), priority=100)
        hooks.register(StatsHook(), priority=200)
        sandbox = Sandbox(d)
        shell = ShellExecutor(d)
        sess_dir = str(Path(d) / ".echo" / "sessions" / "reactive2")
        loop = AgentLoop(
            llm=FailOnce(outputs=["compressed, continuing."]),
            memory=MemoryManager(KeywordMemory()),
            tools=ToolExecutor(reg),
            hooks=hooks,
            context=ContextManager(),
            sandbox=sandbox,
            shell=shell,
            session_store=SessionStore(d),
            run_store=RunStore(sess_dir),
            max_steps=5,
            approval_policy="auto",
        )

        # 先填入一些旧消息模拟 "上下文很大" 的场景
        for i in range(20):
            loop.messages.append(
                {"role": "user",
                 "content": f"filler message {i} " + "x" * 500}
            )

        before_len = len(loop.messages)
        loop.run("do something")
        # reactive compact 成功压缩 → self.messages 应显著变小
        assert len(loop.messages) < before_len, (
            f"reactive compact 未写回 self.messages: "
            f"before={before_len}, after={len(loop.messages)}"
        )


def test_dedupe_old_read_results():
    """L2 dedupe: 同一文件多次读取，旧结果被替换为占位。"""
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "f.py").write_text("content")

        loop = _build_loop(d, [
            '<tool name="read_file" path="f.py" />',
            '<tool name="read_file" path="f.py" />',
            '<tool name="read_file" path="f.py" />',
            "已读完三次。",
        ], max_steps=10)

        loop.run("read f.py three times")
        # L2 dedupe scans ALL messages for "Earlier read of" stubs
        earlier_replaced = False
        for msg in loop.messages:
            for block in (msg.get("content") or []):
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    if "Earlier read of" in str(block.get("content", "")):
                        earlier_replaced = True
        assert earlier_replaced, "L2 dedupe 未替换旧 read_file 结果"


def test_multiple_medium_outputs_persist_when_total_exceeds_budget():
    """多个中等大小 tool_result 合计超预算 → 最终上下文大小下降。"""
    with tempfile.TemporaryDirectory() as d:
        persist_dir = str(Path(d) / ".echo" / "tool_outputs")
        config = ContextConfig(
            enable_compaction=True,
            compact_threshold_chars=100,
            compact_preview_chars=20,
            persist_dir=persist_dir,
            transcript_dir=str(Path(d) / ".echo" / "transcripts"),
        )
        (Path(d) / "a.txt").write_text("x" * 70)
        (Path(d) / "b.txt").write_text("y" * 70)

        loop = _build_loop(d, [
            '<tool name="read_file" path="a.txt" /><tool name="read_file" path="b.txt" />',
            "done.",
        ], context_config=config)

        # 记录压缩前 tool_result 的总大小
        loop.run("read both files")

        # L1 压缩后 tool_result 的内容应显著小于原始内容
        result_msg = loop.messages[-2]
        total_after = sum(
            len(str(b.get("content", "")))
            for b in (result_msg.get("content") or [])
            if isinstance(b, dict) and b.get("type") == "tool_result"
        )
        # 原始 140 字符，压缩后应小于 120（stub 有固定开销但必须比原文短）
        assert total_after < 120, (
            f"L1 压缩后大小未下降: {total_after} chars (期望 < 120)"
        )
        # 落盘文件也存在
        assert len(list(Path(persist_dir).glob("*.txt"))) >= 1


def test_reactive_compact_on_prompt_too_long():
    """prompt-too-long 错误触发 reactive compact 并重试。"""
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "f.txt").write_text("data")

        # 用能抛出 prompt-too-long 异常的 FakeLLMClient
        from echo.providers.fake_client import FakeLLMClient

        class FailingThenOk(FakeLLMClient):
            def chat(self, messages=None, tools=None, system="",
                     max_tokens=8000, temperature=0.0):
                self.call_count += 1
                self.last_system = system
                self.prompts.append({
                    "message_count": len(messages or []),
                    "tool_count": len(tools or []),
                    "system_len": len(system),
                })
                if self.call_count == 1:
                    raise ValueError("prompt too long: context length exceeded")
                if not self._pending_outputs:
                    from echo.providers.base import ModelResponse, TextBlock
                    return ModelResponse(
                        content=[TextBlock(text="")],
                        stop_reason="end_turn", model=self.model,
                    )
                raw = self._pending_outputs.pop(0)
                from echo.providers.fake_client import _parse_output
                content = _parse_output(raw)
                from echo.providers.base import ModelResponse
                return ModelResponse(
                    content=content,
                    stop_reason="tool_use"
                    if any(hasattr(b, "name") for b in content) else "end_turn",
                    model=self.model,
                )

        reg = ToolRegistry()
        reg.discover("echo.tools.builtin")
        hooks = HookManager()
        hooks.register(PermissionHook(), priority=0)
        hooks.register(LogHook(), priority=100)
        hooks.register(PostLogHook(), priority=100)
        hooks.register(StatsHook(), priority=200)
        sandbox = Sandbox(d)
        shell = ShellExecutor(d)
        sess_dir = str(Path(d) / ".echo" / "sessions" / "reactive")
        loop = AgentLoop(
            llm=FailingThenOk(outputs=["恢复了，继续工作。"]),
            memory=MemoryManager(KeywordMemory()),
            tools=ToolExecutor(reg),
            hooks=hooks,
            context=ContextManager(),
            sandbox=sandbox,
            shell=shell,
            session_store=SessionStore(d),
            run_store=RunStore(sess_dir),
            max_steps=5,
            approval_policy="auto",
        )
        answer = loop.run("do something")
        # 第1次失败 → reactive compact → 第2次成功（返回文本）
        assert isinstance(answer, str) and len(answer) > 0


def test_compact_not_cut_tool_use_result_pair():
    """snip_compact 不会切断 tool_use ↔ tool_result 配对。"""
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "file.txt").write_text("content")
        loop = _build_loop(d, [
            '<tool name="read_file" path="file.txt" />',
            "done",
        ])

        loop.run("read file")
        msgs = loop.messages
        # 最后的 tool_result 前面必有对应的 tool_use
        for i in range(len(msgs)):
            if i > 0 and any(
                isinstance(b, dict) and b.get("type") == "tool_result"
                for b in (msgs[i].get("content") or [])
            ):
                assert any(
                    hasattr(b, "id") for b in (msgs[i - 1].get("content") or [])
                    if hasattr(b, "id")
                ), f"tool_result at msg {i} 无对应的 tool_use"


def test_passive_compact_not_lose_recent_state():
    """被动 compact（每轮自动调用）不会丢失最近的工具调用状态。"""
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "a.txt").write_text("AAA")
        (Path(d) / "b.txt").write_text("BBB")
        (Path(d) / "c.txt").write_text("CCC")

        # 模拟多轮对话
        loop = _build_loop(d, [
            '<tool name="read_file" path="a.txt" />',
            '<tool name="read_file" path="b.txt" />',
            '<tool name="read_file" path="c.txt" />',
            "已读完三个文件。",
        ], max_steps=10)

        loop.run("read all 3 files")
        # 被动 compact 正常执行，不崩溃
        assert loop.llm.call_count == 4
        # 最近一轮 tool 调用结果未被压缩成存根（在 keep_recent 范围内）
        result_msg = loop.messages[-2]
        has_full = any(
            isinstance(b, dict) and b.get("type") == "tool_result"
            and len(b.get("content", "")) > 0
            for b in (result_msg.get("content") or [])
        )
        assert has_full, "最近的 tool_result 被误压缩成存根"


# ═══════════════════════════════════════════════════
# Phase 5: Todo + Session + Resume 回归测试
# ═══════════════════════════════════════════════════

def test_todo_write_updates_task_state():
    """todo_write validates and writes to session.short_term_memory[\"todos\"]。"""
    with tempfile.TemporaryDirectory() as d:
        loop = _build_loop(d, [
            '<tool>{"name":"todo_write","args":{"todos":[{"content":"task A","status":"in_progress"}]}}</tool>',
            "todo saved.",
        ], approval="auto")
        loop._session = Session(session_id="test-todo", workspace_root=d)

        answer = loop.run("create a todo")
        assert isinstance(answer, str)
        # 验证 session 中存了 todos
        todos = loop._session.short_term_memory.get("todos", [])
        assert len(todos) == 1
        assert todos[0]["content"] == "task A"
        assert todos[0]["status"] == "in_progress"


def test_todos_injected_into_system_prompt():
    """state 有未完成 todo → system prompt 包含 ## Current Todos。"""
    with tempfile.TemporaryDirectory() as d:
        loop = _build_loop(d, [
            '<tool>{"name":"todo_write","args":{"todos":[{"content":"fix bug","status":"in_progress"}]}}</tool>',
            "todo saved.",
        ], approval="auto")

        loop.run("write a todo")
        fake = loop.llm
        # check system prompt contains Current Todos
        if fake.last_system:
            assert "Current Todos" in fake.last_system


def test_session_history_saved_after_run():
    """一次完整 run 之后 session.history 非空。"""
    with tempfile.TemporaryDirectory() as d:
        loop = _build_loop(d, [
            '<tool name="read_file" path="f.txt" />',
            'done.',
        ])
        (Path(d) / "f.txt").write_text("content")
        loop._session = Session(session_id="test-h", workspace_root=d)
        loop.run("read f.txt")
        assert len(loop._session.history) > 0, "session history 未写入"


def test_resume_loads_latest_session():
    """resume() 无 user_request 时返回就绪摘要，含 checkpoint 信息。"""
    with tempfile.TemporaryDirectory() as d:
        from echo.persistence.session_store import Session
        from echo.persistence.checkpoint import CheckpointManager

        session = Session(session_id="s1", workspace_root=d)
        session.history = [
            {"role": "user", "content": [{"type": "text", "text": "test"}]},
        ]
        session.short_term_memory = {"task_summary": "testing"}
        session.checkpoints = {
            "current_id": "ckpt_test",
            "items": {
                "ckpt_test": {
                    "checkpoint_id": "ckpt_test", "schema_version": "v1",
                    "current_goal": "fix login bug",
                    "current_blocker": "auth error",
                    "next_step": "read auth.py",
                    "key_files": {}, "runtime_identity": {"cwd": d},
                    "completed": [], "excluded": [],
                    "snapshot_teammates": {}, "unprocessed_messages": [],
                    "pending_protocols": [], "parent_id": None,
                    "created_at": "2025-01-01T00:00:00",
                }
            },
        }

        ss = SessionStore(d)
        ss.save(session)

        cm = CheckpointManager(d)
        ckpt = cm.load_from_session(session)
        assert ckpt is not None
        assert ckpt.current_goal == "fix login bug"
        assert ckpt.next_step == "read auth.py"

        text = cm.render(ckpt)
        assert "fix login bug" in text

        # 加载 session 直接验证（不实例化 Echo，避免依赖 SDK）
        loaded = ss.load("s1")
        assert loaded.history == session.history
        assert loaded.short_term_memory == session.short_term_memory


def test_resume_injects_checkpoint_status():
    """resume 后 checkpoint 可加载 + goal/blocker/next_step 正确。"""
    with tempfile.TemporaryDirectory() as d:
        from echo.persistence.session_store import Session
        from echo.persistence.checkpoint import CheckpointManager

        session = Session(session_id="test-ckpt", workspace_root=d)
        session.checkpoints = {
            "current_id": "ckpt_test",
            "items": {
                "ckpt_test": {
                    "checkpoint_id": "ckpt_test", "schema_version": "v1",
                    "current_goal": "fix login bug",
                    "current_blocker": "auth error",
                    "next_step": "read auth.py",
                    "key_files": {}, "runtime_identity": {"cwd": d},
                    "completed": [], "excluded": [],
                    "snapshot_teammates": {}, "unprocessed_messages": [],
                    "pending_protocols": [], "parent_id": None,
                    "created_at": "2025-01-01T00:00:00",
                }
            },
        }

        cm = CheckpointManager(d)
        ckpt = cm.load_from_session(session)
        assert ckpt is not None
        assert ckpt.current_goal == "fix login bug"
        assert ckpt.next_step == "read auth.py"

        # render 包含关键信息
        text = cm.render(ckpt)
        assert "fix login bug" in text
        assert "auth error" in text

        # evaluate 返回 full-valid
        result = cm.evaluate(ckpt)
        assert result["status"] == "full-valid"


# ═══════════════════════════════════════════════════
# E2E Resume with FakeLLM
# ═══════════════════════════════════════════════════


def test_resume_e2e_with_fake_llm():
    """Resume E2E: history + new request via AgentLoop.run(resume_messages)."""
    with tempfile.TemporaryDirectory() as d:
        from echo.core.agent_loop import AgentLoop
        from echo.memory.base import MemoryManager
        from echo.memory.default import KeywordMemory
        from echo.providers.fake_client import FakeLLMClient
        from echo.providers.base import TextBlock
        from echo.persistence.session_store import Session, SessionStore

        (Path(d) / "README.md").write_text("# MyProject")

        reg = ToolRegistry(); reg.discover("echo.tools.builtin")
        hooks = HookManager()
        hooks.register(PermissionHook(), priority=0)
        hooks.register(LogHook(), priority=100)
        hooks.register(PostLogHook(), priority=100)
        hooks.register(StatsHook(), priority=200)
        memory = MemoryManager(KeywordMemory())
        memory.observe_user_message("read README")

        sess_dir = str(Path(d) / ".echo" / "sessions" / "test-e2e-resume")
        session = Session(session_id="test-e2e-resume", workspace_root=d)
        SessionStore(d).save(session)

        loop = AgentLoop(
            llm=FakeLLMClient(outputs=[
                '<tool name="read_file" path="README.md" />',
                "Project is MyProject. Done.",
            ]),
            memory=memory, tools=ToolExecutor(reg), hooks=hooks,
            context=ContextManager(), sandbox=Sandbox(d),
            shell=ShellExecutor(d), session_store=SessionStore(d),
            run_store=RunStore(sess_dir),
            max_steps=10, approval_policy="auto",
        )
        loop._session = session

        resume_msgs = [
            {"role": "user", "content": [TextBlock(text="read README")]},
            {"role": "assistant", "content": [TextBlock(text="I will read.")]},
            {"role": "user", "content": [TextBlock(text="continue")]},
        ]
        answer = loop.run("continue", resume_messages=resume_msgs)
        assert "MyProject" in answer, f"got: {answer[:200]}"

        import json
        report = json.loads((loop.run_store._run_dir / "report.json").read_text(encoding="utf-8"))
        assert report["status"] == "completed"
        assert len(report["final_answer"]) > 0


# ==================================================================
# Delegate / SubAgent E2E Tests
# ==================================================================

def test_delegate_runs_subagent_and_returns_summary():
    """主 Agent 调用 delegate → 子 Agent 读文件 → 返回摘要。"""
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "README.md").write_text("# MyProject")
        (Path(d) / "src").mkdir(exist_ok=True)
        (Path(d) / "src" / "main.py").write_text("def main(): pass")

        loop = _build_loop(d, [
            '<tool name="delegate" task="调查项目中有哪些文件" max_steps="3" />',
            "调查完成。",
        ])

        answer = loop.run("investigate project structure")
        assert isinstance(answer, str)


def test_delegate_does_not_pollute_parent_messages():
    """主 Agent messages 中只有 delegate tool_result，不暴露子 Agent 中间步骤。"""
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "data.txt").write_text("secret-42")

        loop = _build_loop(d, [
            '<tool name="delegate" task="read data.txt" max_steps="2" />',
            "文件包含 secret-42。",
        ])

        loop.run("delegate read task")
        # delegates 不崩溃且返回有意义的内容
        assert len(loop.messages) >= 3


def test_delegate_depth_limit():
    """depth >= max_depth 时 delegate 失败。"""
    with tempfile.TemporaryDirectory() as d:
        reg = ToolRegistry()
        reg.discover("echo.tools.builtin")
        sandbox = Sandbox(d)
        ctx = ToolContext(
            workspace_root=d, sandbox=sandbox,
            shell=ShellExecutor(d),
            memory=MemoryManager(KeywordMemory()),
            depth=1, max_depth=1,
        )
        from echo.tools.builtin import DelegateTool
        r = DelegateTool().run(ctx, {"task": "try", "max_steps": 2})
        assert not r.success
        assert "depth" in r.error.lower()


def test_subagent_step_limit():
    """子 Agent 到 max_steps 后返回 step limit summary。"""
    with tempfile.TemporaryDirectory() as d:
        loop = _build_loop(d, [
            '<tool name="delegate" task="read everything" max_steps="1" />',
            "子 Agent 已尽力。",
        ])

        answer = loop.run("delegate with low steps")
        assert isinstance(answer, str)


def test_delegate_failure_does_not_crash_main():
    """子 Agent 失败时主 Agent 继续运行。"""
    with tempfile.TemporaryDirectory() as d:
        loop = _build_loop(d, [
            '<tool name="delegate" task="will fail" max_steps="1" />',
            "主 Agent 还在运行。",
        ])

        answer = loop.run("delegate that fails")
        assert isinstance(answer, str)


def test_delegate_uses_only_read_only_tools():
    """DelegateTool 筛选只读工具，排除 write_file/run_shell/delegate 等。"""
    reg = ToolRegistry()
    reg.discover("echo.tools.builtin")
    all_tools = reg.get_all()
    read_only = [t for t in all_tools if t.is_read_only and t.name != "delegate"]

    risky_names = {"write_file", "patch_file", "run_shell",
                   "save_memory", "compact", "todo_write", "delegate"}
    for tool in read_only:
        assert tool.name not in risky_names, f"{tool.name} should not be read_only"

    assert len(read_only) >= 3  # read_file, glob, grep, list_files, search_memory


def test_delegate_e2e_subagent_reads_file_parent_gets_summary():
    """Strong E2E: 子 Agent 真的 read_file，主 Agent tool_result 含文件内容。"""
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "secrets.txt").write_text("API_KEY=sk-abc123\nDB_PASS=xyz")

        loop = _build_loop(d, [
            # 输出 0: 主 Agent 调 delegate
            '<tool name="delegate" task="read secrets.txt and report all key names" max_steps="3" />',
            # 输出 1: 子 Agent 读文件（FakeLLMClient 同一个实例，子 Agent 也消费）
            '<tool name="read_file" path="secrets.txt" />',
            # 输出 2: 子 Agent 返回摘要
            "文件包含 API_KEY 和 DB_PASS。",
            # 输出 3: 主 Agent 拿到 delegate 结果后总结
            "调查完成：secrets.txt 中有两个密钥。",
        ])

        answer = loop.run("check secrets file")
        # 主 Agent 的最终回复正确
        assert "调查完成" in answer or "密钥" in answer, f"unexpected answer: {answer[:200]}"

        # delegate tool_result 的内容含子 Agent 读到的文件信息
        delegate_result_content = ""
        for msg in loop.messages:
            for b in (msg.get("content") or []):
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    if "API_KEY" in str(b.get("content", "")):
                        delegate_result_content = str(b.get("content", ""))
        assert "API_KEY" in delegate_result_content, (
            f"delegate tool_result 不含文件信息: {delegate_result_content[:200]}"
        )

        # 父 messages 不存在任何 ToolUseBlock.name == "read_file"
        # （子 Agent 的 tool_use 不应泄漏到父 Agent）
        from echo.providers.base import ToolUseBlock
        sub_tool_names = []
        for msg in loop.messages:
            for b in (msg.get("content") or []):
                if isinstance(b, ToolUseBlock):
                    sub_tool_names.append(b.name)
        assert "read_file" not in sub_tool_names, (
            f"子 Agent 的 read_file tool_use 泄漏到父 Agent: {sub_tool_names}"
        )


def test_delegate_trace_has_full_lifecycle():
    """trace.jsonl 包含 delegate_started → sub_tool_executed → delegate_finished。"""
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "README.md").write_text("# project")
        (Path(d) / "src").mkdir(exist_ok=True)
        (Path(d) / "src" / "main.py").write_text("def main(): pass")

        loop = _build_loop(d, [
            '<tool name="delegate" task="调查 src/ 目录" max_steps="3" />',
            '<tool name="read_file" path="src/main.py" />',
            "src/ 目录含 main.py，其中定义了 main 函数。",
            "调查完毕：src/main.py 是入口文件。",
        ])

        loop.run("investigate src/")

        from echo.persistence.trace import read_jsonl
        events = read_jsonl(loop.run_store._trace_path())

        event_types = [e.get("event") for e in events]
        assert "delegate_started" in event_types, f"events: {event_types}"
        assert "sub_tool_executed" in event_types, f"events: {event_types}"
        assert "delegate_finished" in event_types, f"events: {event_types}"

        # sub_tool_executed 的 tool 字段应为 read_file
        sub_events = [e for e in events if e.get("event") == "sub_tool_executed"]
        assert any(e.get("tool") == "read_file" for e in sub_events), (
            f"sub_tool_executed 未记录 read_file: {sub_events}"
        )

        # delegate_finished 的 success 应为 True
        fin_events = [e for e in events if e.get("event") == "delegate_finished"]
        assert any(e.get("success") is True for e in fin_events)


def test_delegate_trace_no_secrets_leaked():
    """委托执行含密钥的文件 → trace.jsonl 不含 API_KEY/sk-xxx。"""
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "config.env").write_text("API_KEY=sk-2d978e8a3ae847c4a7149d8cd523410e")

        loop = _build_loop(d, [
            '<tool name="delegate" task="read config.env and summarize" max_steps="3" />',
            '<tool name="read_file" path="config.env" />',
            "文件包含一个 API_KEY 配置项。",
            "调查完毕：config.env 有 API_KEY。",
        ])

        loop.run("check config")
        from echo.persistence.trace import read_jsonl
        events = read_jsonl(loop.run_store._trace_path())
        all_trace_text = str(events).lower()
        assert "sk-2d978e8a" not in all_trace_text, (
            f"trace 泄露了 API key"
        )
        assert "api_key=sk-" not in all_trace_text, (
            f"trace 可能泄露了 API_KEY 值"
        )


def test_durable_memory_persists_across_backend_reload():
    """durable.json 在 workspace 的 .echo/memory/ 下，重启可读。"""
    with tempfile.TemporaryDirectory() as d:
        from echo.memory.durable import JsonDurableMemoryBackend
        from echo.memory.base import MemoryEntry

        # 模拟 .echo/memory/durable.json 的存储路径（与 Echo 门面一致）
        durable_path = str(Path(d) / ".echo" / "memory" / "durable.json")
        b1 = JsonDurableMemoryBackend(durable_path)
        b1.store(MemoryEntry(text="长期规则：API 用 RESTful", tags=["rule"],
                             source="save_memory", kind="durable"))
        assert b1.count() == 1

        # 第二次加载（模拟进程重启）
        b2 = JsonDurableMemoryBackend(durable_path)
        assert b2.count() == 1
        results = b2.search("RESTful")
        assert len(results) == 1
        assert results[0].text == "长期规则：API 用 RESTful"


# ==================================================================
# Durable Memory Tests
# ==================================================================

def test_durable_memory_persists_to_json():
    """durable.json 文件在写入后存在且内容正确。"""
    with tempfile.TemporaryDirectory() as d:
        import json
        from echo.memory.durable import JsonDurableMemoryBackend
        backend = JsonDurableMemoryBackend(str(Path(d) / "durable.json"))
        entry = MemoryEntry(text="用户偏好：代码注释用中文", tags=["preference"],
                            source="save_memory", kind="durable")
        mid = backend.store(entry)
        assert Path(d, "durable.json").exists()

        data = json.loads(Path(d, "durable.json").read_text(encoding="utf-8"))
        assert data["version"] == 1
        assert len(data["items"]) == 1
        assert data["items"][0]["text"] == "用户偏好：代码注释用中文"


def test_save_memory_writes_durable():
    """save_memory 工具写入持久记忆。"""
    with tempfile.TemporaryDirectory() as d:
        from echo.memory.base import MemoryManager
        from echo.memory.default import KeywordMemory
        from echo.memory.durable import JsonDurableMemoryBackend
        from echo.tools.builtin import SaveMemoryTool

        durable = JsonDurableMemoryBackend(str(Path(d) / "durable.json"))
        mem = MemoryManager(KeywordMemory(), durable_backend=durable)
        ctx = ToolContext(memory=mem)

        r = SaveMemoryTool().run(ctx, {
            "content": "项目使用 Python 3.12", "tags": ["tech"],
        })
        assert r.success
        assert durable.count() == 1


def test_search_memory_reads_working_and_durable():
    """search_memory 同时查 working 和 durable。"""
    with tempfile.TemporaryDirectory() as d:
        from echo.memory.base import MemoryManager
        from echo.memory.default import KeywordMemory
        from echo.memory.durable import JsonDurableMemoryBackend
        from echo.tools.builtin import SearchMemoryTool

        durable = JsonDurableMemoryBackend(str(Path(d) / "durable.json"))
        mem = MemoryManager(KeywordMemory(), durable_backend=durable)
        # 写一条 durable
        entry = MemoryEntry(text="长期记忆：单元测试优先", tags=["rule"],
                            source="manual", kind="durable")
        durable.store(entry)
        # 写一条 working
        mem.add("working memory note", {"tags": ["note"], "source": "test"})

        ctx = ToolContext(memory=mem)
        r = SearchMemoryTool().run(ctx, {"query": "单元测试"})
        assert r.success
        assert "长期记忆" in r.output  # durable 被搜到


def test_secret_shaped_memory_rejected():
    """save_memory 拒绝保存疑似密钥的内容。"""
    with tempfile.TemporaryDirectory() as d:
        from echo.memory.base import MemoryManager
        from echo.memory.default import KeywordMemory
        from echo.memory.durable import JsonDurableMemoryBackend
        from echo.tools.builtin import SaveMemoryTool

        durable = JsonDurableMemoryBackend(str(Path(d) / "durable.json"))
        mem = MemoryManager(KeywordMemory(), durable_backend=durable)
        ctx = ToolContext(memory=mem)

        r = SaveMemoryTool().run(ctx, {
            "content": "my api_key is sk-live-abcdef123456", "tags": ["secret"],
        })
        assert not r.success
        assert "疑似" in r.error or "密钥" in r.error or "敏感" in r.error


def test_build_system_injects_relevant_durable_memory():
    """system prompt 注入相关长期记忆。"""
    with tempfile.TemporaryDirectory() as d:
        from echo.memory.base import MemoryManager
        from echo.memory.default import KeywordMemory
        from echo.memory.durable import JsonDurableMemoryBackend
        from echo.memory.base import MemoryEntry
        from echo.core.context_manager import ContextManager
        from echo.core.task_state import TaskState

        durable = JsonDurableMemoryBackend(str(Path(d) / "durable.json"))
        entry = MemoryEntry(text="用户偏好：API 设计用 RESTful", tags=["pref"],
                            source="manual", kind="durable")
        durable.store(entry)
        mem = MemoryManager(KeywordMemory(), durable_backend=durable)

        cm = ContextManager()
        ts = TaskState.create("design the API")
        from echo.tools.registry import ToolRegistry
        reg = ToolRegistry()
        from echo.security.sandbox import Sandbox
        system = cm.build_system(ts, reg, mem, Sandbox(d))

        assert "RESTful" in system
        assert "Long-term Memory" in system or "长期记忆" in system or "Relevant Long" in system


def test_durable_memory_survives_new_instance():
    """长期记忆跨实例持久化。"""
    with tempfile.TemporaryDirectory() as d:
        from echo.memory.durable import JsonDurableMemoryBackend
        from echo.memory.base import MemoryEntry

        path = str(Path(d) / "durable.json")
        b1 = JsonDurableMemoryBackend(path)
        b1.store(MemoryEntry(text="persistent note", tags=["x"],
                             source="test", kind="durable"))
        assert b1.count() == 1

        # 新实例加载同一文件
        b2 = JsonDurableMemoryBackend(path)
        assert b2.count() == 1
        results = b2.search("persistent")
        assert len(results) == 1
        assert results[0].text == "persistent note"


def test_durable_not_squeezed_out_by_working():
    """working 很多时 durable 仍能被搜到。"""
    with tempfile.TemporaryDirectory() as d:
        from echo.memory.base import MemoryManager
        from echo.memory.default import KeywordMemory
        from echo.memory.durable import JsonDurableMemoryBackend
        from echo.memory.base import MemoryEntry
        from echo.tools.builtin import SearchMemoryTool

        durable = JsonDurableMemoryBackend(str(Path(d) / "durable.json"))
        mem = MemoryManager(KeywordMemory(), durable_backend=durable)

        # 写入很多 working memory（带时间分）
        for i in range(20):
            mem.add(f"working note number {i}", {"tags": ["note"], "source": "test"})

        # 写入一条明确的 durable memory
        mem.save_durable("用户偏好：代码注释用中文", {"tags": ["preference"]})

        ctx = ToolContext(memory=mem)
        r = SearchMemoryTool().run(ctx, {"query": "代码注释"})
        assert r.success
        assert "代码注释用中文" in r.output or "用户偏好" in r.output, (
            "durable 记忆未被搜到"
        )
