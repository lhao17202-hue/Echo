"""内置 Hook —— 权限校验、日志、大输出告警、运行统计。"""

import logging
from echo.hooks.base import BaseHook, HookEvent
from echo.security.permission import PermissionGuard
from echo.tools.base import BaseTool

logger = logging.getLogger("echo.hooks")


class PermissionHook(BaseHook):
    """权限校验 Hook —— PRE_TOOL_USE 事件的唯一权限入口。

    三级风险 + 三种审批策略：

      safe 工具（read_file, glob, grep, list_files, search_memory 等）:
        → 永远放行

      warn 工具（write_file, patch_file, save_memory, todo_write, compact 等）:
        approval_policy = auto  → 直接放行
        approval_policy = ask   → 交互确认（input）
        approval_policy = never → 拦截

      danger 工具（run_shell）:
        先过 DENY_LIST（硬拦截，不询问）
        approval_policy = auto  → 放行（但 PermissionGuard.check_shell_command 仍然生效）
        approval_policy = ask   → 打印命令并交互确认
        approval_policy = never → 拦截

    approval_policy 通过 kwargs["approval_policy"] 传入（由 AgentLoop 注入）。
    """

    event = HookEvent.PRE_TOOL_USE

    def handle(self, **kwargs) -> str | None:
        tool = kwargs.get("tool")
        tool_input = kwargs.get("tool_input", {})
        policy = kwargs.get("approval_policy", "ask")

        if tool is None:
            return None
        if not isinstance(tool, BaseTool):
            logger.warning(f"PermissionHook 收到非 BaseTool: {type(tool).__name__}，放行")
            return None

        risk = tool.risk_level

        # ── safe: 永远放行 ─────────────────────────
        if risk == "safe":
            return None

        # ── warn: 按策略 ────────────────────────────
        if risk == "warn":
            if policy == "auto":
                return None
            elif policy == "never":
                return f"拒绝: '{tool.name}' 需要授权（policy=never）"
            else:  # ask
                answer = input(f"\n⚠ 确认执行 '{tool.name}'? [y/N] ").strip().lower()
                if answer in ("y", "yes"):
                    return None
                return f"用户取消了 '{tool.name}'"

        # ── danger: DENY_LIST + 策略 ────────────────
        if risk == "danger":
            # 先检查 shell 命令的 deny list
            command = tool_input.get("command", "")
            if command and PermissionGuard.is_denied(command):
                return f"拒绝: 命令在 deny list 中 —— '{command[:120]}'"

            if policy == "auto":
                # auto 模式下仍检查 destructive 模式
                allowed, msg = PermissionGuard.check_shell_command(command)
                if not allowed:
                    return f"拒绝: {msg}"
                return None
            elif policy == "never":
                return f"拒绝: '{tool.name}' 需要显式授权（policy=never）"
            else:  # ask
                # 打印命令并询问
                if command:
                    print(f"\n🔴 危险命令:\n  {command[:200]}")
                answer = input(f"确认执行 '{tool.name}'? [y/N] ").strip().lower()
                if answer in ("y", "yes"):
                    return None
                return f"用户取消了 '{tool.name}'"

        return None


class LogHook(BaseHook):
    """工具调用日志 Hook — PRE_TOOL_USE。"""

    event = HookEvent.PRE_TOOL_USE

    def handle(self, **kwargs) -> str | None:
        tool = kwargs.get("tool")
        tool_input = kwargs.get("tool_input", {})
        tool_name = getattr(tool, "name", "unknown") if tool else "unknown"
        logger.info(f"> {tool_name} {tool_input}")
        return None


class PostLogHook(BaseHook):
    """工具结果日志 Hook — POST_TOOL_USE。"""

    event = HookEvent.POST_TOOL_USE

    def handle(self, **kwargs) -> str | None:
        result = kwargs.get("result")
        text = getattr(result, "output", str(result)) if result else ""
        preview = str(text)[:200].replace("\n", " ")
        logger.debug(f"  -> {preview}")
        return None


class LargeOutputHook(BaseHook):
    """大输出告警 Hook — POST_TOOL_USE。"""

    event = HookEvent.POST_TOOL_USE

    def handle(self, **kwargs) -> str | None:
        result = kwargs.get("result")
        text = getattr(result, "output", str(result)) if result else ""
        if len(str(text)) > 100_000:
            logger.warning(f"Large tool output: {len(str(text))} chars")
        return None


class StatsHook(BaseHook):
    """运行统计 Hook — RUN_STOP。"""

    event = HookEvent.RUN_STOP

    def handle(self, **kwargs) -> str | None:
        state = kwargs.get("state")
        if state:
            logger.info(
                f"Run finished: {state.tool_steps} steps, "
                f"{state.attempts} attempts, status={state.status.value}"
            )
        return None
