"""Frontend approval dialog regression tests."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).parent.parent
FRONTEND = ROOT / "frontend"


def run_node(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["node", "--input-type=module"],
        input=script,
        cwd=FRONTEND,
        text=True,
        capture_output=True,
        timeout=30,
    )


def test_approval_display_handles_missing_tool_input_without_crashing():
    """Pending approval payloads without tool_input should still be renderable."""
    result = run_node(
        """
        import { createServer } from 'vite';
        const server = await createServer({ server: { middlewareMode: true }, appType: 'custom', logLevel: 'error' });
        try {
          const mod = await server.ssrLoadModule('/src/lib/approvalDisplay.ts');
          const display = mod.formatApprovalDisplay({
            request_id: 'approval_test',
            tool_name: 'write_file',
            risk_level: 'warn',
            status: 'pending',
          });
          if (display.command !== '') throw new Error(`unexpected command: ${display.command}`);
          if (display.inputJson !== '{}') throw new Error(`unexpected inputJson: ${display.inputJson}`);
        } finally {
          await server.close();
        }
        """
    )

    assert result.returncode == 0, result.stderr


def test_policy_options_include_danger_mode():
    """The frontend policy control should offer danger as a selectable mode."""
    result = run_node(
        """
        import { createServer } from 'vite';
        const server = await createServer({ server: { middlewareMode: true }, appType: 'custom', logLevel: 'error' });
        try {
          const mod = await server.ssrLoadModule('/src/lib/approvalPolicy.ts');
          if (!mod.APPROVAL_POLICIES.includes('danger')) {
            throw new Error(`missing danger policy: ${mod.APPROVAL_POLICIES.join(',')}`);
          }
        } finally {
          await server.close();
        }
        """
    )

    assert result.returncode == 0, result.stderr


def test_policy_choices_exclude_current_policy():
    """Opening the policy selector should show the remaining choices, not require native select arrows."""
    result = run_node(
        """
        import { createServer } from 'vite';
        const server = await createServer({ server: { middlewareMode: true }, appType: 'custom', logLevel: 'error' });
        try {
          const mod = await server.ssrLoadModule('/src/lib/approvalPolicy.ts');
          const choices = mod.approvalPolicyChoices('danger');
          if (choices.includes('danger')) throw new Error(`current policy should be hidden: ${choices.join(',')}`);
          for (const policy of ['ask', 'auto', 'never']) {
            if (!choices.includes(policy)) throw new Error(`missing ${policy}: ${choices.join(',')}`);
          }
        } finally {
          await server.close();
        }
        """
    )

    assert result.returncode == 0, result.stderr
