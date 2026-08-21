import type { ApprovalRequestDTO } from '../types/api'

export function formatApprovalDisplay(approval: Partial<ApprovalRequestDTO>) {
  const toolInput: Record<string, unknown> = approval.tool_input && typeof approval.tool_input === 'object' ? approval.tool_input : {}
  const command = approval.command || (typeof toolInput.command === 'string' ? toolInput.command : '')

  return {
    command,
    inputJson: JSON.stringify(toolInput, null, 2),
  }
}
