export type TraceEventDTO = {
  event: string
  run_id?: string | null
  event_id?: string | null
  created_at?: string | null
  timestamp?: number | null
  payload: Record<string, unknown>
}

export type ToolCallSummary = {
  name: string
  input_summary: string
  success?: boolean | null
  output_summary: string
}

export type ChatRequest = {
  message: string
  session_id?: string | null
}

export type ChatResponse = {
  session_id: string
  run_id: string
  answer: string
  status: string
  trace: TraceEventDTO[]
  tools: ToolCallSummary[]
  files_touched: string[]
}

export type ApprovalRequestDTO = {
  request_id: string
  tool_name: string
  risk_level: string
  tool_input: Record<string, unknown>
  command: string
  status: string
}

export type ApprovalPolicy = 'ask' | 'auto' | 'never' | 'danger'

export type ApprovalPolicyUpdateRequest = {
  approval_policy: ApprovalPolicy
}

export type ApprovalDecisionRequest = {
  approved: boolean
}

export type ApprovalDecisionResponse = {
  request_id: string
  status: string
}

export type SessionSummary = {
  session_id: string
  title: string
  updated_at?: string | null
  run_count: number
}

export type SessionUpdateRequest = {
  title: string
}

export type MessageDTO = {
  role: string
  content: string
}

export type SessionDetail = {
  session_id: string
  title: string
  messages: MessageDTO[]
}

export type WorkspaceInfo = {
  name: string
  root: string
}

export type GitStatus = {
  branch: string
  dirty: boolean
  changed_files: string[]
}

export type ConfigSummary = {
  provider: string
  model: string
  base_url: string
  approval_policy: ApprovalPolicy
  api_key_configured: boolean
}

export type RuntimeStatus = {
  background_tasks: number
  cron_tasks: number
  mcp_servers: number
  tools: number
  approval_policy: ApprovalPolicy
}

export type RunFileSummary = {
  path: string
  status: string
}

export type RunFileDiff = {
  path: string
  status: string
  diff: string
}
