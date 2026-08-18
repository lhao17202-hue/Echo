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

export type SessionSummary = {
  session_id: string
  title: string
  updated_at?: string | null
  run_count: number
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
