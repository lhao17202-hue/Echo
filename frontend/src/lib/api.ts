import type {
  ApprovalDecisionRequest,
  ApprovalDecisionResponse,
  ApprovalPolicyUpdateRequest,
  ApprovalRequestDTO,
  ChatRequest,
  ChatResponse,
  ConfigSummary,
  GitStatus,
  RunFileDiff,
  RunFileSummary,
  RuntimeStatus,
  SessionDetail,
  SessionSummary,
  SessionUpdateRequest,
  TraceEventDTO,
  WorkspaceInfo,
} from '../types/api'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

async function readError(response: Response): Promise<string> {
  const text = await response.text()
  if (!text) return `请求失败：HTTP ${response.status}`

  try {
    const data = JSON.parse(text) as { detail?: unknown; answer?: unknown; message?: unknown }
    if (typeof data.answer === 'string') return data.answer
    if (typeof data.detail === 'string') return data.detail
    if (typeof data.message === 'string') return data.message
    if (Array.isArray(data.detail)) return data.detail.map((item) => item?.msg ?? String(item)).join('；')
  } catch {
    // fall through to plain text
  }

  return text
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: {
        'Content-Type': 'application/json',
        ...(init?.headers ?? {}),
      },
    })
  } catch (error) {
    if (error instanceof TypeError) {
      throw new Error('无法连接 Echo 后端，请确认 8000 端口的 FastAPI 服务正在运行。')
    }
    throw error
  }

  if (!response.ok) {
    throw new Error(await readError(response))
  }

  if (response.status === 204) {
    return undefined as T
  }

  return response.json() as Promise<T>
}

export function sendChatMessage(input: ChatRequest): Promise<ChatResponse> {
  return request<ChatResponse>('/api/chat', {
    method: 'POST',
    body: JSON.stringify(input),
  })
}

export function getPendingApprovals(): Promise<ApprovalRequestDTO[]> {
  return request<ApprovalRequestDTO[]>('/api/approvals/pending')
}

export function sendApprovalDecision(requestId: string, input: ApprovalDecisionRequest): Promise<ApprovalDecisionResponse> {
  return request<ApprovalDecisionResponse>(`/api/approvals/${encodeURIComponent(requestId)}/decision`, {
    method: 'POST',
    body: JSON.stringify(input),
  })
}

export function listSessions(query?: string): Promise<SessionSummary[]> {
  const search = query?.trim()
  const suffix = search ? `?query=${encodeURIComponent(search)}` : ''
  return request<SessionSummary[]>(`/api/sessions${suffix}`)
}

export function getSession(sessionId: string): Promise<SessionDetail> {
  return request<SessionDetail>(`/api/sessions/${encodeURIComponent(sessionId)}`)
}

export function renameSession(sessionId: string, input: SessionUpdateRequest): Promise<SessionSummary> {
  return request<SessionSummary>(`/api/sessions/${encodeURIComponent(sessionId)}`, {
    method: 'PATCH',
    body: JSON.stringify(input),
  })
}

export function deleteSession(sessionId: string): Promise<void> {
  return request<void>(`/api/sessions/${encodeURIComponent(sessionId)}`, {
    method: 'DELETE',
  })
}

export function getRunTrace(runId: string): Promise<TraceEventDTO[]> {
  return request<TraceEventDTO[]>(`/api/runs/${encodeURIComponent(runId)}/trace`)
}

export function getWorkspaceInfo(): Promise<WorkspaceInfo> {
  return request<WorkspaceInfo>('/api/workspace')
}

export function getGitStatus(): Promise<GitStatus> {
  return request<GitStatus>('/api/git/status')
}

export function getConfigSummary(): Promise<ConfigSummary> {
  return request<ConfigSummary>('/api/config')
}

export function updateApprovalPolicy(input: ApprovalPolicyUpdateRequest): Promise<ConfigSummary> {
  return request<ConfigSummary>('/api/config/approval', {
    method: 'PATCH',
    body: JSON.stringify(input),
  })
}

export function getRuntimeStatus(): Promise<RuntimeStatus> {
  return request<RuntimeStatus>('/api/runtime/status')
}

export function getRunFiles(runId: string): Promise<RunFileSummary[]> {
  return request<RunFileSummary[]>(`/api/runs/${encodeURIComponent(runId)}/files`)
}

export function getRunFileDiff(runId: string, path: string): Promise<RunFileDiff> {
  return request<RunFileDiff>(`/api/runs/${encodeURIComponent(runId)}/files/diff?path=${encodeURIComponent(path)}`)
}
