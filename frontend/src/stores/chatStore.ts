import { create } from 'zustand'

import { deleteSession as deleteSessionRequest, getSession, listSessions, renameSession, sendChatMessage } from '../lib/api'
import type { SessionSummary, ToolCallSummary, TraceEventDTO } from '../types/api'

export type Message = {
  id: string
  role: 'user' | 'assistant'
  content: string
}

type ChatState = {
  sessions: SessionSummary[]
  sessionQuery: string
  isLoadingSessions: boolean
  isOpeningSession: boolean
  input: string
  messages: Message[]
  sessionId: string | null
  currentRunId: string | null
  currentStatus: string
  trace: TraceEventDTO[]
  tools: ToolCallSummary[]
  filesTouched: string[]
  isSending: boolean
  error: string | null
  loadSessions: (query?: string) => Promise<void>
  setSessionQuery: (query: string) => Promise<void>
  openSession: (sessionId: string) => Promise<void>
  renameCurrentOrSession: (sessionId: string, title: string) => Promise<void>
  deleteSession: (sessionId: string) => Promise<void>
  newChat: () => void
  setInput: (input: string) => void
  send: () => Promise<void>
}

const emptyRunState = {
  currentRunId: null,
  currentStatus: 'idle',
  trace: [],
  tools: [],
  filesTouched: [],
}

export const useChatStore = create<ChatState>((set, get) => ({
  sessions: [],
  sessionQuery: '',
  isLoadingSessions: false,
  isOpeningSession: false,
  input: '',
  messages: [],
  sessionId: null,
  currentRunId: null,
  currentStatus: 'idle',
  trace: [],
  tools: [],
  filesTouched: [],
  isSending: false,
  error: null,
  loadSessions: async (query) => {
    const search = query ?? get().sessionQuery
    set({ isLoadingSessions: true })
    try {
      const sessions = await listSessions(search)
      set({ sessions, isLoadingSessions: false })
    } catch (error) {
      const message = error instanceof Error ? error.message : '加载历史会话失败'
      set({ isLoadingSessions: false, error: message })
    }
  },
  setSessionQuery: async (query) => {
    set({ sessionQuery: query })
    await get().loadSessions(query)
  },
  openSession: async (sessionId) => {
    set({ isSending: false, isOpeningSession: true, error: null })
    try {
      const session = await getSession(sessionId)
      set({
        sessionId: session.session_id,
        ...emptyRunState,
        currentStatus: 'loaded',
        isOpeningSession: false,
        messages: session.messages
          .filter((message) => message.role === 'user' || message.role === 'assistant')
          .map((message, index) => ({
            id: `${session.session_id}-${index}`,
            role: message.role as 'user' | 'assistant',
            content: message.content,
          })),
      })
    } catch (error) {
      const message = error instanceof Error ? error.message : '打开历史会话失败'
      set({ isOpeningSession: false, error: message })
    }
  },
  renameCurrentOrSession: async (targetSessionId, title) => {
    const trimmedTitle = title.trim()
    if (!trimmedTitle) return
    try {
      await renameSession(targetSessionId, { title: trimmedTitle })
      await get().loadSessions()
    } catch (error) {
      const message = error instanceof Error ? error.message : '重命名会话失败'
      set({ error: message })
    }
  },
  deleteSession: async (targetSessionId) => {
    try {
      await deleteSessionRequest(targetSessionId)
      if (get().sessionId === targetSessionId) {
        get().newChat()
      }
      await get().loadSessions()
    } catch (error) {
      const message = error instanceof Error ? error.message : '删除会话失败'
      set({ error: message })
    }
  },
  newChat: () =>
    set({
      input: '',
      messages: [],
      sessionId: null,
      ...emptyRunState,
      isSending: false,
      error: null,
    }),
  setInput: (input) => set({ input }),
  send: async () => {
    const text = get().input.trim()
    if (!text || get().isSending) return

    const userMessage: Message = { id: `user-${Date.now()}`, role: 'user', content: text }
    set((state) => ({
      input: '',
      isSending: true,
      error: null,
      messages: [...state.messages, userMessage],
    }))

    try {
      const response = await sendChatMessage({ message: text, session_id: get().sessionId })
      void get().loadSessions()
      set((state) => ({
        sessionId: response.session_id,
        currentRunId: response.run_id,
        currentStatus: response.status,
        trace: response.trace,
        tools: response.tools,
        filesTouched: response.files_touched,
        isSending: false,
        messages: [
          ...state.messages,
          { id: `assistant-${Date.now()}`, role: 'assistant', content: response.answer },
        ],
      }))
    } catch (error) {
      const message = error instanceof Error ? error.message : '请求 Echo 后端失败'
      set({ isSending: false, error: message })
    }
  },
}))
