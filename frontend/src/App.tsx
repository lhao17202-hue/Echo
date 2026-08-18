import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ArrowUp, Bot, Folder, GitBranch, History, Menu, Plus, Settings, Terminal } from 'lucide-react'
import TextareaAutosize from 'react-textarea-autosize'
import { create } from 'zustand'

import './App.css'
import { sendChatMessage } from './lib/api'
import type { ToolCallSummary, TraceEventDTO } from './types/api'

type Message = {
  id: string
  role: 'user' | 'assistant'
  content: string
}

type ChatState = {
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
  setInput: (input: string) => void
  send: () => Promise<void>
}

const initialMessages: Message[] = []

const useChatStore = create<ChatState>((set, get) => ({
  input: '',
  messages: initialMessages,
  sessionId: null,
  currentRunId: null,
  currentStatus: 'idle',
  trace: [],
  tools: [],
  filesTouched: [],
  isSending: false,
  error: null,
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

const queryClient = new QueryClient()

function Sidebar() {
  const sessions = ['集成 Echo 主链路', '调试 MCP tools', '设计 Codex 风格前端']

  return (
    <aside className="hidden w-72 shrink-0 border-r border-slate-200 bg-slate-100/80 lg:flex lg:flex-col">
      <div className="flex h-14 items-center justify-between px-4">
        <div className="flex items-center gap-2 font-semibold text-slate-900">
          <Terminal className="h-5 w-5" />
          Echo
        </div>
        <button className="rounded-lg p-2 text-slate-500 hover:bg-slate-200 hover:text-slate-900">
          <Menu className="h-4 w-4" />
        </button>
      </div>

      <div className="px-3 py-2">
        <button className="flex w-full items-center gap-2 rounded-xl bg-white px-3 py-2 text-sm font-medium text-slate-800 shadow-sm ring-1 ring-slate-200 hover:bg-slate-50">
          <Plus className="h-4 w-4" />
          新对话
        </button>
      </div>

      <nav className="flex-1 space-y-6 overflow-y-auto px-3 py-4 text-sm">
        <section>
          <div className="mb-2 px-2 text-xs font-medium uppercase tracking-wide text-slate-400">项目</div>
          <div className="space-y-1">
            <a className="flex items-center gap-2 rounded-lg px-2 py-2 text-slate-700 hover:bg-slate-200/70" href="#">
              <Folder className="h-4 w-4" />
              Echo
            </a>
            <a className="flex items-center gap-2 rounded-lg px-2 py-2 text-slate-700 hover:bg-slate-200/70" href="#">
              <GitBranch className="h-4 w-4" />
              master
            </a>
          </div>
        </section>

        <section>
          <div className="mb-2 px-2 text-xs font-medium uppercase tracking-wide text-slate-400">最近对话</div>
          <div className="space-y-1">
            {sessions.map((session) => (
              <a key={session} className="block truncate rounded-lg px-2 py-2 text-slate-700 hover:bg-slate-200/70" href="#">
                {session}
              </a>
            ))}
          </div>
        </section>
      </nav>

      <div className="border-t border-slate-200 p-3">
        <button className="flex w-full items-center gap-2 rounded-lg px-2 py-2 text-sm text-slate-600 hover:bg-slate-200/70">
          <Settings className="h-4 w-4" />
          设置
        </button>
      </div>
    </aside>
  )
}

function ChatMessage({ message }: { message: Message }) {
  const isUser = message.role === 'user'

  return (
    <div className={`flex gap-3 ${isUser ? 'justify-end' : 'justify-start'}`}>
      {!isUser && (
        <div className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-slate-900 text-white">
          <Bot className="h-4 w-4" />
        </div>
      )}
      <div
        className={
          isUser
            ? 'max-w-[75%] rounded-2xl bg-slate-900 px-4 py-2.5 text-sm leading-6 text-white'
            : 'max-w-[75%] rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm leading-6 text-slate-800 shadow-sm'
        }
      >
        {message.content}
      </div>
    </div>
  )
}

function ChatInput() {
  const input = useChatStore((state) => state.input)
  const isSending = useChatStore((state) => state.isSending)
  const error = useChatStore((state) => state.error)
  const setInput = useChatStore((state) => state.setInput)
  const send = useChatStore((state) => state.send)

  return (
    <div className="mx-auto w-full max-w-3xl px-4 pb-6">
      {error && <div className="mb-3 rounded-xl border border-red-200 bg-red-50 px-4 py-2 text-sm text-red-700">{error}</div>}
      <div className="rounded-3xl border border-slate-200 bg-white p-3 shadow-xl shadow-slate-200/70">
        <TextareaAutosize
          className="max-h-40 w-full resize-none border-0 bg-transparent px-2 py-2 text-sm leading-6 text-slate-900 outline-none placeholder:text-slate-400 disabled:text-slate-400"
          minRows={2}
          placeholder="向 Echo 发送消息..."
          value={input}
          disabled={isSending}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
              event.preventDefault()
              void send()
            }
          }}
        />
        <div className="flex items-center justify-between pt-2">
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <span className="rounded-full bg-slate-100 px-2 py-1 font-medium text-slate-600">Auto</span>
            <span>本地</span>
          </div>
          <button
            className="flex h-9 w-9 items-center justify-center rounded-full bg-slate-900 text-white shadow-sm hover:bg-slate-700 disabled:cursor-not-allowed disabled:bg-slate-300"
            disabled={!input.trim() || isSending}
            onClick={() => void send()}
          >
            <ArrowUp className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  )
}

function RunInspector() {
  const currentRunId = useChatStore((state) => state.currentRunId)
  const currentStatus = useChatStore((state) => state.currentStatus)
  const trace = useChatStore((state) => state.trace)
  const tools = useChatStore((state) => state.tools)
  const filesTouched = useChatStore((state) => state.filesTouched)

  return (
    <aside className="hidden w-80 shrink-0 border-l border-slate-200 bg-white xl:flex xl:flex-col">
      <div className="border-b border-slate-200 p-4">
        <div className="text-sm font-semibold text-slate-900">Run Inspector</div>
        <div className="mt-1 text-xs text-slate-500">{currentRunId ? currentRunId : '等待第一次运行'}</div>
      </div>

      <div className="space-y-5 overflow-y-auto p-4 text-sm">
        <section>
          <div className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-400">状态</div>
          <div className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-slate-700">{currentStatus}</div>
        </section>

        <section>
          <div className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-400">Timeline</div>
          <div className="space-y-2">
            {trace.length === 0 ? (
              <div className="rounded-xl border border-dashed border-slate-200 px-3 py-4 text-xs text-slate-400">暂无 trace 事件</div>
            ) : (
              trace.map((event, index) => (
                <div key={event.event_id ?? `${event.event}-${index}`} className="rounded-xl border border-slate-200 px-3 py-2">
                  <div className="font-medium text-slate-800">{event.event}</div>
                  <div className="mt-1 text-xs text-slate-400">{event.created_at ?? event.timestamp ?? 'no timestamp'}</div>
                </div>
              ))
            )}
          </div>
        </section>

        <section>
          <div className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-400">Tools</div>
          <div className="space-y-2">
            {tools.length === 0 ? (
              <div className="rounded-xl border border-dashed border-slate-200 px-3 py-4 text-xs text-slate-400">暂无工具调用</div>
            ) : (
              tools.map((tool, index) => (
                <div key={`${tool.name}-${index}`} className="rounded-xl border border-slate-200 px-3 py-2">
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-medium text-slate-800">{tool.name}</span>
                    <span className="text-xs text-slate-400">{tool.success === false ? 'failed' : 'ok'}</span>
                  </div>
                  {tool.output_summary && <div className="mt-1 text-xs text-slate-500">{tool.output_summary}</div>}
                </div>
              ))
            )}
          </div>
        </section>

        <section>
          <div className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-400">Files</div>
          <div className="space-y-1">
            {filesTouched.length === 0 ? (
              <div className="rounded-xl border border-dashed border-slate-200 px-3 py-4 text-xs text-slate-400">暂无文件变更</div>
            ) : (
              filesTouched.map((file) => (
                <div key={file} className="truncate rounded-lg bg-slate-50 px-2 py-1 text-xs text-slate-600">
                  {file}
                </div>
              ))
            )}
          </div>
        </section>
      </div>
    </aside>
  )
}

function ChatShell() {
  const messages = useChatStore((state) => state.messages)
  const currentRunId = useChatStore((state) => state.currentRunId)
  const isSending = useChatStore((state) => state.isSending)

  return (
    <main className="flex min-w-0 flex-1 flex-col bg-[#f8fafc]">
      <header className="flex h-14 items-center justify-between border-b border-slate-200 bg-white/80 px-4 backdrop-blur">
        <div className="flex items-center gap-2 text-sm font-medium text-slate-700">
          <History className="h-4 w-4" />
          Echo Agent
        </div>
        {currentRunId && (
          <div className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs text-slate-500">
            Run {currentRunId}
          </div>
        )}
      </header>

      <section className="flex-1 overflow-y-auto">
        <div className="mx-auto flex min-h-full w-full max-w-3xl flex-col justify-end gap-4 px-4 py-8">
          {messages.map((message) => (
            <ChatMessage key={message.id} message={message} />
          ))}
          {isSending && <ChatMessage message={{ id: 'assistant-loading', role: 'assistant', content: 'Echo 正在思考...' }} />}
        </div>
      </section>

      <ChatInput />
    </main>
  )
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <div className="flex min-h-screen bg-slate-100 text-slate-950">
        <Sidebar />
        <ChatShell />
        <RunInspector />
      </div>
    </QueryClientProvider>
  )
}

export default App
