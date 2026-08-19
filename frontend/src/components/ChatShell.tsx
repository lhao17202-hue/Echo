import { useEffect, useState } from 'react'
import { ArrowUp, Bot, History } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import rehypeHighlight from 'rehype-highlight'
import remarkGfm from 'remark-gfm'
import TextareaAutosize from 'react-textarea-autosize'
import 'highlight.js/styles/github.css'

import { getConfigSummary, getRuntimeStatus } from '../lib/api'
import { useChatStore, type Message } from '../stores/chatStore'
import type { ConfigSummary, RuntimeStatus } from '../types/api'

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
        <div className={`markdown-body ${isUser ? 'markdown-body-invert' : ''}`}>
          <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}>
            {message.content}
          </ReactMarkdown>
        </div>
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
  const [config, setConfig] = useState<ConfigSummary | null>(null)
  const [runtime, setRuntime] = useState<RuntimeStatus | null>(null)
  const [isConnected, setConnected] = useState(true)

  useEffect(() => {
    Promise.all([getConfigSummary(), getRuntimeStatus()])
      .then(([configSummary, runtimeStatus]) => {
        setConfig(configSummary)
        setRuntime(runtimeStatus)
        setConnected(true)
      })
      .catch(() => setConnected(false))
  }, [])

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
          <div className="flex min-w-0 items-center gap-2 text-xs text-slate-500">
            <span className={`rounded-full px-2 py-1 font-medium ${isConnected ? 'bg-emerald-50 text-emerald-700' : 'bg-red-50 text-red-700'}`}>
              {isConnected ? '后端已连接' : '后端未连接'}
            </span>
            <span className="rounded-full bg-slate-100 px-2 py-1 font-medium text-slate-600">{config?.approval_policy || 'approval'}</span>
            {runtime && <span className="truncate">{runtime.tools} tools · {runtime.mcp_servers} MCP</span>}
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

function EmptyChat() {
  return (
    <div className="mb-8 rounded-3xl border border-dashed border-slate-200 bg-white/70 px-6 py-8 text-center shadow-sm">
      <div className="text-base font-semibold text-slate-800">开始一个 Echo 会话</div>
      <div className="mt-2 text-sm leading-6 text-slate-500">输入任务、提问或继续让本地 Agent 操作工作区。</div>
    </div>
  )
}

function statusLabel(status: string, isSending: boolean, hasSession: boolean) {
  if (isSending) return '运行中'
  if (!hasSession) return '新会话'
  if (status === 'loaded') return '已加载历史'
  if (status === 'completed') return '已完成'
  if (status === 'failed') return '运行失败'
  return status
}

export function ChatShell() {
  const messages = useChatStore((state) => state.messages)
  const currentRunId = useChatStore((state) => state.currentRunId)
  const currentStatus = useChatStore((state) => state.currentStatus)
  const sessionId = useChatStore((state) => state.sessionId)
  const isSending = useChatStore((state) => state.isSending)

  return (
    <main className="flex min-w-0 flex-1 flex-col bg-[#f8fafc]">
      <header className="flex h-14 items-center justify-between border-b border-slate-200 bg-white/80 px-4 backdrop-blur">
        <div className="flex items-center gap-2 text-sm font-medium text-slate-700">
          <History className="h-4 w-4" />
          Echo Agent
          <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs font-normal text-slate-500">
            {statusLabel(currentStatus, isSending, Boolean(sessionId))}
          </span>
        </div>
        <div className="flex min-w-0 items-center gap-2">
          {sessionId && <div className="hidden max-w-48 truncate rounded-full border border-slate-200 bg-white px-3 py-1 text-xs text-slate-500 md:block">Session {sessionId}</div>}
          {currentRunId && <div className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs text-slate-500">Run {currentRunId}</div>}
        </div>
      </header>

      <section className="flex-1 overflow-y-auto">
        <div className="mx-auto flex min-h-full w-full max-w-3xl flex-col justify-end gap-4 px-4 py-8">
          {messages.length === 0 && !isSending && <EmptyChat />}
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
