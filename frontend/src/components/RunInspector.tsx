import { useEffect, useState } from 'react'

import { DiffViewer } from './DiffViewer'
import { getRunFileDiff, getRunFiles } from '../lib/api'
import { useChatStore } from '../stores/chatStore'
import type { RunFileDiff, RunFileSummary } from '../types/api'

function formatPayload(payload: Record<string, unknown>) {
  const entries = Object.entries(payload).filter(([, value]) => value !== null && value !== undefined && value !== '')
  if (entries.length === 0) return ''
  return entries
    .map(([key, value]) => `${key}: ${typeof value === 'string' ? value : JSON.stringify(value)}`)
    .join('\n')
}

function eventLabel(event: string) {
  const labels: Record<string, string> = {
    run_started: '运行开始',
    model_request: '请求模型',
    model_response: '模型响应',
    tool_started: '工具开始',
    tool_executed: '工具完成',
    tool_failed: '工具失败',
    file_changed: '文件变更',
    run_completed: '运行完成',
    run_failed: '运行失败',
  }
  return labels[event] ?? event
}

function fileStatusLabel(status: string) {
  switch (status) {
    case 'modified':
      return '已修改'
    case 'current':
      return '当前内容'
    case 'missing':
      return '不存在'
    default:
      return status
  }
}

export function RunInspector() {
  const currentRunId = useChatStore((state) => state.currentRunId)
  const currentStatus = useChatStore((state) => state.currentStatus)
  const trace = useChatStore((state) => state.trace)
  const tools = useChatStore((state) => state.tools)
  const filesTouched = useChatStore((state) => state.filesTouched)
  const [runFiles, setRunFiles] = useState<RunFileSummary[]>([])
  const [selectedPath, setSelectedPath] = useState<string | null>(null)
  const [selectedDiff, setSelectedDiff] = useState<RunFileDiff | null>(null)
  const [isDiffLoading, setIsDiffLoading] = useState(false)
  const [diffError, setDiffError] = useState<string | null>(null)

  useEffect(() => {
    setSelectedPath(null)
    setSelectedDiff(null)
    setDiffError(null)
    if (!currentRunId) {
      setRunFiles([])
      return
    }
    void getRunFiles(currentRunId)
      .then((files) => setRunFiles(files))
      .catch(() => setRunFiles([]))
  }, [currentRunId])

  const visibleFiles = runFiles.length > 0 ? runFiles : filesTouched.map((path) => ({ path, status: 'modified' }))

  function openDiff(path: string) {
    if (!currentRunId) return
    setSelectedPath(path)
    setIsDiffLoading(true)
    setDiffError(null)
    setSelectedDiff(null)
    void getRunFileDiff(currentRunId, path)
      .then(setSelectedDiff)
      .catch((error) => setDiffError(error instanceof Error ? error.message : '加载 diff 失败'))
      .finally(() => setIsDiffLoading(false))
  }

  return (
    <aside className="hidden w-80 shrink-0 border-l border-slate-200 bg-white xl:flex xl:flex-col">
      <div className="border-b border-slate-200 p-4">
        <div className="text-sm font-semibold text-slate-900">Run Inspector</div>
        <div className="mt-1 text-xs text-slate-500">{currentRunId ? currentRunId : '发送消息后会显示执行详情'}</div>
      </div>

      <div className="space-y-5 overflow-y-auto p-4 text-sm">
        <section>
          <div className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-400">状态</div>
          <div className={`rounded-xl border px-3 py-2 text-slate-700 ${currentStatus === 'failed' ? 'border-red-200 bg-red-50 text-red-700' : 'border-slate-200 bg-slate-50'}`}>
            {currentStatus}
          </div>
        </section>

        <section>
          <div className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-400">Timeline</div>
          <div className="space-y-2">
            {trace.length === 0 ? (
              <div className="rounded-xl border border-dashed border-slate-200 px-3 py-4 text-xs leading-5 text-slate-400">
                {currentRunId ? '暂无 trace 事件' : 'Agent 运行时会在这里显示模型、工具和文件事件'}
              </div>
            ) : (
              trace.map((event, index) => {
                const payload = formatPayload(event.payload)
                return (
                  <div key={event.event_id ?? `${event.event}-${index}`} className="rounded-xl border border-slate-200 px-3 py-2">
                    <div className="flex items-center justify-between gap-2">
                      <div className="font-medium text-slate-800">{eventLabel(event.event)}</div>
                      <div className="rounded-full bg-slate-100 px-2 py-0.5 text-[10px] text-slate-500">{event.event}</div>
                    </div>
                    <div className="mt-1 text-xs text-slate-400">{event.created_at ?? event.timestamp ?? 'no timestamp'}</div>
                    {payload && <pre className="mt-2 max-h-28 overflow-auto whitespace-pre-wrap rounded-lg bg-slate-50 p-2 text-[11px] text-slate-500">{payload}</pre>}
                  </div>
                )
              })
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
                    <span className={`rounded-full px-2 py-0.5 text-[10px] ${tool.success === false ? 'bg-red-50 text-red-600' : 'bg-emerald-50 text-emerald-700'}`}>
                      {tool.success === false ? 'failed' : 'ok'}
                    </span>
                  </div>
                  {tool.input_summary && <div className="mt-1 text-xs text-slate-500">输入：{tool.input_summary}</div>}
                  {tool.output_summary && <div className="mt-1 text-xs text-slate-500">输出：{tool.output_summary}</div>}
                </div>
              ))
            )}
          </div>
        </section>

        <section>
          <div className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-400">Files</div>
          <div className="space-y-1">
            {visibleFiles.length === 0 ? (
              <div className="rounded-xl border border-dashed border-slate-200 px-3 py-4 text-xs text-slate-400">暂无文件变更</div>
            ) : (
              visibleFiles.map((file) => {
                const isSelected = selectedPath === file.path
                return (
                  <button
                    key={file.path}
                    className={`flex w-full items-center justify-between gap-2 rounded-lg px-2 py-1 text-left text-xs hover:bg-slate-100 ${isSelected ? 'bg-slate-200 text-slate-900' : 'bg-slate-50 text-slate-600'}`}
                    onClick={() => openDiff(file.path)}
                  >
                    <span className="truncate">{file.path}</span>
                    <span className="shrink-0 text-[10px] text-slate-400">{isSelected && isDiffLoading ? '加载中' : fileStatusLabel(file.status)}</span>
                  </button>
                )
              })
            )}
          </div>
        </section>

        <DiffViewer diff={selectedDiff} loading={isDiffLoading} error={diffError} onClose={() => { setSelectedPath(null); setSelectedDiff(null); setDiffError(null) }} />
      </div>
    </aside>
  )
}
