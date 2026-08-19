import { X } from 'lucide-react'

import type { RunFileDiff } from '../types/api'

type DiffViewerProps = {
  diff: RunFileDiff | null
  loading: boolean
  error: string | null
  onClose: () => void
}

function statusLabel(status?: string) {
  switch (status) {
    case 'modified':
      return '有工作区变更'
    case 'current':
      return '当前文件内容'
    case 'missing':
      return '文件不存在'
    default:
      return status ?? ''
  }
}

function lineClassName(line: string) {
  if (line.startsWith('+++') || line.startsWith('---')) return 'text-slate-300'
  if (line.startsWith('+')) return 'bg-emerald-950/60 text-emerald-200'
  if (line.startsWith('-')) return 'bg-red-950/60 text-red-200'
  if (line.startsWith('@@')) return 'bg-sky-950/60 text-sky-200'
  return 'text-slate-300'
}

export function DiffViewer({ diff, loading, error, onClose }: DiffViewerProps) {
  if (!diff && !loading && !error) return null

  const lines = diff?.diff ? diff.diff.split('\n') : []

  return (
    <div className="rounded-2xl border border-slate-200 bg-slate-950 text-slate-100 shadow-sm">
      <div className="flex items-center justify-between gap-2 border-b border-slate-800 px-3 py-2">
        <div className="min-w-0">
          <div className="truncate text-xs font-medium">{diff?.path ?? '加载文件变化...'}</div>
          {diff && <div className="text-[11px] text-slate-400">{statusLabel(diff.status)}</div>}
        </div>
        <button className="rounded-md p-1 text-slate-400 hover:bg-slate-800 hover:text-white" onClick={onClose} title="关闭 diff">
          <X className="h-3.5 w-3.5" />
        </button>
      </div>
      <div className="max-h-72 overflow-auto p-3">
        {loading ? (
          <div className="text-xs text-slate-400">正在加载文件变化...</div>
        ) : error ? (
          <div className="text-xs text-red-300">{error}</div>
        ) : lines.length > 0 ? (
          <pre className="min-w-full text-[11px] leading-5">
            {lines.map((line, index) => (
              <div key={`${index}-${line}`} className={`whitespace-pre-wrap break-words px-2 ${lineClassName(line)}`}>
                {line || ' '}
              </div>
            ))}
          </pre>
        ) : (
          <div className="text-xs text-slate-400">暂无 diff 内容</div>
        )}
      </div>
    </div>
  )
}
