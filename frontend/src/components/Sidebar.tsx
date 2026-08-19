import { useEffect, useState } from 'react'
import { Edit3, Folder, GitBranch, Menu, Plus, RefreshCw, Search, Settings, Terminal, Trash2 } from 'lucide-react'

import { getGitStatus, getWorkspaceInfo } from '../lib/api'
import { useChatStore } from '../stores/chatStore'
import type { GitStatus, WorkspaceInfo } from '../types/api'

type SidebarProps = {
  onOpenSettings: () => void
}

export function Sidebar({ onOpenSettings }: SidebarProps) {
  const sessions = useChatStore((state) => state.sessions)
  const sessionId = useChatStore((state) => state.sessionId)
  const sessionQuery = useChatStore((state) => state.sessionQuery)
  const isLoadingSessions = useChatStore((state) => state.isLoadingSessions)
  const isOpeningSession = useChatStore((state) => state.isOpeningSession)
  const loadSessions = useChatStore((state) => state.loadSessions)
  const setSessionQuery = useChatStore((state) => state.setSessionQuery)
  const openSession = useChatStore((state) => state.openSession)
  const renameCurrentOrSession = useChatStore((state) => state.renameCurrentOrSession)
  const deleteSession = useChatStore((state) => state.deleteSession)
  const newChat = useChatStore((state) => state.newChat)
  const [workspace, setWorkspace] = useState<WorkspaceInfo | null>(null)
  const [gitStatus, setGitStatus] = useState<GitStatus | null>(null)
  const [isWorkspaceLoading, setWorkspaceLoading] = useState(false)
  const [isMenuOpen, setMenuOpen] = useState(false)

  function refreshWorkspace() {
    setWorkspaceLoading(true)
    void Promise.all([getWorkspaceInfo(), getGitStatus()])
      .then(([workspaceInfo, status]) => {
        setWorkspace(workspaceInfo)
        setGitStatus(status)
      })
      .catch(() => {
        setWorkspace(null)
        setGitStatus(null)
      })
      .finally(() => setWorkspaceLoading(false))
  }

  useEffect(() => {
    void loadSessions()
    refreshWorkspace()
  }, [loadSessions])

  function renameSession(targetSessionId: string, currentTitle: string) {
    const title = window.prompt('重命名会话', currentTitle)
    if (title === null) return
    void renameCurrentOrSession(targetSessionId, title)
  }

  function removeSession(targetSessionId: string, title: string) {
    if (!window.confirm(`删除会话“${title}”？此操作不可撤销。`)) return
    void deleteSession(targetSessionId)
  }

  return (
    <aside className="hidden w-72 shrink-0 border-r border-slate-200 bg-slate-100/80 lg:flex lg:flex-col">
      <div className="flex h-14 items-center justify-between px-4">
        <div className="flex items-center gap-2 font-semibold text-slate-900">
          <Terminal className="h-5 w-5" />
          Echo
        </div>
        <div className="relative">
          <button
            className="rounded-lg p-2 text-slate-500 hover:bg-slate-200 hover:text-slate-900"
            title="菜单"
            onClick={() => setMenuOpen((open) => !open)}
          >
            <Menu className="h-4 w-4" />
          </button>
          {isMenuOpen && (
            <div className="absolute right-0 top-10 z-30 w-56 rounded-2xl border border-slate-200 bg-white p-2 text-sm shadow-xl shadow-slate-200/80">
              <button className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-slate-700 hover:bg-slate-100" onClick={() => { setMenuOpen(false); newChat() }}>
                <Plus className="h-4 w-4" />
                新建会话
              </button>
              <button className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-slate-700 hover:bg-slate-100" onClick={() => { setMenuOpen(false); void loadSessions() }}>
                <RefreshCw className="h-4 w-4" />
                刷新会话
              </button>
              <button className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-slate-700 hover:bg-slate-100" onClick={() => { setMenuOpen(false); refreshWorkspace() }}>
                <Folder className="h-4 w-4" />
                刷新工作区
              </button>
              <button className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-slate-700 hover:bg-slate-100" onClick={() => { setMenuOpen(false); onOpenSettings() }}>
                <Settings className="h-4 w-4" />
                Agent 设置
              </button>
            </div>
          )}
        </div>
      </div>

      <div className="px-3 py-2">
        <button
          className="flex w-full items-center gap-2 rounded-xl bg-white px-3 py-2 text-sm font-medium text-slate-800 shadow-sm ring-1 ring-slate-200 hover:bg-slate-50"
          onClick={newChat}
        >
          <Plus className="h-4 w-4" />
          新对话
        </button>
      </div>

      <nav className="flex-1 space-y-6 overflow-y-auto px-3 py-4 text-sm">
        <section>
          <div className="mb-2 flex items-center justify-between px-2 text-xs font-medium uppercase tracking-wide text-slate-400">
            <span>项目</span>
            <button
              className="rounded-md p-1 text-slate-400 hover:bg-slate-200/70 hover:text-slate-700 disabled:opacity-50"
              disabled={isWorkspaceLoading}
              onClick={refreshWorkspace}
              title="刷新工作区状态"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${isWorkspaceLoading ? 'animate-spin' : ''}`} />
            </button>
          </div>
          <div className="space-y-2 rounded-2xl border border-slate-200 bg-white/70 p-2">
            <div className="flex items-start gap-2 rounded-lg px-2 py-2 text-slate-700">
              <Folder className="mt-0.5 h-4 w-4 shrink-0" />
              <div className="min-w-0">
                <div className="truncate font-medium">{workspace?.name ?? 'Echo'}</div>
                <div className="mt-0.5 truncate text-[11px] text-slate-400" title={workspace?.root}>
                  {workspace?.root ?? '工作区路径加载中...'}
                </div>
              </div>
            </div>
            <div className="flex items-center gap-2 rounded-lg px-2 py-2 text-slate-700">
              <GitBranch className="h-4 w-4" />
              <span className="truncate">{gitStatus?.branch ?? 'unknown'}</span>
              {gitStatus?.dirty && <span className="ml-auto rounded-full bg-amber-100 px-1.5 py-0.5 text-[10px] text-amber-700">{gitStatus.changed_files.length}</span>}
            </div>
            {gitStatus?.dirty && (
              <div className="space-y-1 border-t border-slate-200 pt-2">
                <div className="px-2 text-[11px] font-medium text-slate-400">变更文件</div>
                {gitStatus.changed_files.slice(0, 8).map((file) => (
                  <div key={file} className="truncate rounded-md px-2 py-1 text-[11px] text-slate-500" title={file}>
                    {file}
                  </div>
                ))}
                {gitStatus.changed_files.length > 8 && <div className="px-2 text-[11px] text-slate-400">还有 {gitStatus.changed_files.length - 8} 个文件...</div>}
              </div>
            )}
          </div>
        </section>

        <section>
          <div className="mb-2 flex items-center justify-between px-2 text-xs font-medium uppercase tracking-wide text-slate-400">
            <span>最近对话</span>
            <button
              className="rounded-md p-1 text-slate-400 hover:bg-slate-200/70 hover:text-slate-700 disabled:opacity-50"
              disabled={isLoadingSessions}
              onClick={() => void loadSessions()}
              title="刷新历史对话"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${isLoadingSessions ? 'animate-spin' : ''}`} />
            </button>
          </div>
          <div className="mb-2 flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-2 py-1.5 text-slate-500">
            <Search className="h-3.5 w-3.5 shrink-0" />
            <input
              className="min-w-0 flex-1 bg-transparent text-xs text-slate-700 outline-none placeholder:text-slate-400"
              placeholder="搜索会话标题..."
              value={sessionQuery}
              onChange={(event) => void setSessionQuery(event.target.value)}
            />
          </div>
          <div className="space-y-1">
            {isLoadingSessions ? (
              <div className="rounded-lg px-2 py-2 text-xs text-slate-400">正在加载...</div>
            ) : sessions.length === 0 ? (
              <div className="rounded-lg px-2 py-2 text-xs text-slate-400">{sessionQuery ? '没有匹配的历史对话' : '暂无历史对话'}</div>
            ) : (
              sessions.map((session) => (
                <div
                  key={session.session_id}
                  className={`group flex items-start gap-1 rounded-lg px-2 py-2 text-slate-700 hover:bg-slate-200/70 ${
                    session.session_id === sessionId ? 'bg-slate-200/80 font-medium text-slate-900' : ''
                  }`}
                >
                  <button
                    className="min-w-0 flex-1 truncate text-left disabled:cursor-wait disabled:opacity-60"
                    disabled={isOpeningSession}
                    onClick={() => void openSession(session.session_id)}
                  >
                    <span className="block truncate">{session.title}</span>
                    {session.updated_at && <span className="mt-0.5 block truncate text-[11px] font-normal text-slate-400">{session.updated_at}</span>}
                  </button>
                  <div className="flex shrink-0 items-center gap-0.5 opacity-70 group-hover:opacity-100">
                    <button
                      className="rounded-md p-1 text-slate-400 hover:bg-white hover:text-slate-700"
                      title="重命名会话"
                      onClick={() => renameSession(session.session_id, session.title)}
                    >
                      <Edit3 className="h-3.5 w-3.5" />
                    </button>
                    <button
                      className="rounded-md p-1 text-slate-400 hover:bg-red-50 hover:text-red-600"
                      title="删除会话"
                      onClick={() => removeSession(session.session_id, session.title)}
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        </section>
      </nav>

      <div className="border-t border-slate-200 p-3">
        <button className="flex w-full items-center gap-2 rounded-lg px-2 py-2 text-sm text-slate-600 hover:bg-slate-200/70" onClick={onOpenSettings}>
          <Settings className="h-4 w-4" />
          设置
        </button>
      </div>
    </aside>
  )
}
