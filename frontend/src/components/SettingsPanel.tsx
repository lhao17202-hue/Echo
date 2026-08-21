import { useEffect, useState } from 'react'
import { RefreshCw, X } from 'lucide-react'

import { getConfigSummary, getRuntimeStatus, updateApprovalPolicy } from '../lib/api'
import { ApprovalPolicyDropdown } from './ApprovalPolicyDropdown'
import type { ApprovalPolicy, ConfigSummary, RuntimeStatus } from '../types/api'

type SettingsPanelProps = {
  open: boolean
  onClose: () => void
}

export function SettingsPanel({ open, onClose }: SettingsPanelProps) {
  const [config, setConfig] = useState<ConfigSummary | null>(null)
  const [runtime, setRuntime] = useState<RuntimeStatus | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isLoading, setLoading] = useState(false)
  const [isUpdatingPolicy, setUpdatingPolicy] = useState(false)

  function loadSettings() {
    setLoading(true)
    setError(null)
    Promise.all([getConfigSummary(), getRuntimeStatus()])
      .then(([configSummary, runtimeStatus]) => {
        setConfig(configSummary)
        setRuntime(runtimeStatus)
      })
      .catch((error) => {
        setError(error instanceof Error ? error.message : '加载设置失败')
      })
      .finally(() => setLoading(false))
  }

  function changeApprovalPolicy(approvalPolicy: ApprovalPolicy) {
    setUpdatingPolicy(true)
    setError(null)
    updateApprovalPolicy({ approval_policy: approvalPolicy })
      .then((configSummary) => {
        setConfig(configSummary)
        return getRuntimeStatus()
      })
      .then((runtimeStatus) => setRuntime(runtimeStatus))
      .catch((error) => {
        setError(error instanceof Error ? error.message : '更新审批模式失败')
      })
      .finally(() => setUpdatingPolicy(false))
  }

  useEffect(() => {
    if (!open) return
    loadSettings()
  }, [open])

  if (!open) return null

  return (
    <div className="fixed inset-y-0 right-0 z-20 w-full max-w-md border-l border-slate-200 bg-white shadow-2xl">
      <div className="flex h-14 items-center justify-between border-b border-slate-200 px-4">
        <div>
          <div className="text-sm font-semibold text-slate-900">设置</div>
          <div className="text-xs text-slate-500">Echo Web V3.1 Agent 能力总览</div>
        </div>
        <div className="flex items-center gap-1">
          <button
            className="rounded-lg p-2 text-slate-500 hover:bg-slate-100 hover:text-slate-900 disabled:opacity-50"
            disabled={isLoading}
            onClick={loadSettings}
            title="刷新设置"
          >
            <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
          </button>
          <button className="rounded-lg p-2 text-slate-500 hover:bg-slate-100 hover:text-slate-900" onClick={onClose} title="关闭设置">
            <X className="h-4 w-4" />
          </button>
        </div>
      </div>

      <div className="space-y-4 overflow-y-auto p-4 text-sm">
        {error && <div className="rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-red-700">{error}</div>}

        <section className="rounded-2xl border border-slate-200 p-4">
          <div className="mb-1 font-medium text-slate-900">模型配置</div>
          <div className="mb-3 text-xs leading-5 text-slate-500">这些配置决定 Web Agent 调用哪个模型；API Key 只显示是否配置，不显示明文。</div>
          <dl className="space-y-2 text-xs text-slate-600">
            <Row label="Provider" value={config?.provider ?? '加载中...'} />
            <Row label="Model" value={config?.model ?? '加载中...'} />
            <Row label="Base URL" value={config?.base_url ?? '加载中...'} />
            <Row label="API Key" value={config ? (config.api_key_configured ? '已配置' : '未配置') : '加载中...'} />
            <div className="flex items-center justify-between gap-3">
              <dt className="shrink-0 text-slate-400">Approval</dt>
              <dd>
                {config && (
                  <ApprovalPolicyDropdown
                    value={config.approval_policy}
                    disabled={isUpdatingPolicy}
                    onChange={changeApprovalPolicy}
                  />
                )}
              </dd>
            </div>
          </dl>
          {config?.approval_policy === 'danger' && (
            <div className="mt-3 rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-xs leading-5 text-red-800">
              Danger 模式会取消工作区路径沙箱限制，工具可以访问电脑上的任意路径。
            </div>
          )}
        </section>

        <section className="rounded-2xl border border-slate-200 p-4">
          <div className="mb-1 font-medium text-slate-900">Runtime 能力</div>
          <div className="mb-3 text-xs leading-5 text-slate-500">这些数字来自后端 runtime，帮助判断当前 Agent 能使用哪些本地能力和集成。</div>
          <dl className="space-y-2 text-xs text-slate-600">
            <Row label="Tools" value={`${runtime?.tools ?? '加载中...'} 个可用工具`} />
            <Row label="Background" value={`${runtime?.background_tasks ?? '加载中...'} 个后台任务`} />
            <Row label="Cron" value={`${runtime?.cron_tasks ?? '加载中...'} 个定时任务`} />
            <Row label="MCP" value={`${runtime?.mcp_servers ?? '加载中...'} 个 MCP 服务`} />
            <Row label="Approval" value={runtime?.approval_policy ?? '加载中...'} />
          </dl>
        </section>
      </div>
    </div>
  )
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-start justify-between gap-3">
      <dt className="shrink-0 text-slate-400">{label}</dt>
      <dd className="break-all text-right text-slate-700">{value}</dd>
    </div>
  )
}
