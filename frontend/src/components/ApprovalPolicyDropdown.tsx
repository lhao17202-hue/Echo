import { useState } from 'react'

import { approvalPolicyChoices, APPROVAL_POLICY_LABELS } from '../lib/approvalPolicy'
import type { ApprovalPolicy } from '../types/api'

type ApprovalPolicyDropdownProps = {
  value: ApprovalPolicy
  disabled?: boolean
  compact?: boolean
  placement?: 'top' | 'bottom'
  onChange: (value: ApprovalPolicy) => void
}

export function ApprovalPolicyDropdown({ value, disabled = false, compact = false, placement = 'bottom', onChange }: ApprovalPolicyDropdownProps) {
  const [open, setOpen] = useState(false)
  const choices = approvalPolicyChoices(value)

  function choose(policy: ApprovalPolicy) {
    setOpen(false)
    onChange(policy)
  }

  return (
    <div className="relative inline-block text-left">
      <button
        type="button"
        className={
          compact
            ? 'inline-flex h-7 items-center rounded-full border border-slate-200 bg-white px-3 text-xs font-medium text-slate-600 shadow-sm transition hover:border-slate-300 hover:bg-slate-50 disabled:cursor-not-allowed disabled:text-slate-400'
            : 'inline-flex h-8 items-center rounded-xl border border-slate-200 bg-white px-3 text-sm font-semibold text-slate-700 shadow-sm transition hover:border-slate-300 hover:bg-slate-50 disabled:cursor-not-allowed disabled:text-slate-400'
        }
        aria-haspopup="menu"
        aria-expanded={open}
        disabled={disabled}
        onClick={() => setOpen((current) => !current)}
      >
        {APPROVAL_POLICY_LABELS[value]}
      </button>
      {open && !disabled && (
        <div
          className={`absolute left-0 z-50 min-w-full overflow-hidden rounded-2xl border border-slate-200 bg-white p-1 shadow-xl shadow-slate-200/80 ${placement === 'top' ? 'bottom-full mb-2' : 'top-full mt-2'}`}
          role="menu"
        >
          {choices.map((policy) => (
            <button
              key={policy}
              type="button"
              className="block w-full rounded-xl px-3 py-2 text-left text-xs font-medium text-slate-700 transition hover:bg-slate-900 hover:text-white"
              role="menuitem"
              onClick={() => choose(policy)}
            >
              {APPROVAL_POLICY_LABELS[policy]}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
