import type { ApprovalPolicy } from '../types/api'

export const APPROVAL_POLICIES = ['ask', 'auto', 'never', 'danger'] as const satisfies readonly ApprovalPolicy[]

export const APPROVAL_POLICY_LABELS: Record<ApprovalPolicy, string> = {
  ask: 'Ask',
  auto: 'Auto',
  never: 'Never',
  danger: 'Danger',
}

export function approvalPolicyChoices(current: ApprovalPolicy): ApprovalPolicy[] {
  return APPROVAL_POLICIES.filter((policy) => policy !== current)
}
