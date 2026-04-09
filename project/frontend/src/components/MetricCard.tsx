export interface ReadonlyMetricCardProps {
  readonly label: string
  readonly value: string
  readonly hint?: string
  readonly tone?: 'cyan' | 'emerald' | 'amber' | 'violet'
}

const toneClassMap: Record<NonNullable<ReadonlyMetricCardProps['tone']>, string> = {
  cyan: 'metric-cyan',
  emerald: 'metric-emerald',
  amber: 'metric-amber',
  violet: 'metric-violet',
}

export function MetricCard({ label, value, hint, tone = 'cyan' }: ReadonlyMetricCardProps) {
  return (
    <article className={`metric-card ${toneClassMap[tone]}`}>
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      {hint ? <div className="metric-hint">{hint}</div> : null}
    </article>
  )
}
