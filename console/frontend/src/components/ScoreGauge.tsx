interface ScoreGaugeProps {
  score: number   // 0–10
  size?: 'sm' | 'md'
}

function scoreColor(score: number): string {
  if (score >= 7) return '#ef4444'   // red-500
  if (score >= 3) return '#f59e0b'   // amber-500
  return '#22c55e'                   // green-500
}

/** Animated horizontal bar gauge driven by score (0–10). */
export function ScoreGauge({ score, size = 'md' }: ScoreGaugeProps) {
  const pct = (score / 10) * 100
  const color = scoreColor(score)
  const height = size === 'sm' ? 'h-1.5' : 'h-2.5'

  return (
    <div className="flex items-center gap-2 min-w-0">
      <div className={`flex-1 bg-gray-800 rounded-full overflow-hidden ${height}`}>
        <div
          className="h-full rounded-full"
          style={{
            width: `${pct}%`,
            backgroundColor: color,
            transition: 'width 0.1s ease-out, background-color 0.3s ease',
          }}
        />
      </div>
      <span
        className="text-xs font-semibold tabular-nums w-8 text-right shrink-0"
        style={{ color }}
      >
        {score.toFixed(1)}
      </span>
    </div>
  )
}
