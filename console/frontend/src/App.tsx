import { useState, useRef } from 'react'
import { useAttackStream } from './hooks/useAttackStream'
import { ScoreGauge } from './components/ScoreGauge'
import type { IterationRecord } from './types'

// ---------------------------------------------------------------------------
// Start form
// ---------------------------------------------------------------------------

interface StartFormProps {
  onStart: (sessionId: string, objective: string) => void
}

function StartForm({ onStart }: StartFormProps) {
  const [objective, setObjective] = useState('elicit unlicensed investment advice')
  const [systemPrompt, setSystemPrompt] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setLoading(true)
    setError('')
    try {
      const res = await fetch('/api/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ objective, system_prompt: systemPrompt }),
      })
      if (!res.ok) throw new Error(`Server error: ${res.status}`)
      const { session_id } = await res.json() as { session_id: string }
      onStart(session_id, objective)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-6">
      <div className="w-full max-w-xl">
        <h1 className="text-2xl font-semibold text-red-400 mb-1 tracking-tight">
          RedTeamAgentLoop
        </h1>
        <p className="text-gray-500 text-sm mb-8">Live Attack Console</p>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs text-gray-400 mb-1 uppercase tracking-widest">
              Objective
            </label>
            <textarea
              className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 text-sm text-gray-100
                         placeholder-gray-600 focus:outline-none focus:border-red-500 resize-none"
              rows={3}
              value={objective}
              onChange={(e) => setObjective(e.target.value)}
              placeholder="What should the target NOT do?"
              required
            />
          </div>

          <div>
            <label className="block text-xs text-gray-400 mb-1 uppercase tracking-widest">
              System Prompt <span className="text-gray-600">(optional)</span>
            </label>
            <textarea
              className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 text-sm text-gray-100
                         placeholder-gray-600 focus:outline-none focus:border-gray-600 resize-none"
              rows={2}
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              placeholder="Target system prompt to test against"
            />
          </div>

          {error && <p className="text-red-400 text-xs">{error}</p>}

          <button
            type="submit"
            disabled={loading || !objective.trim()}
            className="w-full bg-red-600 hover:bg-red-500 disabled:opacity-40 disabled:cursor-not-allowed
                       text-white text-sm font-semibold py-2.5 rounded transition-colors"
          >
            {loading ? 'Starting…' : '▶  Start Red-Team Run'}
          </button>
        </form>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Stats bar
// ---------------------------------------------------------------------------

interface StatsBarProps {
  attempts: number
  vulns: number
  strategies: number
  currentScore: number
  status: string
}

function StatsBar({ attempts, vulns, strategies, currentScore, status }: StatsBarProps) {
  const statusColor = status === 'ended' ? 'text-gray-500' : 'text-green-400 animate-pulse'
  return (
    <div className="flex items-center gap-6 px-4 py-2 bg-gray-900 border-b border-gray-800 text-xs">
      <span className={`font-semibold ${statusColor}`}>
        {status === 'ended' ? '■ DONE' : '● LIVE'}
      </span>
      <span className="text-gray-400">Attempts: <span className="text-gray-100">{attempts}</span></span>
      <span className="text-gray-400">
        Vulns: <span className={vulns > 0 ? 'text-red-400 font-bold' : 'text-gray-100'}>{vulns}</span>
      </span>
      <span className="text-gray-400">Strategies: <span className="text-gray-100">{strategies}</span></span>
      <div className="flex items-center gap-2 ml-auto">
        <span className="text-gray-500">Live score</span>
        <div className="w-28">
          <ScoreGauge score={currentScore} size="sm" />
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Iteration table row
// ---------------------------------------------------------------------------

function IterationRow({ rec }: { rec: IterationRecord }) {
  const [expanded, setExpanded] = useState(false)

  function rowBg() {
    if (rec.isVuln) return 'bg-red-950/40 border-l-2 border-red-500'
    if (rec.score >= 3) return 'bg-amber-950/20 border-l-2 border-amber-700'
    return 'border-l-2 border-transparent'
  }

  return (
    <div
      className={`px-4 py-2 border-b border-gray-800/60 cursor-pointer hover:bg-gray-800/30 transition-colors ${rowBg()}`}
      onClick={() => setExpanded((v) => !v)}
    >
      <div className="flex items-center gap-3">
        <span className="text-gray-600 w-6 text-right text-xs shrink-0">{rec.iteration}</span>
        <span className="text-xs px-1.5 py-0.5 rounded bg-gray-800 text-gray-400 shrink-0 max-w-36 truncate">
          {rec.strategy || '…'}
        </span>
        <span className="text-gray-400 text-xs truncate flex-1">
          {rec.prompt ? rec.prompt.slice(0, 80) + (rec.prompt.length > 80 ? '…' : '') : '…'}
        </span>
        <div className="w-32 shrink-0">
          <ScoreGauge score={rec.score} size="sm" />
        </div>
        {rec.isVuln && (
          <span className="text-red-400 text-xs font-bold shrink-0">VULN</span>
        )}
      </div>

      {expanded && (
        <div className="mt-2 ml-9 space-y-2 text-xs">
          <div>
            <span className="text-gray-500 block mb-0.5">Prompt</span>
            <p className="text-gray-300 whitespace-pre-wrap break-words bg-gray-900 rounded p-2">
              {rec.prompt || '(empty)'}
            </p>
          </div>
          <div>
            <span className="text-gray-500 block mb-0.5">Response</span>
            <p className="text-gray-300 whitespace-pre-wrap break-words bg-gray-900 rounded p-2">
              {rec.response || '(empty)'}
            </p>
          </div>
          {rec.isVuln && rec.violatedPolicy && (
            <div>
              <span className="text-gray-500 block mb-0.5">Violated policy</span>
              <p className="text-red-300 bg-gray-900 rounded p-2">{rec.violatedPolicy}</p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Live console (uses useAttackStream hook)
// ---------------------------------------------------------------------------

interface LiveConsoleProps {
  sessionId: string
  objective: string
  onReset: () => void
}

function LiveConsole({ sessionId, objective, onReset }: LiveConsoleProps) {
  const state = useAttackStream({ sessionId, objective })
  const bottomRef = useRef<HTMLDivElement>(null)

  return (
    <div className="flex flex-col h-screen">
      {/* Header */}
      <div className="px-4 py-3 bg-gray-900 border-b border-gray-800 flex items-center gap-3">
        <span className="text-red-400 font-semibold text-sm">RedTeamAgentLoop</span>
        <span className="text-gray-600">›</span>
        <span className="text-gray-300 text-sm truncate flex-1">{objective}</span>
        <button
          onClick={onReset}
          className="text-xs text-gray-500 hover:text-gray-300 transition-colors shrink-0"
        >
          ✕ New run
        </button>
      </div>

      {/* Stats */}
      <StatsBar
        attempts={state.stats.attempts}
        vulns={state.stats.vulns}
        strategies={state.stats.strategiesUsed.size}
        currentScore={state.currentScore}
        status={state.status}
      />

      {/* Column headers */}
      <div className="flex items-center gap-3 px-4 py-1.5 bg-gray-900/50 border-b border-gray-800 text-xs text-gray-600 uppercase tracking-wider">
        <span className="w-6 text-right shrink-0">#</span>
        <span className="w-36 shrink-0">Strategy</span>
        <span className="flex-1">Prompt</span>
        <span className="w-32 shrink-0">Score</span>
        <span className="w-10 shrink-0" />
      </div>

      {/* Iteration list */}
      <div className="flex-1 overflow-y-auto">
        {state.iterations.length === 0 && (
          <div className="text-center text-gray-600 text-sm mt-16">
            Waiting for first iteration…
          </div>
        )}
        {state.iterations.map((rec) => (
          <IterationRow key={rec.iteration} rec={rec} />
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Footer */}
      {state.status === 'ended' && (
        <div className="px-4 py-3 bg-gray-900 border-t border-gray-800 text-xs text-gray-500 flex items-center gap-4">
          <span>Session complete.</span>
          <span>{state.stats.attempts} iterations</span>
          <span className={state.stats.vulns > 0 ? 'text-red-400' : 'text-green-400'}>
            {state.stats.vulns} vulnerabilities found
          </span>
          <button
            onClick={onReset}
            className="ml-auto text-gray-400 hover:text-white transition-colors"
          >
            ▶ New run
          </button>
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Root App
// ---------------------------------------------------------------------------

export default function App() {
  const [session, setSession] = useState<{ id: string; objective: string } | null>(null)

  if (!session) {
    return <StartForm onStart={(id, obj) => setSession({ id, objective: obj })} />
  }

  return (
    <LiveConsole
      sessionId={session.id}
      objective={session.objective}
      onReset={() => setSession(null)}
    />
  )
}
