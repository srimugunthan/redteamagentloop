/** Mirrors console/backend/models.py AttackEvent */
export interface AttackEvent {
  event_type:
    | 'iteration_start'
    | 'prompt_ready'
    | 'response_ready'
    | 'score_delta'
    | 'score_ready'
    | 'vuln_logged'
    | 'session_end'
  session_id: string
  iteration?: number
  strategy?: string
  prompt?: string
  response?: string
  score?: number
  score_delta?: number
  violated_policy?: string
  error?: string
}

export interface IterationRecord {
  iteration: number
  strategy: string
  prompt: string
  response: string
  score: number
  isVuln: boolean
  violatedPolicy?: string
}

export interface ConsoleState {
  sessionId: string
  objective: string
  iterations: IterationRecord[]
  currentScore: number       // animating value driven by score_delta
  currentIteration: number   // in-flight iteration being built
  stats: {
    attempts: number
    vulns: number
    strategiesUsed: Set<string>
  }
  status: 'idle' | 'running' | 'ended' | 'error'
  errorMessage?: string
}
