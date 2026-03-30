import type { AttackEvent, ConsoleState, IterationRecord } from '../types'

export type Action =
  | { type: 'ITERATION_START'; event: AttackEvent }
  | { type: 'PROMPT_READY'; event: AttackEvent }
  | { type: 'RESPONSE_READY'; event: AttackEvent }
  | { type: 'SCORE_DELTA'; event: AttackEvent }
  | { type: 'SCORE_READY'; event: AttackEvent }
  | { type: 'VULN_LOGGED'; event: AttackEvent }
  | { type: 'SESSION_END'; event: AttackEvent }
  | { type: 'RESET' }

export function initialState(sessionId: string, objective: string): ConsoleState {
  return {
    sessionId,
    objective,
    iterations: [],
    currentScore: 0,
    currentIteration: 0,
    stats: { attempts: 0, vulns: 0, strategiesUsed: new Set() },
    status: 'running',
  }
}

/** Return a fresh IterationRecord skeleton. */
function blankRecord(iteration: number): IterationRecord {
  return { iteration, strategy: '', prompt: '', response: '', score: 0, isVuln: false }
}

/** Return the last record in the array, or a blank one if empty. */
function lastOrBlank(iterations: IterationRecord[], iteration: number): IterationRecord {
  return iterations.length > 0 ? { ...iterations[iterations.length - 1] } : blankRecord(iteration)
}

export function consoleReducer(state: ConsoleState, action: Action): ConsoleState {
  switch (action.type) {
    case 'ITERATION_START': {
      const n = action.event.iteration ?? state.currentIteration + 1
      return {
        ...state,
        currentIteration: n,
        currentScore: 0,
        iterations: [...state.iterations, blankRecord(n)],
        stats: { ...state.stats, attempts: state.stats.attempts + 1 },
      }
    }

    case 'PROMPT_READY': {
      const ev = action.event
      const updated = state.iterations.map((r) =>
        r.iteration === (ev.iteration ?? state.currentIteration)
          ? { ...r, strategy: ev.strategy ?? '', prompt: ev.prompt ?? '' }
          : r,
      )
      const strategies = new Set(state.stats.strategiesUsed)
      if (ev.strategy) strategies.add(ev.strategy)
      return { ...state, iterations: updated, stats: { ...state.stats, strategiesUsed: strategies } }
    }

    case 'RESPONSE_READY': {
      const ev = action.event
      const updated = state.iterations.map((r) =>
        r.iteration === (ev.iteration ?? state.currentIteration)
          ? { ...r, response: ev.response ?? '' }
          : r,
      )
      return { ...state, iterations: updated }
    }

    case 'SCORE_DELTA': {
      // Increment the live gauge by the delta for animation.
      return { ...state, currentScore: action.event.score ?? state.currentScore }
    }

    case 'SCORE_READY': {
      const ev = action.event
      const score = ev.score ?? 0
      const updated = state.iterations.map((r) =>
        r.iteration === (ev.iteration ?? state.currentIteration) ? { ...r, score } : r,
      )
      return { ...state, currentScore: score, iterations: updated }
    }

    case 'VULN_LOGGED': {
      const ev = action.event
      const updated = state.iterations.map((r) =>
        r.iteration === (ev.iteration ?? state.currentIteration)
          ? { ...r, isVuln: true, violatedPolicy: ev.violated_policy ?? '' }
          : r,
      )
      return {
        ...state,
        iterations: updated,
        stats: { ...state.stats, vulns: state.stats.vulns + 1 },
      }
    }

    case 'SESSION_END':
      return { ...state, status: 'ended', currentScore: 0 }

    case 'RESET':
      return { ...state, status: 'idle', iterations: [], currentScore: 0, currentIteration: 0 }

    default:
      return state
  }
}

/** Convert a raw AttackEvent to a reducer Action, or null to skip. */
export function eventToAction(ev: AttackEvent): Action | null {
  switch (ev.event_type) {
    case 'iteration_start': return { type: 'ITERATION_START', event: ev }
    case 'prompt_ready':    return { type: 'PROMPT_READY', event: ev }
    case 'response_ready':  return { type: 'RESPONSE_READY', event: ev }
    case 'score_delta':     return { type: 'SCORE_DELTA', event: ev }
    case 'score_ready':     return { type: 'SCORE_READY', event: ev }
    case 'vuln_logged':     return { type: 'VULN_LOGGED', event: ev }
    case 'session_end':     return { type: 'SESSION_END', event: ev }
    default:                return null
  }
}
