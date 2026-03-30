import { useEffect, useReducer, useRef } from 'react'
import type { AttackEvent } from '../types'
import { consoleReducer, eventToAction, initialState } from '../store/consoleReducer'

const MAX_RETRIES = 5
const BACKOFF_BASE_MS = 1000

interface UseAttackStreamOptions {
  sessionId: string
  objective: string
}

/**
 * Opens an EventSource to /api/stream/{sessionId} and pipes events into the
 * consoleReducer.  Reconnects automatically on error with exponential backoff.
 * Closes the connection once a session_end event is received or on unmount.
 */
export function useAttackStream({ sessionId, objective }: UseAttackStreamOptions) {
  const [state, dispatch] = useReducer(consoleReducer, initialState(sessionId, objective))
  const retries = useRef(0)
  const esRef = useRef<EventSource | null>(null)
  const closedRef = useRef(false)

  useEffect(() => {
    closedRef.current = false

    function connect() {
      if (closedRef.current) return

      const es = new EventSource(`/api/stream/${sessionId}`)
      esRef.current = es

      es.onmessage = (msgEvent) => {
        try {
          const attackEvent: AttackEvent = JSON.parse(msgEvent.data as string)
          const action = eventToAction(attackEvent)
          if (action) dispatch(action)
          if (attackEvent.event_type === 'session_end') {
            closedRef.current = true
            es.close()
          }
        } catch {
          // Malformed JSON — ignore
        }
      }

      es.onerror = () => {
        es.close()
        if (closedRef.current) return
        if (retries.current >= MAX_RETRIES) {
          dispatch({ type: 'SESSION_END', event: { event_type: 'session_end', session_id: sessionId, error: 'Connection lost' } })
          return
        }
        const delay = BACKOFF_BASE_MS * Math.pow(2, retries.current)
        retries.current += 1
        setTimeout(connect, delay)
      }
    }

    connect()

    return () => {
      closedRef.current = true
      esRef.current?.close()
    }
  }, [sessionId, objective])

  return state
}
