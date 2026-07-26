import { useEffect, useState } from 'react'
import {
  Activity,
  BrainCircuit,
  CircleAlert,
  CloudMoon,
  Database,
  MemoryStick,
  MessageSquareText,
  MoonStar,
  Play,
  RefreshCw,
  Sparkles,
} from 'lucide-react'
import './App.css'

type Health = {
  status: 'ok' | 'degraded'
  database: string
  cactus: string
  models: { chat: string; embedding: string }
}

type Episode = {
  id: string
  kind: 'source' | 'retrieval'
  source_type: 'slack' | 'meeting' | 'recall'
  content: string
  summary: string | null
  participants: string[]
  occurred_at: string
  mem0_id: string | null
}

type ReplayResult = { processed: number; skipped: number; indexed: number }
type Mode = 'awake' | 'sleep' | 'procedural'

const modes: Array<{ id: Mode; label: string; icon: typeof Activity }> = [
  { id: 'awake', label: 'Awake', icon: Activity },
  { id: 'sleep', label: 'Sleep', icon: MoonStar },
  { id: 'procedural', label: 'Procedural', icon: BrainCircuit },
]

async function fetchSnapshot() {
  const [healthResponse, episodeResponse] = await Promise.all([
    fetch('/api/health'),
    fetch('/api/episodes'),
  ])
  if (!healthResponse.ok || !episodeResponse.ok) throw new Error('The Synapse API is not ready')
  return {
    health: (await healthResponse.json()) as Health,
    episodes: (await episodeResponse.json()) as Episode[],
  }
}

function App() {
  const [mode, setMode] = useState<Mode>('awake')
  const [health, setHealth] = useState<Health | null>(null)
  const [episodes, setEpisodes] = useState<Episode[]>([])
  const [loading, setLoading] = useState(true)
  const [replaying, setReplaying] = useState(false)
  const [result, setResult] = useState<ReplayResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  async function load() {
    setError(null)
    try {
      const snapshot = await fetchSnapshot()
      setHealth(snapshot.health)
      setEpisodes(snapshot.episodes)
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : 'Unable to load Synapse')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void fetchSnapshot()
      .then((snapshot) => {
        setHealth(snapshot.health)
        setEpisodes(snapshot.episodes)
      })
      .catch((loadError: unknown) => {
        setError(loadError instanceof Error ? loadError.message : 'Unable to load Synapse')
      })
      .finally(() => setLoading(false))
  }, [])

  async function replay() {
    setReplaying(true)
    setError(null)
    try {
      const response = await fetch('/api/demo/replay', { method: 'POST' })
      if (!response.ok) throw new Error('Replay failed')
      setResult((await response.json()) as ReplayResult)
      await load()
    } catch (replayError) {
      setError(replayError instanceof Error ? replayError.message : 'Replay failed')
    } finally {
      setReplaying(false)
    }
  }

  const title = mode === 'awake' ? 'System 1' : mode === 'sleep' ? 'System 2' : 'Procedural memory'
  return (
    <div className="app-shell">
      <aside className="mode-rail" aria-label="Memory modes">
        <div className="brand-mark" aria-label="Synapse"><Sparkles size={20} /></div>
        <nav>
          {modes.map(({ id, label, icon: Icon }) => (
            <button
              type="button"
              className={mode === id ? 'mode-button active' : 'mode-button'}
              key={id}
              onClick={() => setMode(id)}
              title={label}
              aria-label={label}
            >
              <Icon size={19} /><span>{label}</span>
            </button>
          ))}
        </nav>
      </aside>

      <main>
        <header className="topbar">
          <div><p className="eyebrow">Synapse / {mode}</p><h1>{title}</h1></div>
          <div className="health-row" aria-label="Provider status">
            <Status label="PostgreSQL" state={health?.database} icon={Database} />
            <Status label="Cactus / Gemma" state={health?.cactus} icon={MemoryStick} />
          </div>
        </header>

        {mode === 'awake' && (
          <AwakeView
            episodes={episodes}
            health={health}
            loading={loading}
            replaying={replaying}
            result={result}
            error={error}
            onReplay={replay}
            onRefresh={load}
          />
        )}
        {mode === 'sleep' && <ComingSoon mode="System 2" icon={CloudMoon} />}
        {mode === 'procedural' && <ComingSoon mode="Procedural memory" icon={BrainCircuit} />}
      </main>
    </div>
  )
}

function Status({ label, state, icon: Icon }: { label: string; state?: string; icon: typeof Activity }) {
  const available = state === 'available'
  return (
    <div className={available ? 'status available' : 'status degraded'}>
      <Icon size={15} /><span>{label}</span><i aria-label={available ? 'available' : 'unavailable'} />
    </div>
  )
}

function AwakeView({ episodes, health, loading, replaying, result, error, onReplay, onRefresh }: {
  episodes: Episode[]
  health: Health | null
  loading: boolean
  replaying: boolean
  result: ReplayResult | null
  error: string | null
  onReplay: () => Promise<void>
  onRefresh: () => Promise<void>
}) {
  const indexed = episodes.filter((episode) => episode.mem0_id).length
  return (
    <>
      <section className="control-band">
        <div className="control-copy">
          <span className="live-indicator"><i /> Event listener</span>
          <strong>{episodes.length} episodes observed</strong>
          <span>{indexed} indexed by Mem0</span>
        </div>
        <div className="actions">
          <button className="icon-button" type="button" onClick={() => void onRefresh()} title="Refresh"><RefreshCw size={17} /></button>
          <button className="primary-button" type="button" onClick={() => void onReplay()} disabled={replaying}>
            <Play size={16} fill="currentColor" />{replaying ? 'Replaying' : 'Replay mock day'}
          </button>
        </div>
      </section>

      {health?.cactus !== 'available' && (
        <div className="notice"><CircleAlert size={17} /><span>Cactus is offline. Episodes use deterministic System 1 observations and remain pending Mem0 indexing.</span></div>
      )}
      {result && <div className="result-strip"><span>{result.processed} processed</span><span>{result.skipped} already known</span><span>{result.indexed} indexed</span></div>}
      {error && <div className="error-strip">{error}</div>}

      <section className="timeline-section">
        <div className="section-heading">
          <div><p className="eyebrow">Episodic stream</p><h2>Immediate observations</h2></div>
          <span className="model-chip">{health?.models.chat ?? 'google/gemma-4-E2B-it'}</span>
        </div>
        {loading ? (
          <div className="empty-state">Loading episodes...</div>
        ) : episodes.length === 0 ? (
          <div className="empty-state"><MessageSquareText size={28} /><strong>No episodes yet</strong></div>
        ) : (
          <div className="timeline">
            {episodes.map((episode) => (
              <article className="episode-row" key={episode.id}>
                <div className={`source-icon ${episode.source_type}`}>
                  {episode.source_type === 'meeting' ? <Activity size={17} /> : <MessageSquareText size={17} />}
                </div>
                <div className="episode-main">
                  <div className="episode-meta"><strong>{episode.source_type}</strong><time>{formatTime(episode.occurred_at)}</time><span>{episode.participants.join(', ')}</span></div>
                  <p>{episode.summary ?? episode.content}</p>
                </div>
                <div className={episode.mem0_id ? 'memory-state indexed' : 'memory-state pending'}>{episode.mem0_id ? 'indexed' : 'pending'}</div>
              </article>
            ))}
          </div>
        )}
      </section>
    </>
  )
}

function ComingSoon({ mode, icon: Icon }: { mode: string; icon: typeof Activity }) {
  return <section className="coming-soon"><Icon size={34} /><p className="eyebrow">Next implementation slice</p><h2>{mode}</h2></section>
}

function formatTime(value: string) {
  return new Intl.DateTimeFormat('en', { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' }).format(new Date(value))
}

export default App