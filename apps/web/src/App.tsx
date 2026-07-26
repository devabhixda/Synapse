import { useEffect, useState } from 'react'
import {
  Activity,
  BrainCircuit,
  CircleAlert,
  CloudMoon,
  Database,
  GitMerge,
  MemoryStick,
  MessageSquareText,
  MoonStar,
  Network,
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
type ConsolidationRun = {
  id: string
  status: string
  stage: string
  provider: string
  episode_count: number
  entity_count: number
  relationship_count: number
  graph_diff: Record<string, number>
  error: string | null
  created_at: string
  completed_at: string | null
}
type GraphNode = {
  id: string
  kind: string
  name: string
  aliases: string[]
  description: string | null
  attributes: Record<string, string>
  confidence: number
}
type GraphEdge = {
  id: string
  source: string
  target: string
  predicate: string
  confidence: number
  evidence: Array<{ episode_id: string; source_type: string; excerpt: string }>
}
type SemanticGraph = { nodes: GraphNode[]; edges: GraphEdge[] }
type Mode = 'awake' | 'sleep' | 'procedural'

const modes: Array<{ id: Mode; label: string; icon: typeof Activity }> = [
  { id: 'awake', label: 'Awake', icon: Activity },
  { id: 'sleep', label: 'Sleep', icon: MoonStar },
  { id: 'procedural', label: 'Procedural', icon: BrainCircuit },
]

async function fetchSnapshot() {
  const [healthResponse, episodeResponse, runsResponse, graphResponse] = await Promise.all([
    fetch('/api/health'),
    fetch('/api/episodes'),
    fetch('/api/system2/runs'),
    fetch('/api/system2/graph'),
  ])
  if (!healthResponse.ok || !episodeResponse.ok || !runsResponse.ok || !graphResponse.ok) {
    throw new Error('The Synapse API is not ready')
  }
  return {
    health: (await healthResponse.json()) as Health,
    episodes: (await episodeResponse.json()) as Episode[],
    runs: (await runsResponse.json()) as ConsolidationRun[],
    graph: (await graphResponse.json()) as SemanticGraph,
  }
}

function App() {
  const [mode, setMode] = useState<Mode>('awake')
  const [health, setHealth] = useState<Health | null>(null)
  const [episodes, setEpisodes] = useState<Episode[]>([])
  const [loading, setLoading] = useState(true)
  const [replaying, setReplaying] = useState(false)
  const [consolidating, setConsolidating] = useState(false)
  const [result, setResult] = useState<ReplayResult | null>(null)
  const [runs, setRuns] = useState<ConsolidationRun[]>([])
  const [graph, setGraph] = useState<SemanticGraph>({ nodes: [], edges: [] })
  const [error, setError] = useState<string | null>(null)

  async function load() {
    setError(null)
    try {
      const snapshot = await fetchSnapshot()
      setHealth(snapshot.health)
      setEpisodes(snapshot.episodes)
      setRuns(snapshot.runs)
      setGraph(snapshot.graph)
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
        setRuns(snapshot.runs)
        setGraph(snapshot.graph)
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

  async function consolidate() {
    setConsolidating(true)
    setError(null)
    try {
      const response = await fetch('/api/system2/consolidate', { method: 'POST' })
      if (!response.ok) throw new Error('Sleep cycle failed')
      await load()
    } catch (consolidationError) {
      setError(
        consolidationError instanceof Error ? consolidationError.message : 'Sleep cycle failed',
      )
    } finally {
      setConsolidating(false)
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
        {mode === 'sleep' && (
          <SleepView
            runs={runs}
            graph={graph}
            consolidating={consolidating}
            error={error}
            onConsolidate={consolidate}
            onRefresh={load}
          />
        )}
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

function SleepView({ runs, graph, consolidating, error, onConsolidate, onRefresh }: {
  runs: ConsolidationRun[]
  graph: SemanticGraph
  consolidating: boolean
  error: string | null
  onConsolidate: () => Promise<void>
  onRefresh: () => Promise<void>
}) {
  const latest = runs[0]
  const nodeById = new Map(graph.nodes.map((node) => [node.id, node]))
  return (
    <>
      <section className="control-band sleep-controls">
        <div className="control-copy">
          <span className="sleep-indicator"><MoonStar size={15} /> Consolidation</span>
          <strong>{graph.nodes.length} semantic entities</strong>
          <span>{graph.edges.length} relationships</span>
        </div>
        <div className="actions">
          <button className="icon-button" type="button" onClick={() => void onRefresh()} title="Refresh"><RefreshCw size={17} /></button>
          <button className="primary-button sleep-button" type="button" onClick={() => void onConsolidate()} disabled={consolidating}>
            <CloudMoon size={17} />{consolidating ? 'Consolidating' : 'Start sleep cycle'}
          </button>
        </div>
      </section>

      {error && <div className="error-strip">{error}</div>}
      {latest && (
        <section className="run-summary">
          <div><p className="eyebrow">Latest run</p><strong>{latest.status}</strong><span>{formatTime(latest.created_at)}</span></div>
          <div><span>Episodes</span><strong>{latest.episode_count}</strong></div>
          <div><span>Entities</span><strong>{latest.entity_count}</strong></div>
          <div><span>Relationships</span><strong>{latest.relationship_count}</strong></div>
          <div className="diff-summary">
            <GitMerge size={16} />
            <span>{latest.graph_diff.entities_created ?? 0} new nodes</span>
            <span>{latest.graph_diff.relationships_created ?? 0} new edges</span>
          </div>
        </section>
      )}

      <section className="semantic-section">
        <div className="section-heading">
          <div><p className="eyebrow">Semantic memory</p><h2>Consolidated graph</h2></div>
          <span className="model-chip">{latest?.provider ?? 'No consolidation run'}</span>
        </div>
        {graph.nodes.length === 0 ? (
          <div className="empty-state"><Network size={28} /><strong>Run a sleep cycle to build the graph</strong></div>
        ) : (
          <div className="graph-browser">
            <div className="entity-list">
              <div className="pane-heading"><span>Entities</span><strong>{graph.nodes.length}</strong></div>
              {graph.nodes.map((node) => (
                <article className="entity-item" key={node.id}>
                  <span className={`entity-kind kind-${node.kind.toLowerCase()}`}>{node.kind}</span>
                  <div><strong>{node.name}</strong>{node.aliases.length > 0 && <small>aka {node.aliases.join(', ')}</small>}<p>{node.description}</p></div>
                  <span className="confidence">{Math.round(node.confidence * 100)}%</span>
                </article>
              ))}
            </div>
            <div className="edge-list">
              <div className="pane-heading"><span>Relationships</span><strong>{graph.edges.length}</strong></div>
              {graph.edges.map((edge) => (
                <article className="edge-item" key={edge.id}>
                  <div className="edge-statement">
                    <strong>{nodeById.get(edge.source)?.name ?? 'Unknown'}</strong>
                    <span>{edge.predicate.replaceAll('_', ' ')}</span>
                    <strong>{nodeById.get(edge.target)?.name ?? 'Unknown'}</strong>
                  </div>
                  <div className="evidence-count">{edge.evidence.length} evidence {edge.evidence.length === 1 ? 'episode' : 'episodes'}</div>
                  {edge.evidence[0] && <p>{edge.evidence[0].excerpt}</p>}
                </article>
              ))}
            </div>
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