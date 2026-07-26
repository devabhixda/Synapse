import { useEffect, useState } from 'react'
import {
  Activity,
  BrainCircuit,
  Check,
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
  ScanSearch,
  ShieldCheck,
  Sparkles,
  X,
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
  external_attributes: Array<{
    provider: string
    field_group: string
    payload: Record<string, unknown>
    source_url: string | null
    confidence: number
  }>
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
type WorkflowPattern = {
  id: string
  key: string
  name: string
  description: string
  status: string
  occurrence_count: number
  confidence: number
  evidence_recall_ids: string[]
  first_seen_at: string
  last_seen_at: string
}
type SkillSpec = {
  id: string
  pattern_id: string
  name: string
  version: number
  status: string
  purpose: string
  triggers: string[]
  required_queries: string[]
  steps: string[]
  guardrails: string[]
  expected_output: Record<string, unknown>
  confidence: number
}
type Mode = 'awake' | 'sleep' | 'procedural'

const modes: Array<{ id: Mode; label: string; icon: typeof Activity }> = [
  { id: 'awake', label: 'Awake', icon: Activity },
  { id: 'sleep', label: 'Sleep', icon: MoonStar },
  { id: 'procedural', label: 'Procedural', icon: BrainCircuit },
]

async function fetchSnapshot() {
  const [healthResponse, episodeResponse, runsResponse, graphResponse, patternsResponse, skillsResponse] = await Promise.all([
    fetch('/api/health'),
    fetch('/api/episodes'),
    fetch('/api/system2/runs'),
    fetch('/api/system2/graph'),
    fetch('/api/procedural/patterns'),
    fetch('/api/procedural/skills'),
  ])
  if (!healthResponse.ok || !episodeResponse.ok || !runsResponse.ok || !graphResponse.ok || !patternsResponse.ok || !skillsResponse.ok) {
    throw new Error('The Synapse API is not ready')
  }
  return {
    health: (await healthResponse.json()) as Health,
    episodes: (await episodeResponse.json()) as Episode[],
    runs: (await runsResponse.json()) as ConsolidationRun[],
    graph: (await graphResponse.json()) as SemanticGraph,
    patterns: (await patternsResponse.json()) as WorkflowPattern[],
    skills: (await skillsResponse.json()) as SkillSpec[],
  }
}

function App() {
  const [mode, setMode] = useState<Mode>('awake')
  const [health, setHealth] = useState<Health | null>(null)
  const [episodes, setEpisodes] = useState<Episode[]>([])
  const [loading, setLoading] = useState(true)
  const [replaying, setReplaying] = useState(false)
  const [consolidating, setConsolidating] = useState(false)
  const [detecting, setDetecting] = useState(false)
  const [result, setResult] = useState<ReplayResult | null>(null)
  const [runs, setRuns] = useState<ConsolidationRun[]>([])
  const [graph, setGraph] = useState<SemanticGraph>({ nodes: [], edges: [] })
  const [patterns, setPatterns] = useState<WorkflowPattern[]>([])
  const [skills, setSkills] = useState<SkillSpec[]>([])
  const [error, setError] = useState<string | null>(null)

  async function load() {
    setError(null)
    try {
      const snapshot = await fetchSnapshot()
      setHealth(snapshot.health)
      setEpisodes(snapshot.episodes)
      setRuns(snapshot.runs)
      setGraph(snapshot.graph)
      setPatterns(snapshot.patterns)
      setSkills(snapshot.skills)
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
        setPatterns(snapshot.patterns)
        setSkills(snapshot.skills)
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

  async function detectPatterns() {
    setDetecting(true)
    setError(null)
    try {
      const response = await fetch('/api/procedural/detect', { method: 'POST' })
      if (!response.ok) throw new Error('Pattern detection failed')
      await load()
    } catch (detectionError) {
      setError(detectionError instanceof Error ? detectionError.message : 'Pattern detection failed')
    } finally {
      setDetecting(false)
    }
  }

  async function updateSkillStatus(skillId: string, status: string) {
    setError(null)
    try {
      const response = await fetch(`/api/procedural/skills/${skillId}/status`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status }),
      })
      if (!response.ok) throw new Error('Skill review failed')
      await load()
    } catch (reviewError) {
      setError(reviewError instanceof Error ? reviewError.message : 'Skill review failed')
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
        {mode === 'procedural' && (
          <ProceduralView
            patterns={patterns}
            skills={skills}
            detecting={detecting}
            error={error}
            onDetect={detectPatterns}
            onRefresh={load}
            onStatus={updateSkillStatus}
          />
        )}
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
                  <div><strong>{node.name}</strong>{node.aliases.length > 0 && <small>aka {node.aliases.join(', ')}</small>}{node.external_attributes.length > 0 && <small className="external-source">enriched by {node.external_attributes.map((item) => item.provider).join(', ')}</small>}<p>{node.description}</p></div>
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

function ProceduralView({ patterns, skills, detecting, error, onDetect, onRefresh, onStatus }: {
  patterns: WorkflowPattern[]
  skills: SkillSpec[]
  detecting: boolean
  error: string | null
  onDetect: () => Promise<void>
  onRefresh: () => Promise<void>
  onStatus: (skillId: string, status: string) => Promise<void>
}) {
  const patternById = new Map(patterns.map((pattern) => [pattern.id, pattern]))
  return (
    <>
      <section className="control-band procedural-controls">
        <div className="control-copy">
          <span className="procedural-indicator"><BrainCircuit size={15} /> Pattern synthesis</span>
          <strong>{patterns.length} workflow patterns</strong>
          <span>{skills.filter((skill) => skill.status === 'approved').length} approved skills</span>
        </div>
        <div className="actions">
          <button className="icon-button" type="button" onClick={() => void onRefresh()} title="Refresh"><RefreshCw size={17} /></button>
          <button className="primary-button procedural-button" type="button" onClick={() => void onDetect()} disabled={detecting}>
            <ScanSearch size={17} />{detecting ? 'Detecting' : 'Detect patterns'}
          </button>
        </div>
      </section>
      {error && <div className="error-strip">{error}</div>}
      <section className="procedural-section">
        <div className="section-heading">
          <div><p className="eyebrow">Procedural memory</p><h2>Observed workflows</h2></div>
          <span className="model-chip">minimum 2 occurrences</span>
        </div>
        {skills.length === 0 ? (
          <div className="empty-state"><BrainCircuit size={28} /><strong>Repeat recall workflows, then detect patterns</strong></div>
        ) : (
          <div className="skill-browser">
            {skills.map((skill) => {
              const pattern = patternById.get(skill.pattern_id)
              return (
                <article className="skill-panel" key={skill.id}>
                  <header className="skill-header">
                    <div>
                      <span className={`skill-status status-${skill.status}`}>{skill.status}</span>
                      <h3>{skill.name.replaceAll('_', ' ')}</h3>
                      <p>{skill.purpose}</p>
                    </div>
                    <div className="skill-metrics">
                      <span>v{skill.version}</span>
                      <strong>{pattern?.occurrence_count ?? 0} recalls</strong>
                      <span>{Math.round(skill.confidence * 100)}% confidence</span>
                    </div>
                  </header>
                  <div className="skill-content">
                    <section><h4>Triggers</h4><ul>{skill.triggers.map((item) => <li key={item}>{item}</li>)}</ul></section>
                    <section><h4>Workflow</h4><ol>{skill.steps.map((item) => <li key={item}>{item}</li>)}</ol></section>
                    <section><h4><ShieldCheck size={14} /> Guardrails</h4><ul>{skill.guardrails.map((item) => <li key={item}>{item}</li>)}</ul></section>
                  </div>
                  <footer className="skill-actions">
                    <span>{pattern?.evidence_recall_ids.length ?? 0} recall events as evidence</span>
                    <div>
                      <button type="button" className="reject-button" onClick={() => void onStatus(skill.id, 'rejected')}><X size={15} /> Reject</button>
                      <button type="button" className="approve-button" onClick={() => void onStatus(skill.id, 'approved')}><Check size={15} /> Approve</button>
                    </div>
                  </footer>
                </article>
              )
            })}
          </div>
        )}
      </section>
    </>
  )
}

function formatTime(value: string) {
  return new Intl.DateTimeFormat('en', { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' }).format(new Date(value))
}

export default App