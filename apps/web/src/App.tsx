import { useEffect, useState } from 'react'
import dagre from '@dagrejs/dagre'
import {
  Background,
  BackgroundVariant,
  Controls,
  Handle,
  MarkerType,
  MiniMap,
  Position,
  ReactFlow,
  type Edge,
  type Node,
  type NodeProps,
} from '@xyflow/react'
import {
  Activity,
  BrainCircuit,
  Check,
  CircleAlert,
  CloudMoon,
  Database,
  ExternalLink,
  Focus,
  GitMerge,
  List,
  MemoryStick,
  MessageSquareText,
  MoonStar,
  Network,
  RefreshCw,
  ScanSearch,
  ShieldCheck,
  Sparkles,
  X,
} from 'lucide-react'
import '@xyflow/react/dist/style.css'
import './App.css'

type Health = {
  status: 'ok' | 'degraded'
  database: string
  inference: string
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
  attributes: Record<string, string>
  confidence: number
  evidence: Array<{ episode_id: string; source_type: string; excerpt: string }>
}
type SemanticGraph = { nodes: GraphNode[]; edges: GraphEdge[] }
type SemanticNodeData = { graphNode: GraphNode; degree: number } & Record<string, unknown>
type SemanticEdgeData = { graphEdge: GraphEdge } & Record<string, unknown>
type SemanticFlowNode = Node<SemanticNodeData, 'semantic'>
type SemanticFlowEdge = Edge<SemanticEdgeData>
type GraphSelection = { type: 'node' | 'edge'; id: string } | null
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

const MISSING_GROUP = 'Add ?group_id=<account> to the URL to choose which memory to open'

// Read from the URL rather than a constant so the account is visible, shareable,
// and never guessed on the user's behalf.
function currentGroupId(): string {
  return new URLSearchParams(window.location.search).get('group_id')?.trim() ?? ''
}

function scoped(path: string): string {
  const groupId = currentGroupId()
  if (!groupId) throw new Error(MISSING_GROUP)
  const separator = path.includes('?') ? '&' : '?'
  return `${path}${separator}group_id=${encodeURIComponent(groupId)}`
}

async function fetchSnapshot() {
  const [healthResponse, episodeResponse, runsResponse, graphResponse, patternsResponse, skillsResponse] = await Promise.all([
    fetch('/api/health'),
    fetch(scoped('/api/episodes')),
    fetch(scoped('/api/system2/runs')),
    fetch(scoped('/api/system2/graph')),
    fetch(scoped('/api/procedural/patterns')),
    fetch(scoped('/api/procedural/skills')),
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
  const [consolidating, setConsolidating] = useState(false)
  const [detecting, setDetecting] = useState(false)
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

  async function consolidate() {
    setConsolidating(true)
    setError(null)
    try {
      const response = await fetch(scoped('/api/system2/consolidate'), { method: 'POST' })
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
      const response = await fetch(scoped('/api/procedural/detect'), { method: 'POST' })
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
      const response = await fetch(scoped(`/api/procedural/skills/${skillId}/status`), {
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
            <Status label="SQLite" state={health?.database} icon={Database} />
            <Status label="Inference" state={health?.inference} icon={MemoryStick} />
          </div>
        </header>

        {mode === 'awake' && (
          <AwakeView
            episodes={episodes}
            health={health}
            loading={loading}
            error={error}
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

function AwakeView({ episodes, health, loading, error, onRefresh }: {
  episodes: Episode[]
  health: Health | null
  loading: boolean
  error: string | null
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
        </div>
      </section>

      {health?.inference !== 'available' && (
        <div className="notice"><CircleAlert size={17} /><span>The model server is unreachable. Episodes use deterministic System 1 observations and remain pending Mem0 indexing.</span></div>
      )}
      {error && <div className="error-strip">{error}</div>}

      <section className="timeline-section">
        <div className="section-heading">
          <div><p className="eyebrow">Episodic stream</p><h2>Immediate observations</h2></div>
          <span className="model-chip">{health?.models.chat || 'chat model'}</span>
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
          <SemanticGraphWorkbench graph={graph} />
        )}
      </section>
    </>
  )
}

function SemanticGraphWorkbench({ graph }: { graph: SemanticGraph }) {
  const [view, setView] = useState<'graph' | 'details'>('graph')
  const [selection, setSelection] = useState<GraphSelection>(null)
  const [visibleKinds, setVisibleKinds] = useState<Set<string>>(new Set())
  const kinds = [...new Set(graph.nodes.map((node) => node.kind))].sort()
  const visibleNodes = visibleKinds.size === 0
    ? graph.nodes
    : graph.nodes.filter((node) => visibleKinds.has(node.kind))
  const visibleNodeIds = new Set(visibleNodes.map((node) => node.id))
  const visibleEdges = graph.edges.filter(
    (edge) => visibleNodeIds.has(edge.source) && visibleNodeIds.has(edge.target),
  )
  const nodeById = new Map(graph.nodes.map((node) => [node.id, node]))
  const selectedNode = selection?.type === 'node'
    ? graph.nodes.find((node) => node.id === selection.id) ?? null
    : null
  const selectedEdge = selection?.type === 'edge'
    ? graph.edges.find((edge) => edge.id === selection.id) ?? null
    : null
  const flowGraph = layoutSemanticGraph(visibleNodes, visibleEdges)
  flowGraph.nodes = flowGraph.nodes.map((node) => ({
    ...node,
    selected: selection?.type === 'node' && selection.id === node.id,
  }))
  flowGraph.edges = flowGraph.edges.map((edge) => ({
    ...edge,
    selected: selection?.type === 'edge' && selection.id === edge.id,
  }))
  const filterKey = visibleKinds.size === 0 ? 'all' : [...visibleKinds].sort().join('-')

  function toggleKind(kind: string) {
    setVisibleKinds((current) => {
      const next = new Set(current)
      if (next.has(kind)) next.delete(kind)
      else next.add(kind)
      return next
    })
    setSelection(null)
  }

  return (
    <div className="semantic-workbench">
      <div className="graph-toolbar">
        <div className="view-segment" aria-label="Semantic memory view">
          <button type="button" className={view === 'graph' ? 'active' : ''} onClick={() => setView('graph')}><Network size={14} /> Graph</button>
          <button type="button" className={view === 'details' ? 'active' : ''} onClick={() => setView('details')}><List size={14} /> Details</button>
        </div>
        <div className="kind-filters" aria-label="Entity type filters">
          {kinds.map((kind) => (
            <button
              type="button"
              key={kind}
              className={visibleKinds.size === 0 || visibleKinds.has(kind) ? `kind-filter kind-${kind.toLowerCase()} active` : 'kind-filter'}
              onClick={() => toggleKind(kind)}
            >
              <i />{kind}
            </button>
          ))}
          {visibleKinds.size > 0 && <button type="button" className="clear-filter" onClick={() => setVisibleKinds(new Set())}>All types</button>}
        </div>
      </div>

      {view === 'graph' ? (
        <div className="graph-stage">
          <div className="flow-canvas">
            <ReactFlow<SemanticFlowNode, SemanticFlowEdge>
              key={filterKey}
              nodes={flowGraph.nodes}
              edges={flowGraph.edges}
              nodeTypes={{ semantic: SemanticNode }}
              fitView
              fitViewOptions={{ padding: 0.08, minZoom: 0.56, maxZoom: 1.15 }}
              minZoom={0.4}
              maxZoom={1.8}
              nodesDraggable={false}
              nodesConnectable={false}
              elementsSelectable
              onNodeClick={(_, node) => setSelection({ type: 'node', id: node.id })}
              onEdgeClick={(_, edge) => setSelection({ type: 'edge', id: edge.id })}
              onPaneClick={() => setSelection(null)}
              proOptions={{ hideAttribution: true }}
            >
              <Background color="#d8d9d1" gap={24} size={1} variant={BackgroundVariant.Dots} />
              <Controls showInteractive={false} />
              <MiniMap
                pannable
                zoomable
                nodeColor={(node) => kindColor(String((node.data as SemanticNodeData | undefined)?.graphNode.kind ?? 'Concept'))}
                maskColor="rgba(245, 245, 239, .78)"
              />
            </ReactFlow>
            <div className="canvas-count"><Focus size={13} />{visibleNodes.length} nodes · {visibleEdges.length} edges</div>
          </div>
          <GraphInspector
            graph={graph}
            node={selectedNode}
            edge={selectedEdge}
            nodeById={nodeById}
            onSelectNode={(id) => setSelection({ type: 'node', id })}
          />
        </div>
      ) : (
        <GraphDetails graph={graph} nodeById={nodeById} onSelect={(next) => { setSelection(next); setView('graph') }} />
      )}
    </div>
  )
}

function SemanticNode({ data, selected }: NodeProps<SemanticFlowNode>) {
  const node = data.graphNode
  return (
    <div className={`semantic-node node-${node.kind.toLowerCase()} ${selected ? 'selected' : ''}`}>
      <Handle type="target" position={Position.Left} />
      <div className="node-heading"><span>{node.kind}</span><strong>{Math.round(node.confidence * 100)}%</strong></div>
      <p>{node.name}</p>
      <div className="node-foot"><span>{data.degree} links</span>{node.external_attributes.length > 0 && <span className="enriched-dot">enriched</span>}</div>
      <Handle type="source" position={Position.Right} />
    </div>
  )
}

function GraphInspector({ graph, node, edge, nodeById, onSelectNode }: {
  graph: SemanticGraph
  node: GraphNode | null
  edge: GraphEdge | null
  nodeById: Map<string, GraphNode>
  onSelectNode: (id: string) => void
}) {
  if (node) {
    const relationships = graph.edges.filter((item) => item.source === node.id || item.target === node.id)
    return (
      <aside className="graph-inspector">
        <header><span className={`entity-kind kind-${node.kind.toLowerCase()}`}>{node.kind}</span><strong>{node.name}</strong><small>{Math.round(node.confidence * 100)}% confidence</small></header>
        {node.description && <p className="inspector-description">{node.description}</p>}
        {node.aliases.length > 0 && <InspectorGroup title="Aliases"><div className="alias-row">{node.aliases.map((alias) => <span key={alias}>{alias}</span>)}</div></InspectorGroup>}
        {Object.keys(node.attributes).length > 0 && <InspectorGroup title="Attributes"><dl>{Object.entries(node.attributes).map(([key, value]) => <div key={key}><dt>{key.replaceAll('_', ' ')}</dt><dd>{value}</dd></div>)}</dl></InspectorGroup>}
        {node.external_attributes.map((external) => (
          <InspectorGroup title="External enrichment" key={`${external.provider}-${external.field_group}`}>
            <div className="provenance-heading"><strong>{external.provider}</strong><span>{Math.round(external.confidence * 100)}%</span>{external.source_url && <a href={external.source_url} target="_blank" rel="noreferrer" title="Open source"><ExternalLink size={13} /></a>}</div>
            <dl>{externalProfileEntries(external.payload).map(([key, value]) => <div key={key}><dt>{key}</dt><dd>{value}</dd></div>)}</dl>
          </InspectorGroup>
        ))}
        <InspectorGroup title={`Relationships · ${relationships.length}`}>
          <div className="inspector-relationships">
            {relationships.map((item) => {
              const outbound = item.source === node.id
              const peer = nodeById.get(outbound ? item.target : item.source)
              return <button type="button" key={item.id} onClick={() => peer && onSelectNode(peer.id)}><span>{outbound ? item.predicate.replaceAll('_', ' ') : `← ${item.predicate.replaceAll('_', ' ')}`}</span><strong>{peer?.name ?? 'Unknown'}</strong></button>
            })}
          </div>
        </InspectorGroup>
      </aside>
    )
  }

  if (edge) {
    const source = nodeById.get(edge.source)
    const target = nodeById.get(edge.target)
    return (
      <aside className="graph-inspector">
        <header><span className="entity-kind">Relationship</span><strong>{edge.predicate.replaceAll('_', ' ')}</strong><small>{Math.round(edge.confidence * 100)}% confidence</small></header>
        <div className="inspector-edge-statement"><button type="button" onClick={() => onSelectNode(edge.source)}>{source?.name ?? 'Unknown'}</button><span>{edge.predicate.replaceAll('_', ' ')}</span><button type="button" onClick={() => onSelectNode(edge.target)}>{target?.name ?? 'Unknown'}</button></div>
        {Object.keys(edge.attributes).length > 0 && <InspectorGroup title="Attributes"><dl>{Object.entries(edge.attributes).map(([key, value]) => <div key={key}><dt>{key.replaceAll('_', ' ')}</dt><dd>{value}</dd></div>)}</dl></InspectorGroup>}
        <InspectorGroup title={`Evidence · ${edge.evidence.length}`}>
          <div className="evidence-stack">{edge.evidence.map((item) => <article key={item.episode_id}><span>{item.source_type}</span><p>{item.excerpt}</p></article>)}</div>
        </InspectorGroup>
      </aside>
    )
  }

  const enriched = graph.nodes.filter((item) => item.external_attributes.length > 0).length
  const kinds = [...new Set(graph.nodes.map((item) => item.kind))]
  return (
    <aside className="graph-inspector overview-inspector">
      <header><span className="entity-kind">Overview</span><strong>Semantic memory</strong><small>Current consolidated state</small></header>
      <div className="overview-metrics"><div><strong>{graph.nodes.length}</strong><span>entities</span></div><div><strong>{graph.edges.length}</strong><span>relationships</span></div><div><strong>{enriched}</strong><span>enriched</span></div></div>
      <InspectorGroup title="Entity types"><div className="kind-distribution">{kinds.map((kind) => <div key={kind}><i style={{ background: kindColor(kind) }} /><span>{kind}</span><strong>{graph.nodes.filter((item) => item.kind === kind).length}</strong></div>)}</div></InspectorGroup>
    </aside>
  )
}

function InspectorGroup({ title, children }: { title: string; children: React.ReactNode }) {
  return <section className="inspector-group"><h3>{title}</h3>{children}</section>
}

function GraphDetails({ graph, nodeById, onSelect }: {
  graph: SemanticGraph
  nodeById: Map<string, GraphNode>
  onSelect: (selection: GraphSelection) => void
}) {
  return (
    <div className="graph-details-grid">
      <div className="entity-list">
        <div className="pane-heading"><span>Entities</span><strong>{graph.nodes.length}</strong></div>
        {graph.nodes.map((node) => (
          <button type="button" className="entity-item" key={node.id} onClick={() => onSelect({ type: 'node', id: node.id })}>
            <span className={`entity-kind kind-${node.kind.toLowerCase()}`}>{node.kind}</span>
            <div><strong>{node.name}</strong>{node.aliases.length > 0 && <small>aka {node.aliases.join(', ')}</small>}{node.external_attributes.length > 0 && <small className="external-source">enriched by {node.external_attributes.map((item) => item.provider).join(', ')}</small>}<p>{node.description}</p></div>
            <span className="confidence">{Math.round(node.confidence * 100)}%</span>
          </button>
        ))}
      </div>
      <div className="edge-list">
        <div className="pane-heading"><span>Relationships</span><strong>{graph.edges.length}</strong></div>
        {graph.edges.map((edge) => (
          <button type="button" className="edge-item" key={edge.id} onClick={() => onSelect({ type: 'edge', id: edge.id })}>
            <div className="edge-statement"><strong>{nodeById.get(edge.source)?.name ?? 'Unknown'}</strong><span>{edge.predicate.replaceAll('_', ' ')}</span><strong>{nodeById.get(edge.target)?.name ?? 'Unknown'}</strong></div>
            <div className="evidence-count">{edge.evidence.length} evidence {edge.evidence.length === 1 ? 'episode' : 'episodes'}</div>
            {edge.evidence[0] && <p>{edge.evidence[0].excerpt}</p>}
          </button>
        ))}
      </div>
    </div>
  )
}

function layoutSemanticGraph(nodes: GraphNode[], edges: GraphEdge[]): { nodes: SemanticFlowNode[]; edges: SemanticFlowEdge[] } {
  const layout = new dagre.graphlib.Graph().setDefaultEdgeLabel(() => ({}))
  layout.setGraph({ rankdir: 'LR', ranksep: 115, nodesep: 38, marginx: 30, marginy: 30 })
  const degree = new Map(nodes.map((node) => [node.id, edges.filter((edge) => edge.source === node.id || edge.target === node.id).length]))
  nodes.forEach((node) => layout.setNode(node.id, { width: node.kind === 'Decision' || node.kind === 'Blocker' ? 218 : 188, height: 82 }))
  edges.forEach((edge) => layout.setEdge(edge.source, edge.target))
  dagre.layout(layout)

  return {
    nodes: nodes.map((graphNode) => {
      const position = layout.node(graphNode.id)
      const width = graphNode.kind === 'Decision' || graphNode.kind === 'Blocker' ? 218 : 188
      return {
        id: graphNode.id,
        type: 'semantic',
        position: { x: position.x - width / 2, y: position.y - 41 },
        data: { graphNode, degree: degree.get(graphNode.id) ?? 0 },
      }
    }),
    edges: edges.map((graphEdge) => ({
      id: graphEdge.id,
      source: graphEdge.source,
      target: graphEdge.target,
      label: graphEdge.predicate.replaceAll('_', ' '),
      data: { graphEdge },
      markerEnd: { type: MarkerType.ArrowClosed, width: 14, height: 14 },
      className: `semantic-edge edge-${graphEdge.predicate}`,
      style: { strokeWidth: Math.max(1.2, graphEdge.confidence * 1.8) },
      labelStyle: { fontSize: 9, fontWeight: 700, fill: '#555b52' },
      labelBgStyle: { fill: '#f7f7f2', fillOpacity: 0.92 },
      labelBgPadding: [5, 3] as [number, number],
      labelBgBorderRadius: 3,
    })),
  }
}

function kindColor(kind: string) {
  return ({ Company: '#32746d', Product: '#4870a8', Project: '#59804e', Person: '#ad7042', Task: '#746aa3', Decision: '#875c91', Blocker: '#a84f47' } as Record<string, string>)[kind] ?? '#737970'
}

function externalProfileEntries(payload: Record<string, unknown>): Array<[string, string]> {
  const entries: Array<[string, string]> = []
  const basic = payload.basic_info
  if (basic && typeof basic === 'object') {
    const info = basic as Record<string, unknown>
    if (info.employee_count_range) entries.push(['employee range', String(info.employee_count_range)])
    if (Array.isArray(info.industries)) entries.push(['industries', info.industries.join(', ')])
    if (info.description) entries.push(['provider description', String(info.description)])
  }
  const headcount = payload.headcount
  if (headcount && typeof headcount === 'object' && (headcount as Record<string, unknown>).total != null) entries.push(['headcount', String((headcount as Record<string, unknown>).total)])
  const funding = payload.funding
  if (funding && typeof funding === 'object' && (funding as Record<string, unknown>).total_investment_usd != null) entries.push(['funding', `$${Number((funding as Record<string, unknown>).total_investment_usd).toLocaleString()}`])
  return entries.slice(0, 4)
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