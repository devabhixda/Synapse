from synapse.procedural.models import SkillTemplate

TEMPLATES: dict[str, SkillTemplate] = {
    "company_research": SkillTemplate(
        name="research_company_context",
        purpose=(
            "Build a cited company briefing from personal episodes, consolidated internal "
            "knowledge, and approved external enrichment."
        ),
        triggers=[
            "The user asks about a company's industry, size, funding, or profile.",
            "The user wants internal context about a vendor, partner, or account.",
        ],
        required_queries=[
            "Recall the company's identity, industry, size, and available enrichment.",
            "Recall internal decisions, projects, or relationships involving the company.",
        ],
        steps=[
            "Call Synapse recall for company firmographics and external profile context.",
            "Call Synapse recall for internal episodes and semantic relationships.",
            "Separate externally enriched attributes from internally observed facts.",
            "Return a compact briefing with citations for every substantive claim.",
        ],
        guardrails=[
            "Do not present external enrichment as an internally observed company fact.",
            "Do not invent missing headcount, funding, or industry information.",
            "Preserve episode and external-source citations in the final response.",
        ],
        expected_output={
            "format": "company_briefing",
            "sections": ["identity", "firmographics", "internal_context", "citations"],
        },
    ),
    "architecture_context": SkillTemplate(
        name="implement_architecture_decision",
        purpose=(
            "Retrieve an evidence-backed architecture decision and its affected systems "
            "before proposing repository changes."
        ),
        triggers=[
            "The user asks what architecture decision applies to a system or repository.",
            "A coding task depends on an internal platform, migration, or database decision.",
        ],
        required_queries=[
            "Recall the architecture decision, affected systems, owner, and blockers.",
            "Inspect the cited episodes when constraints or chronology are ambiguous.",
        ],
        steps=[
            "Call Synapse recall for the relevant architecture decision.",
            "Inspect cited episodes for ownership, constraints, and chronology.",
            "Inspect the target repository using the coding agent's normal tools.",
            "Propose the smallest change consistent with the recalled decision.",
            "Run repository tests and report citations alongside the implementation summary.",
        ],
        guardrails=[
            "Synapse supplies context only; repository edits remain with the coding agent.",
            "Do not treat a superseded or unsupported statement as the current decision.",
            "Do not execute code or shell commands through Synapse MCP tools.",
        ],
        expected_output={
            "format": "implementation_plan",
            "sections": ["decision", "constraints", "changes", "validation", "citations"],
        },
    ),
    "project_status_briefing": SkillTemplate(
        name="prepare_project_status_briefing",
        purpose="Produce a cited project update covering decisions, owners, blockers, and actions.",
        triggers=["The user asks for project status, current blockers, or next actions."],
        required_queries=[
            "Recall project decisions, recent activity, blockers, owners, and next actions."
        ],
        steps=[
            "Call Synapse recall for the project status and relevant time window.",
            "Separate confirmed decisions from unresolved or superseded statements.",
            "Group the response into progress, blockers, owners, and next actions.",
            "Attach evidence citations to each section.",
        ],
        guardrails=[
            "Do not infer completion from an assigned action.",
            "Call out missing owners or deadlines rather than inventing them.",
        ],
        expected_output={
            "format": "status_briefing",
            "sections": ["progress", "blockers", "owners", "next_actions", "citations"],
        },
    ),
}