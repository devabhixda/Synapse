# Synapse Memory

- Use the `synapse-memory` MCP server's `recall` tool whenever a task depends on company
  decisions, systems, owners, blockers, project context, or prior organizational work.
- Treat `recall` as the mandatory retrieval path. Do not bypass it with direct database or
  graph queries because semantic fallback must create a retrieval episode for future use.
- Use `get_episode` only to inspect an episode cited by `recall`.
- Call `list_skill_specs` before a multi-step company research, architecture, or project
  briefing task. If an approved workflow applies, fetch it with `get_skill_spec` and follow
  its recall queries, steps, output contract, and guardrails.
- Never assume a candidate or rejected procedural skill is available; MCP exposes approved
  versions only.
- Synapse MCP tools are read-only. Repository inspection, edits, commands, and validation
  remain the responsibility of the VS Code agent and require the normal user approval flow.
- The server uses the stable MCP Python SDK v1 (`mcp>=1.27,<2`). SDK documentation:
  https://github.com/modelcontextprotocol/python-sdk/tree/v1.x