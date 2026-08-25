# CLAUDE.md — `neuralprophet-shm` code repository

Claude Code entry point for this repository. Every shared rule for the code subproject lives in
`instructions-pipeline.md`, and the project-wide operating protocol lives in
`../instructions-core.md`. Both are binding; this file adds only what is specific to this
harness.

@instructions-pipeline.md
@../instructions-core.md

> **If your harness does not expand `@`-imports, read `instructions-pipeline.md` and
> `../instructions-core.md` now, before doing anything else in this repository. Their contents
> are binding.**

## Claude Code specifics

**Model policy.** Orchestrator (main thread) = big model: Fable 5 (`claude-fable-5`) or Opus 5
(`claude-opus-5`). Delegated work = Sonnet 5 (`claude-sonnet-5`) subagents, spawned with the
Agent tool.

**Settings.** Read-only access to the raw `.adc` archive in `_UNIPG/__Mura-realtime/` is granted
by `additionalDirectories` in the parent folder's `.claude/settings.local.json`; the same file
denies `Write`, `Edit` and `NotebookEdit` on that path.
