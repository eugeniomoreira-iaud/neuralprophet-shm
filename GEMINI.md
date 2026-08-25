# GEMINI.md — `neuralprophet-shm` code repository

Antigravity / Gemini entry point for this repository. Every shared rule for the code subproject
lives in `instructions-pipeline.md`, and the project-wide operating protocol lives in
`../instructions-core.md`. Both are binding; this file adds only what is specific to this
harness.

@instructions-pipeline.md
@../instructions-core.md

> **If your harness does not expand `@`-imports, read `instructions-pipeline.md` and
> `../instructions-core.md` now, before doing anything else in this repository. Their contents
> are binding.**

## Antigravity / Gemini specifics

**Model policy.** Orchestrator (main thread) = big model: Gemini Pro (e.g. Gemini 3.1 Pro).
Delegated work = Pro (`pro`) subagents, spawned with the `invoke_subagent` tool.

**Settings.** Read-only access to the raw `.adc` archive in `_UNIPG/__Mura-realtime/` is declared
by `readOnly` in the parent folder's `.agents/settings.local.json`.
