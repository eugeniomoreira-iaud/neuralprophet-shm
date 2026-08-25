# AGENTS.md — `neuralprophet-shm` code repository

Codex entry point for this repository. Every shared rule for the code subproject lives in
`instructions-pipeline.md`, and the project-wide operating protocol lives in
`../instructions-core.md`. Both are binding; this file adds only what is specific to Codex.

> **Read `instructions-pipeline.md` and `../instructions-core.md` now, before doing anything else
> in this repository. Their contents are binding.**

## Codex specifics

**Model policy.** The main Codex thread uses the most capable model selected for the session and
handles planning, decisions, and review. When Codex multi-agent tools are available, delegated
heavy or mechanical work uses a less advanced model. If subagents are unavailable, do the work
inline rather than skipping it.

**Settings.** Codex does not rely on the permissions declared by the parent folder's Claude or
generic-agent settings. Treat the raw `.adc` archive in `_UNIPG/__Mura-realtime/` as strictly
read-only regardless of the permissions exposed by the current environment: never write to it,
move it, or copy it into this repository.
