---
tags: [project, session]
status: developing
created: 2026-10-05
modified: 2026-10-05
source: project-conversation
---
# 2026-10-05 — Vault memory and hooks setup

## Request

Set up the Obsidian Memory Starter in this repo, plus hooks that add to memory after each prompt when needed.

## Decisions

- **Agreed:** the vault lives in `Knowledge/`, ignored by the parent repo and versioned as its own local Git repo. Live results and next steps moved from `CLAUDE.md` into `PROGRESS.md` and `PROJECT_PLAN.md` so each fact has a single owner.
- **Agreed:** hooks are in `.claude/settings.local.json` (personal, git-ignored). The Stop hook is a gate, not a capture: it asks Claude to checkpoint only when a turn edited project files, ran python, used decision-like wording, or ran 12+ tool calls, and memory was not touched. PreCompact and SessionEnd write drafts to `Inbox/Auto/`.

## Work and evidence

- Files: `Knowledge/`, `PROGRESS.md`, `PROJECT_PLAN.md`, `CLAUDE.md` (memory section), `scripts/vault-hook.cjs`, `scripts/vault-lint.cjs`, `.claude/settings.local.json`, `.gitignore`.
- Checks: hook pipe-tests and lint (results in [[Log]]).
- Limits: a live hook firing is only confirmed when a new session is opened in the repo folder.

## Next action

Open a new Claude Code session in `IE4483-sentiment-analysis/` and confirm that SessionStart injects the handoff. Then start Stage 1 of the [roadmap](../../../PROJECT_PLAN.md).
