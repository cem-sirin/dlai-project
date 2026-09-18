---
name: agents-md
description: Create, audit, and maintain concise AGENTS.md-style project instructions. Use when agent instructions are verbose, obvious, repetitive, stale, or hard to scan.
---

# AGENTS.md

Treat `AGENTS.md` as a practical execution guide for a capable coding agent.

Keep it concise and project-specific, but not cryptically minimal. Include information that prevents mistakes, avoids repeated confusion, or saves meaningful rediscovery time.

## What belongs

Before adding or keeping a rule, ask:

1. Would removing this make an agent more likely to make a mistake or waste time?
2. Is it specific to this project or workflow?
3. Is it durable enough to remain useful across tasks?
4. Is it not already obvious from nearby code or config?

If none apply, omit it.

Good candidates include:

- non-obvious commands or package-manager choices
- unusual directory or data locations
- generated files that should not be edited directly
- project-specific test, build, deploy, or validation workflows
- important conventions that are easy to violate
- traps, exceptions, or environment quirks that repeatedly matter

Do not document generic competence such as "inspect the repo first", "check logs on failure", "run tests", or "be careful with errors" unless the project has a specific exception worth stating.

## Prefer dense formats

Choose the shortest format that stays clear.

Prefer mappings and annotated paths:

```text
~/Data/<project>/     # Raw data
src/generated/        # Generated; do not edit
main                  # Ship directly; no PR workflow
```

Prefer command tables when several commands matter:

```markdown
| Task | Command |
|---|---|
| Test file | `uv run pytest path/to/test.py` |
| Lint | `ruff check .` |
| Typecheck | `ty check` |
```

Prefer a short code block, table, or example over a long prose list when the structure carries the meaning better.

Use bullets for actual rules. Keep one idea per bullet and keep rationale brief unless it prevents a likely mistake.

## Writing style

- Use short headings only when they improve scanning.
- Prefer exact paths and commands over descriptions of them.
- Prefer examples over explanations when an example is sufficient.
- Avoid welcome text, conclusions, pleasantries, and generic quality slogans.
- Avoid repeating formatter, linter, typechecker, or manifest information that is already self-evident.
- Avoid file-by-file repository tours unless the layout is genuinely non-obvious.
- Avoid verbose numbered procedures when a command block or table is clearer.
- Do not preserve a section merely because it already exists.

A concise sentence is fine when prose is the clearest representation. Do not contort everything into bullets or tables.

## Editing an existing file

Audit deletion-first, but use judgment.

- Remove stale or duplicated rules.
- Collapse repeated prose into a compact example, mapping, or table.
- Merge rules that express the same constraint.
- Replace vague guidance with exact commands or paths where useful.
- Keep useful context even if it is not strictly necessary, as long as it earns its space.

Files under roughly 60 lines are usually easy to scan; longer files are fine when the project genuinely needs the extra instructions. Treat length as a warning signal, not a hard limit.

## Final check

Read each rule and ask:

> If this disappeared, what mistake, confusion, or repeated work becomes more likely?

If there is no good answer, consider deleting it.
