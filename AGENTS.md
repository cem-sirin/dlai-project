<!-- BEGIN VALINOR UNIVERSAL POLICY -->
<!-- valinor-policy-version: 7 -->

# Universal

- Let delegated agents plan their own work.
- Solo work stays in the current checkout; use separate worktrees for parallel agents.
- Single-project worktrees live under `~/Worktrees/<project>/<task>`; cross-project tasks may use `~/Space/<task>/<project-or-worktree>`.
- Merge verified work, then remove the task worktrees and their environments/build outputs; abandoned worktrees are removed too.
- Browser: terminal `agent-browser` / `playwright-cli` only, fresh own session. Ephemeral browser/debug artifacts such as screenshots, traces, and temporary downloads go in a per-run `/tmp/<tool>-<run-id>/` directory, never directly in `~`, a repo, `~/Worktrees`, or `~/Space`; delete the run directory on normal exit and promote only artifacts deliberately worth retaining.
- Project development servers are transient: use `hq serve` with a hard TTL instead of creating systemd user services. Use the singleton tailnet-only `hq share` for temporary file sharing. Do not add project systemd timers as an ad-hoc scheduler; scheduled jobs belong in the future HQ scheduler.
- Keep code under the repo's canonical `~/Code/...` path and project data under its declared `~/Data/...` root; repo-specific path declarations win.
- If a project uses Kokolekt, first understand the current Kokolekt design and follow its current contracts and conventions. Do not impose legacy cache, snapshot, run, or storage conventions on it.
- Valinor is working storage, not an archive. During development, prefer rebuild/redownload over preserving obsolete generated state or migrations. Prefer clean breaking changes over compatibility layers when the project does not require compatibility; fix callers forward rather than preserving dead development APIs.
- Web/TypeScript: use Bun for installs, lockfiles and scripts, and `bun check` for plain TypeScript. Keep Svelte's `svelte-check`, Astro/Nuxt-specific validation, Vite for SvelteKit, and Wrangler for Cloudflare Workers; select deployment adapters by runtime rather than replacing framework checks with Bun's incomplete Svelte coverage.
- Outside framework-managed storage, preserve human-authored or genuinely irreplaceable inputs and outputs. Build outputs, temporary files, caches, generated debug artifacts, and obsolete local state are disposable unless the project explicitly says otherwise.

<!-- END VALINOR UNIVERSAL POLICY -->

# Mackerel Road

## Layout

```text
src/mackerel_road/game/         # game types and route-map generation
src/mackerel_road/nn/           # neural mechanic, planning, objectives, training
src/mackerel_road/experiments/  # protocol, artifacts, summaries, plots, remote runner
tests/{game,nn,experiments}/    # tests mirror the source packages
report/                         # LaTeX source, figures, and compiled submission PDF
results/                        # selected model state and tracked result data
```

Use `mackerel_road.nn` for model/training changes and
`mackerel_road.experiments` for experiment/report plumbing. Keep game types and
map generation independent of PyTorch.

## Commands

| Task | Command |
|---|---|
| Sync CPU/dev environment | `uv sync --no-extra cuda` |
| Tests | `uv run pytest` |
| Ruff lint | `uv run ruff check .` |
| Ruff format | `uv run ruff format .` |
| Typecheck | `uv run ty check` |
| Check formatting | `uv run ruff format --check .` |

The package distribution is `mackerel-road`; its CLI entry points are
`mackerel-train`, `mackerel-extract`, `mackerel-summarize`, `mackerel-plot`, and
`mackerel-remote`.

## Compute boundary

- This Amsterdam VPS has no GPU. Do not run training, CUDA/PyTorch jobs,
  large-model inference, full-dataset processing, or long experiments here.
- Run approved neural optimization through the remote GPU machine and the canonical
  wrapper, for example:

  ```bash
  uv run mackerel-remote \
    "-m mackerel_road.nn.training --device cuda"
  ```

- `REMOTE_HOST` belongs only in the untracked local `.env`; never expose or
  commit credentials. The direct SSH/rsync route is separate from the yunidb
  SOCKS proxy.

## Generated state and reports

Keep `artifacts/`, `checkpoints/`, `runs/`, `outputs/`, `tmp/`, and credentials
out of Git. The current tracked report is `report/main.pdf`; edit its source in
`report/main.tex` and follow `report/README.md` for builds. Do not run report
training or regenerate large datasets on this VPS.

The report builder writes figures to `report/figures/` and the result summary
to `results/README.md`. Do not recreate a separate `docs/` tree.
