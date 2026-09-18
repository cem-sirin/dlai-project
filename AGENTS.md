<!-- BEGIN VALINOR UNIVERSAL POLICY -->
<!-- valinor-policy-version: 2 -->

# Universal

- Let delegated agents plan their own work.
- Solo: no worktree. Delegated: use worktree.
- Merge verified work, then remove worktree.
- Bounded, non-critical delegated work: prefer `gpt-5.6-luna`.
- Browser: terminal `agent-browser` / `playwright-cli` only, fresh own session.
- Code in `~/Code/dlai-s2-2026`, data in `~/Data/dlai-s2-2026`, work in `~/Worktrees/dlai-s2-2026/<task>`.

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
