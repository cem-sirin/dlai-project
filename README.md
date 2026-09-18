# Mackerel Road

A neural game mechanic trained through differentiable game-level objectives,
without transition labels or a teacher engine. Two porters race to deliver fish
across generated route maps under moving rain clouds and stochastic delays.

Course project for [Deep Learning & Applied AI 2026](https://github.com/erodola/DLAI-s2-2026).

**[Read the report](report/main.pdf)** · **[Results](results/README.md)**

## Contents

| Path | Contents |
|---|---|
| `src/mackerel_road/game/` | Game types and seeded route-map generation |
| `src/mackerel_road/nn/` | Neural mechanic, differentiable planning, losses, training, evaluation |
| `src/mackerel_road/experiments/` | Artifact extraction, summaries, figures, report builder, remote runner |
| `tests/` | Regression and integration tests |
| `report/` | Report, appendix, LaTeX sources, official template, figures |
| `results/` | Measured protocol and results, selected model, generated summary |

## Rebuild the report

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), and a LaTeX installation
with `latexmk`. No GPU or PyTorch is needed to regenerate the reporting artifacts.

```bash
uv sync --no-extra cuda
uv run -m mackerel_road.experiments.submission results/neural-evaluated.json
(cd report && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex)
```

The builder regenerates `report/figures/`, numerical LaTeX tables, and
`results/README.md` from stored measurements. It preserves the selected model
when its input is the compact, weight-free result file.

## Reproduce the experiment

The matched ablation uses four objective variants, three optimizer seeds,
64 training maps, 16 validation maps, and 100 disjoint test maps shared across
seeds and variants. The appendix specifies the losses, checkpoint selection,
corruption intervention, and limits of the evaluation.

Run the following **on a CUDA-capable machine**, not the CPU-only development VPS.
Use a tmux session for the bounded training batch.

```bash
uv sync --extra cuda
mkdir -p artifacts
uv run mackerel-train --device cuda \
  --variants viability,+balance,+specialization,+skill_weather \
  --seeds 3 --steps 120 --batch-maps 4 --validation-maps 16 \
  --test-maps 100 --test-seed-start 30000 > artifacts/training.log 2>&1
uv run mackerel-extract artifacts/training.log artifacts/training.json
uv run -m mackerel_road.experiments.submission artifacts/training.json
```

For an already configured remote GPU, `mackerel-remote` accepts the equivalent
quoted Python arguments beginning with `-m mackerel_road.nn.training`. Its
`REMOTE_HOST` setting belongs only in the local, ignored `.env`; credentials and
raw training logs are never published. The wrapper stages source safely and
uses a persistent, frozen remote CUDA environment.

## Checks

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

Without the CUDA extra, neural tests skip; full neural tests and full-project
type checking require PyTorch. Run those on the GPU machine. The reported
outcomes are computational playtests, not evidence of human enjoyment.
