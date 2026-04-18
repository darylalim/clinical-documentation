# clinical-ai

Clinical AI platform.

## MedASR transcription

Transcribe physician dictation with [Google MedASR](https://huggingface.co/google/medasr) — a 105M-parameter Conformer-CTC model specialized for radiology and general medical speech.

### One-time setup

1. **Accept the license.** Visit https://huggingface.co/google/medasr, sign in, and click *Agree and access repository*. MedASR is gated under the Health AI Developer Foundations terms.
2. **Install dependencies** (creates `.venv/` automatically):
   ```bash
   uv sync
   ```
   The first install pulls a pinned build of `transformers` from source (MedASR needs v5.0+) and may take a few minutes.
3. **Set your Hugging Face token.** Create a read-access token at https://huggingface.co/settings/tokens, then copy the template and paste it in:
   ```bash
   cp .env.example .env
   # edit .env and set HF_TOKEN=hf_...
   ```
   The script loads `.env` automatically via `python-dotenv`. `.env` is gitignored — do not commit it.

### Run it

Transcribe the sample audio bundled with the model repo:
```bash
uv run transcribe.py --sample
```

Transcribe your own file:
```bash
uv run transcribe.py path/to/audio.wav
```

All options:
```bash
uv run transcribe.py --help
```

| Flag | Default | Notes |
|---|---|---|
| `--model` | `google/medasr` | Any HF ASR checkpoint. |
| `--device` | `auto` | `cpu`, `cuda`, or `mps`. Auto picks the best available. |
| `--chunk-s` | `20.0` | Seconds per inference chunk (long audio is split). |
| `--stride-s` | `2.0` | Overlap between chunks to avoid word splits. |

### Notes

- Input audio is resampled to 16 kHz mono automatically.
- First model run downloads ~400 MB of weights to `~/.cache/huggingface`.
- On Apple Silicon, `--device mps` works but is slower than CUDA; `cpu` is a reliable fallback.
- MedASR is a **foundation** model — outputs are preliminary and must be verified before any clinical use. See the [model card limitations](https://huggingface.co/google/medasr#limitations) for speaker, language, and vocabulary caveats.

## Development

[Ruff](https://docs.astral.sh/ruff/) handles both linting and formatting. It's in the `dev` dependency group and installed by `uv sync`.

```bash
uv run ruff format .        # apply formatting
uv run ruff check .         # report lint issues
uv run ruff check --fix .   # apply auto-fixable lints
```

Config lives under `[tool.ruff]` in `pyproject.toml` — 100-char line length, `py310` target, rule set `E W F I B UP SIM C4 RUF`.

### Type checking

[ty](https://docs.astral.sh/ty/) is Astral's Python type checker and language server. It's in the `dev` dependency group and installed by `uv sync`.

```bash
uv run ty check             # type-check the project
uv run ty server            # run the language server (LSP over stdio)
```

Editor integration: most editors can launch the LSP via `uv run ty server`. For VS Code, install the [ty extension](https://marketplace.visualstudio.com/items?itemName=astral-sh.ty); for Zed/Neovim/Helix, point the LSP client at that command. Config lives under `[tool.ty]` in `pyproject.toml` (currently just `python-version = "3.10"`).

### Testing

[pytest](https://docs.pytest.org/) is the test runner, with [pytest-cov](https://pytest-cov.readthedocs.io/) for coverage and [pytest-mock](https://pytest-mock.readthedocs.io/) for the `mocker` fixture. All three live in the `dev` dependency group.

```bash
uv run pytest                                       # run unit tests
uv run pytest --cov=transcribe --cov-report=term-missing   # with coverage
uv run pytest -m integration                        # opt into real-model tests
```

Tests live under `tests/`. Anything that hits the real MedASR model or the network must be marked `@pytest.mark.integration` — the default invocation deselects that marker, so unit tests stay fast and offline. Config lives under `[tool.pytest.ini_options]` and `[tool.coverage.run]` in `pyproject.toml`.
