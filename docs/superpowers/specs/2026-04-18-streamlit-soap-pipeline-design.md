# Streamlit SOAP Pipeline — Design

**Date:** 2026-04-18
**Status:** Approved (pending user review of this document)
**Scope:** Replace the CLI-only entry point with a Streamlit UI that chains MedASR (transcription) and MedGemma (SOAP note generation), running fully locally on a 32 GB Apple Silicon Mac.
**Note:** The `clinical_ai` package was later renamed to `clinical_documentation` on 2026-04-19. References below reflect the original naming.

---

## 1. Goal

Build a single-user, local clinical-documentation prototype that:

1. Accepts a recorded patient visit as audio.
2. Transcribes it with **Google MedASR** (existing pipeline, unchanged).
3. Lets the user review and edit the transcript before LLM generation.
4. Generates a SOAP note (Subjective / Objective / Assessment / Plan) from the edited transcript using **MedGemma 27B (text-only, 4-bit MLX)**.
5. Lets the user edit the SOAP note and download it as Markdown.

Out of scope for v1: persistence, multi-session history, summary-only output mode, multi-user auth, image input, EHR integration, PHI logging, deployment beyond `streamlit run`.

## 2. Locked-in technology choices

| Choice | Value | Reason |
|---|---|---|
| UI framework | **Streamlit** | Better fit than Gradio for review-and-edit workflows; lighter than FastAPI+React for v1. |
| ASR model | **`google/medasr`** via `transformers` pipeline on **MPS** | Already working in `transcribe.py`; no change. |
| LLM model | **`mlx-community/medgemma-27b-text-it-4bit`** via **`mlx-lm`** | Best clinical reasoning available locally on 32 GB; text-only matches the use case; `mlx-lm` is the right abstraction for text-in/text-out. |
| Memory strategy | Both models loaded once at startup, kept resident | 32 GB unified memory comfortably fits ~14 GB MedGemma + ~400 MB MedASR + KV cache + Streamlit. MLX uses unified memory efficiently. |
| Workflow | Two-step with editable transcript and editable SOAP | Prevents ASR errors from compounding into hallucinated notes — the one quality risk that matters. |
| Output | SOAP only (S/O/A/P sections) | Matches stated use case; summary mode deferred (YAGNI). |
| CLI fate | Refactor shared logic into a `clinical_ai/` package; CLI becomes a thin shim | Crossing the threshold from one script to two models + two backends; tests already import internals. |
| Storage | None — all state in memory; only artifact is user-initiated `.md` download | Keeps PHI off disk in v1. |

## 3. Architecture

Single Streamlit process, two model backends in one Python interpreter, synchronous per-action execution, no background workers, no DB.

```
┌─────────────────────────────────────────────────────────────┐
│  Streamlit process (one Python interpreter, MPS + MLX)      │
│                                                             │
│   app.py  (UI: uploader → transcript → SOAP)                │
│      │                                                      │
│      ├──► clinical_ai.asr   (HF transformers pipeline,      │
│      │       cached via @st.cache_resource, runs on MPS)    │
│      │       └─► google/medasr             ~400 MB on MPS   │
│      │                                                      │
│      ├──► clinical_ai.llm   (mlx_lm, cached via             │
│      │       @st.cache_resource, runs on MLX/unified mem)   │
│      │       └─► medgemma-27b-text-it-4bit  ~14 GB on MLX   │
│      │                                                      │
│      ├──► clinical_ai.prompts  (SOAP system + user template)│
│      └──► clinical_ai.device   (pick_device for MedASR)     │
│                                                             │
│   transcribe.py  (thin CLI shim — same asr module)          │
└─────────────────────────────────────────────────────────────┘
```

**Key boundaries**
- `asr.py` and `llm.py` know nothing about Streamlit — they're independently importable and the CLI proves it.
- `prompts.py` is pure data + pure functions — easy to iterate on prompt wording without touching the LLM wrapper.
- `app.py` is the only file that imports `streamlit`.

**Import ordering invariant** (load-bearing, preserved from current `transcribe.py`): both `app.py` and `transcribe.py` must call `load_dotenv()` *before* importing anything that touches `transformers`, `torch`, or `mlx_lm`, because HF reads `HF_TOKEN` / `HF_HOME` at import time. The `clinical_ai/` package modules themselves don't call `load_dotenv()` — that's the entry point's job.

**Estimated memory at steady state:** ~16–18 GB on a 32 GB Mac (model weights + KV cache during generation + Streamlit + system).

## 4. Repository layout

```
clinical_ai/
  __init__.py
  device.py      # pick_device()
  asr.py         # load_asr_pipeline() + transcribe()
  llm.py         # load_medgemma() + stream_soap()
  prompts.py     # SOAP system prompt + format_soap_messages()
app.py           # Streamlit entry point (single page, two-step flow)
transcribe.py    # thin CLI shim — argparse + calls into clinical_ai.asr
tests/
  test_device.py
  test_asr.py
  test_llm.py
  test_prompts.py
  test_integration.py     # placeholder, marked @pytest.mark.integration
docs/superpowers/specs/
  2026-04-18-streamlit-soap-pipeline-design.md
```

## 5. Component contracts

### `clinical_ai/device.py`
```python
def pick_device() -> Literal["cpu", "cuda", "mps"]: ...
```
Moved verbatim from `transcribe.py`. Used only by ASR (MLX picks its own device).

### `clinical_ai/asr.py`
```python
def load_asr_pipeline(model_id: str, device: str) -> Pipeline: ...
def transcribe(
    pipe: Pipeline,
    audio: Path | str | bytes,
    chunk_s: float = 20.0,
    stride_s: float = 2.0,
) -> str: ...
```
Loader and per-call function are split so the loader can be cached by Streamlit independently. Defensive `isinstance(result, dict)` check stays.

### `clinical_ai/llm.py`
```python
def load_medgemma(
    model_id: str = "mlx-community/medgemma-27b-text-it-4bit",
) -> tuple[Model, Tokenizer]: ...

def stream_soap(
    model, tokenizer, transcript: str,
    max_tokens: int = 1024, temperature: float = 0.2,
) -> Iterator[str]: ...
```
`stream_soap` is a generator that yields token chunks as `mlx_lm.stream_generate` produces them.

### `clinical_ai/prompts.py`
```python
SOAP_SYSTEM_PROMPT: str  # "You are a medical scribe..."
def format_soap_messages(transcript: str) -> list[dict]:
    # [{"role": "system", ...}, {"role": "user", ...}]
```
Pure functions over strings. Where prompt iteration happens.

### `app.py`
```python
@st.cache_resource
def _asr(): return load_asr_pipeline("google/medasr", pick_device())

@st.cache_resource
def _llm(): return load_medgemma()
```
~120 lines. Holds all `st.*` calls, no business logic.

### `transcribe.py` (CLI shim, kept)
~40 lines: argparse + `load_dotenv()` + calls into `clinical_ai.asr`. Preserves `--sample` and `--device auto` behavior. **No** SOAP generation in the CLI for v1.

## 6. Data flow & UI states

### Lifecycle of one user session

```
[App boot]
    │
    ├─ load_dotenv()                      (reads HF_TOKEN before HF imports run)
    ├─ _asr()    ← @st.cache_resource     (~10–20s: download cache + MPS warmup)
    └─ _llm()    ← @st.cache_resource     (~30–60s: ~14 GB into unified memory)
    │
[Page render]
    │
    ▼
STATE A — empty
  • st.file_uploader(wav/mp3/flac/m4a)
  • Rest of page hidden
    │
    │ user uploads file → bytes in memory
    ▼
STATE B — transcribing
  • st.spinner("Transcribing audio…")
  • Calls asr.transcribe(pipe, bytes)
  • Result stored in st.session_state.tx
    │
    ▼
STATE C — transcript ready, SOAP idle
  • st.text_area("Transcript", value=tx, key="tx_edit", height=300)
  • st.button("Generate SOAP note")
  • Hint: "Review the transcript before generating — fix any misheard terms."
    │
    │ button click → reads st.session_state.tx_edit
    ▼
STATE D — generating SOAP (streaming)
  • st.empty() placeholder
  • for chunk in stream_soap(...):
       buf += chunk
       placeholder.markdown(buf)
  • Final buf stored in st.session_state.soap
    │
    ▼
STATE E — SOAP ready
  • st.text_area("SOAP note", value=soap, key="soap_edit", height=500)
  • st.download_button("Download .md", data=soap_edit, file_name="soap_note.md")
  • st.button("Start over") → clears state
```

### Session state keys (the entire data model)
```python
st.session_state = {
    "audio_bytes":   bytes | None,
    "audio_name":    str | None,
    "tx":            str | None,
    "tx_edit":       str,     # bound to transcript text_area
    "soap":          str | None,
    "soap_edit":     str,     # bound to SOAP text_area
}
```

### State transitions
- **Upload event** → A→B→C in one rerun. New audio invalidates `tx`/`soap` (if `audio_name` changes, clear `tx`, `tx_edit`, `soap`, `soap_edit`).
- **"Generate SOAP" click** → C→D→E in one rerun. Sends `tx_edit` (the user's edits, not the raw ASR output).
- **"Start over"** → E→A. Clears everything except cached models.
- **Re-clicking "Generate SOAP"** while in E regenerates from current `tx_edit`.

## 7. Error handling

Strategy: **fail visibly, fail locally, never silently corrupt session state.**

| Where | What can go wrong | How we handle it |
|---|---|---|
| App boot — `load_dotenv()` | `HF_TOKEN` unset | After `load_dotenv()`, if `os.environ.get("HF_TOKEN")` is empty, `st.error` with copy-paste fix and `st.stop()`. |
| `_asr()` first call | HF download fails, gated-model 403, MPS unavailable | Let exception propagate; catch in `app.py`, render `st.error` with class + message + traceback in `st.expander`. `st.stop()`. Cached resource not populated → next reload retries. |
| `_llm()` first call | MLX download fails, disk full mid-download (~14 GB), `mlx_lm` import fails | Same pattern. Separate catch so user can tell which model failed. |
| Audio upload | Unsupported format, 0-byte, corrupt | `file_uploader(type=[...])` filters extensions; catch `librosa.LibsndfileError` and generic `Exception` in State B; `st.error` and revert to State A. |
| Transcribe call | OOM on MPS, unexpected pipeline shape | OOM: catch `RuntimeError` containing "out of memory", surface friendly message. Shape mismatch: existing `isinstance(result, dict)` check stays. |
| `stream_soap` generator | MLX OOM, tokenizer/template error, hang | OOM: same pattern. Template error: caught at boundary, partial output discarded. No timeout in v1. |
| Download click | (none) | n/a |

### What we explicitly do NOT do
- No retry loops on model load.
- No fallback models.
- No global try/except wrapping the whole page.
- No logging to disk (keeps PHI off filesystem).

### State invariants (enforced by error paths)
1. If `tx` is `None`, `tx_edit`/`soap`/`soap_edit` are also `None`/empty.
2. If `soap` is `None`, `soap_edit` is empty.
3. Cached models are never populated with a partially-loaded object (`@st.cache_resource` gives this for free).

A small helper `clear_downstream_state(after: str)` in `app.py` enforces invariants 1–2.

## 8. Testing

### Unit tests (default `uv run pytest`, fast and offline)

| Module | What we verify |
|---|---|
| `clinical_ai.device` | Existing 3 mock-based `pick_device` tests, moved. |
| `clinical_ai.asr` | `load_asr_pipeline` calls `transformers.pipeline` with right `task`/`model`/`device`. `transcribe` forwards chunk/stride and unwraps `result["text"]` (including `isinstance(result, dict)` fallback). |
| `clinical_ai.prompts` | `format_soap_messages` returns correct shape, embeds transcript verbatim, system prompt names "SOAP" and the four section labels. Pure functions, no mocks. |
| `clinical_ai.llm` | `load_medgemma` calls `mlx_lm.load` with right repo id, returns `(model, tokenizer)`. `stream_soap` calls `mlx_lm.stream_generate` with messages from `format_soap_messages`, threads `max_tokens`/`temperature`, yields chunks in order. |
| `app.py` | Optional: one `streamlit.testing.v1.AppTest` smoke that loads the app with `_asr`/`_llm` patched to stubs, uploads fake audio bytes, asserts STATE A→C transition. Skip if it proves flaky. |

### Mocking discipline
- Use `mocker` (pytest-mock) for consistency.
- Patch at the **point of lookup** (`clinical_ai.asr.pipeline`, `clinical_ai.llm.load`, `clinical_ai.llm.stream_generate`), not source modules.
- For `stream_soap`, patched `stream_generate` returns a generator yielding `["S", "OAP", " note"]` so we assert ordering and reconstruction without invoking MLX.

### Integration tests
`tests/test_integration.py` with one placeholder marked `@pytest.mark.integration` that downloads MedASR sample audio and runs `asr.transcribe` end-to-end. **No** integration test for MedGemma in v1 — pulling 14 GB on every CI run is excessive; manual smoke via the Streamlit app is sufficient.

### Coverage scope
Update `pyproject.toml`:
```toml
[tool.coverage.run]
source = ["clinical_ai"]
```
(Was `["transcribe"]`.) The thin CLI shim is excluded — mostly argparse + glue.

### What we explicitly don't test
- Streamlit page full rendering (smoke test or manual use).
- Real model output quality (prompt engineering, validated by clinician review).
- Network/HF download paths (covered by integration when run).
- MLX behavior itself (dependency's responsibility).

## 9. Dependencies

Add to `pyproject.toml` `dependencies`:
- `streamlit>=1.32`
- `mlx-lm>=0.19` (will pull in `mlx`)

No new dev dependencies — `pytest-mock` already covers everything.

## 10. Open questions / future work (NOT in v1)

- Summary-only output mode (would add a radio toggle and a second prompt template).
- Session history / persistence (would add `st.session_state["sessions"]` and a sidebar).
- Image input via the multimodal MedGemma (would swap to `mlx-vlm`).
- Authentication / multi-user.
- Deployment beyond `streamlit run` (HF Spaces, internal hosting, etc.).
- Audit logging for clinical compliance.
- Integration test for MedGemma (would require CI with ~20 GB free disk).
