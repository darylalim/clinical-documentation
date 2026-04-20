"""Public API surface for the clinical_documentation package.

Asserts `__all__` stays in sync with its defining modules: every name advertised
is importable at the package level and is the same object as its module-level
definition. Catches re-export drift (renames, removals, accidental wrappers)."""

from __future__ import annotations

import clinical_documentation
from clinical_documentation import asr, device, llm


def test_package_reexports_match_module_level_definitions():
    expected = {
        "DEFAULT_MAX_TOKENS": llm.DEFAULT_MAX_TOKENS,
        "DEFAULT_MODEL_ID": llm.DEFAULT_MODEL_ID,
        "load_asr_pipeline": asr.load_asr_pipeline,
        "load_medgemma": llm.load_medgemma,
        "pick_device": device.pick_device,
        "stream_soap": llm.stream_soap,
        "transcribe": asr.transcribe,
    }
    for name, canonical in expected.items():
        assert getattr(clinical_documentation, name) is canonical, (
            f"clinical_documentation.{name} is not the same object as its module-level definition"
        )
    assert set(clinical_documentation.__all__) == set(expected)
