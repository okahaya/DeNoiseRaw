"""The GUI's backing logic, tested without a browser.

Gradio is an optional dependency (``pip install -e ".[gui]"``), so every test
here skips cleanly when it is not installed rather than failing the suite.
"""

import os

import pytest

gradio = pytest.importorskip("gradio")

from denoiseraw.gui import (  # noqa: E402
    BACKEND_LABELS,
    METHOD_LABELS,
    OUTPUT_FORMATS,
    build_interface,
    process_files,
)


class _FakeUpload:
    """Mimics what Gradio hands a callback for an uploaded file: a `.name` path."""

    def __init__(self, path):
        self.name = path


def test_build_interface_constructs_without_error():
    demo = build_interface()
    assert demo is not None
    # Blocks exposes its components; a sanity check that the page actually has
    # the pieces the user interacts with, not just that construction didn't throw.
    rendered = str(demo.config)
    for expected in ("RAWファイルを選択", "実行", "処理結果", "ダウンロード"):
        assert expected in rendered


def test_process_files_denoises_a_real_synthetic_capture(synthetic_dng):
    path, _, _ = synthetic_dng
    log, outputs, before, after = process_files(
        files=[_FakeUpload(path)],
        method_label=next(k for k, v in METHOD_LABELS.items() if v == "classical"),
        backend_label=next(k for k, v in BACKEND_LABELS.items() if v == "wavelet"),
        strength=1.0,
        demosaic="malvar",
        auto_bright=True,
        exposure=0.0,
        output_format_label=next(iter(OUTPUT_FORMATS)),
        self_ensemble=False,
        checkpoint_file=None,
        profile_file=None,
        iso_override=None,
        tile=512,
        progress=None,
    )
    assert "完了: 1/1" in log
    assert len(outputs) == 1
    assert os.path.exists(outputs[0])
    assert os.path.exists(before)
    assert os.path.exists(after)


def test_process_files_batches_multiple_inputs(synthetic_dng):
    path, _, _ = synthetic_dng
    log, outputs, _, _ = process_files(
        files=[_FakeUpload(path), _FakeUpload(path)],
        method_label=next(k for k, v in METHOD_LABELS.items() if v == "classical"),
        backend_label=next(k for k, v in BACKEND_LABELS.items() if v == "wavelet"),
        strength=1.0, demosaic="malvar", auto_bright=True, exposure=0.0,
        output_format_label=next(iter(OUTPUT_FORMATS)),
        self_ensemble=False, checkpoint_file=None, profile_file=None,
        iso_override=None, tile=512, progress=None,
    )
    assert "完了: 2/2" in log
    assert len(outputs) == 2
    assert len(set(outputs)) == 2  # distinct filenames, not overwriting each other


def test_process_files_with_no_input_is_a_clear_message():
    log, outputs, _before, _after = process_files(
        files=[], method_label=next(iter(METHOD_LABELS)), backend_label=next(iter(BACKEND_LABELS)),
        strength=1.0, demosaic="malvar", auto_bright=True, exposure=0.0,
        output_format_label=next(iter(OUTPUT_FORMATS)), self_ensemble=False,
        checkpoint_file=None, profile_file=None, iso_override=None, tile=512, progress=None,
    )
    assert "選択されていません" in log
    assert outputs == []


def test_process_files_model_method_without_checkpoint_explains_itself(synthetic_dng):
    path, _, _ = synthetic_dng
    log, outputs, _, _ = process_files(
        files=[_FakeUpload(path)],
        method_label=next(k for k, v in METHOD_LABELS.items() if v == "model"),
        backend_label=next(iter(BACKEND_LABELS)),
        strength=1.0, demosaic="malvar", auto_bright=True, exposure=0.0,
        output_format_label=next(iter(OUTPUT_FORMATS)), self_ensemble=False,
        checkpoint_file=None, profile_file=None, iso_override=None, tile=512, progress=None,
    )
    assert "チェックポイント" in log
    assert outputs == []


def test_process_files_survives_one_bad_file_in_a_batch(synthetic_dng, tmp_path):
    path, _, _ = synthetic_dng
    broken = tmp_path / "broken.dng"
    broken.write_bytes(b"not a raw file")
    log, outputs, _, _ = process_files(
        files=[_FakeUpload(str(broken)), _FakeUpload(path)],
        method_label=next(k for k, v in METHOD_LABELS.items() if v == "classical"),
        backend_label=next(k for k, v in BACKEND_LABELS.items() if v == "wavelet"),
        strength=1.0, demosaic="malvar", auto_bright=True, exposure=0.0,
        output_format_label=next(iter(OUTPUT_FORMATS)), self_ensemble=False,
        checkpoint_file=None, profile_file=None, iso_override=None, tile=512, progress=None,
    )
    assert "失敗" in log
    assert "完了: 1/2" in log
    assert len(outputs) == 1  # only the good file produced a download
