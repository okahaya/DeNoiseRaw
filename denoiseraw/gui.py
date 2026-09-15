"""A point-and-click front end for the denoising pipeline.

Everything the CLI can do, reachable without touching a terminal: pick one or
more RAW files, adjust the settings, press one button, and download the
results when it's done. Built on Gradio because that gives a native
multi-file picker and drag-and-drop for free, and it works the same whether
you run it on your own machine or hand someone a share link.

Launch it with::

    denoiseraw gui

or ``python -m denoiseraw.gui``. It opens ``http://127.0.0.1:7860`` in the
default browser.
"""

from __future__ import annotations

import os
import tempfile
import traceback
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .classical.denoise import BACKENDS
from .noise.profile import NoiseProfile, ProfileBank
from .pipeline import DenoiseSettings, denoise_raw, write_outputs
from .rawio.develop import DEMOSAIC_METHODS, develop
from .rawio.loader import load_raw
from .rawio.writer import save_image

OUTPUT_FORMATS = {
    "DNG(線形、推奨・現像ソフトで仕上げ用)": ".dng",
    "TIFF(16bitプレビュー)": ".tif",
    "PNG(8bitプレビュー)": ".png",
    "JPEG(共有用プレビュー)": ".jpg",
}

METHOD_LABELS = {
    "自動(学習済みモデルがあれば使用、無ければ古典手法)": "auto",
    "古典手法(学習不要、今すぐ使える)": "classical",
    "学習済みモデル(要チェックポイント)": "model",
}

BACKEND_LABELS = {
    "自動選択": "auto",
    "BM3D(高品質、低速)": "bm3d",
    "BM3D軽量版(依存なし)": "bm3d-lite",
    "wavelet(高速)": "wavelet",
    "Non-Local Means": "nlm",
}
assert set(BACKEND_LABELS.values()) <= set(BACKENDS) | {"auto"}


@dataclass
class JobResult:
    """One processed file's outcome, kept together for the results table."""

    name: str
    output_path: Optional[str]
    summary: str
    ok: bool


def _preview_png(path: str, tmp_dir: str, tag: str, auto_bright: bool = True) -> Optional[str]:
    """Render a small JPEG preview of a RAW (or denoised RAW) file for on-page display."""
    try:
        img = load_raw(path)
        rgb = develop(img, auto_bright=auto_bright)
        # Downscale long side to ~1600px: this is for on-screen comparison,
        # not the delivered output, so there is no reason to ship full-res
        # previews through the browser.
        h, w = rgb.shape[:2]
        scale = min(1.0, 1600.0 / max(h, w))
        if scale < 1.0:
            from PIL import Image

            small = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
            small = small.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
            out_path = os.path.join(tmp_dir, f"{tag}_preview.jpg")
            small.save(out_path, quality=90)
        else:
            out_path = os.path.join(tmp_dir, f"{tag}_preview.jpg")
            save_image(out_path, rgb, bits=8)
        return out_path
    except Exception:
        return None


def _load_profile(profile_file, iso_override: Optional[float]) -> Tuple[Optional[NoiseProfile], str]:
    """Interpret the optional uploaded profile/bank JSON."""
    if profile_file is None:
        return None, ""
    path = profile_file.name if hasattr(profile_file, "name") else profile_file
    try:
        bank = ProfileBank.load(path)
        if bank.profiles:
            iso = iso_override or 100.0
            return bank.at_iso(float(iso)), f"プロファイルバンクを読み込み(ISO {iso:g} を使用)"
    except (KeyError, ValueError, TypeError, OSError):
        pass
    try:
        return NoiseProfile.load(path), "プロファイルJSONを読み込み"
    except Exception as exc:
        return None, f"プロファイルの読み込みに失敗、ブラインド推定に切り替え({exc})"


def process_files(
    files,
    method_label: str,
    backend_label: str,
    strength: float,
    demosaic: str,
    auto_bright: bool,
    exposure: float,
    output_format_label: str,
    self_ensemble: bool,
    checkpoint_file,
    profile_file,
    iso_override: Optional[float],
    tile: int,
    progress=None,
):
    """The callback behind the 実行 button.

    Returns (ログ本文, ダウンロード可能なファイル一覧, 処理前プレビュー, 処理後プレビュー).
    """
    if not files:
        return "RAWファイルが選択されていません。", [], None, None

    method = METHOD_LABELS.get(method_label, "auto")
    backend = BACKEND_LABELS.get(backend_label, "auto")
    ext = OUTPUT_FORMATS.get(output_format_label, ".dng")

    checkpoint_path = None
    if checkpoint_file is not None:
        checkpoint_path = checkpoint_file.name if hasattr(checkpoint_file, "name") else checkpoint_file
    if method == "model" and not checkpoint_path:
        return ("「学習済みモデル」を選びましたが、チェックポイントファイルが指定されていません。"
                "先にチェックポイント(.ckpt)をアップロードするか、方式を「古典手法」か「自動」に"
                "切り替えてください。", [], None, None)

    out_dir = tempfile.mkdtemp(prefix="denoiseraw_gui_")

    profile, profile_note = _load_profile(profile_file, iso_override)

    settings = DenoiseSettings(
        method=method,
        checkpoint=checkpoint_path,
        strength=float(strength),
        classical_backend=backend,
        profile_path=None,  # profile is resolved here, not re-read from disk
        iso=iso_override or None,
        tile=int(tile),
        self_ensemble=bool(self_ensemble),
        demosaic=demosaic,
        exposure=float(exposure),
        auto_bright=bool(auto_bright),
        progress=False,  # tqdm bars are meaningless in a web UI; we report per-file below
    )

    lines: List[str] = []
    if profile_note:
        lines.append(f"[プロファイル] {profile_note}")

    outputs: List[str] = []
    results: List[JobResult] = []
    before_preview = after_preview = None
    n = len(files)
    used_names: set = set()

    for i, f in enumerate(files):
        path = f.name if hasattr(f, "name") else f
        display_name = os.path.basename(path)
        if progress is not None:
            progress((i / max(n, 1), 1.0), desc=f"{display_name} を処理中 ({i + 1}/{n})")
        try:
            img = load_raw(path)
            result = denoise_raw(img, settings, profile=profile)
            stem = os.path.splitext(display_name)[0]
            # Two selected files can share a basename (picked from different
            # folders); writing both to the same output path would silently
            # drop the first one, so disambiguate on collision.
            candidate = f"{stem}_denoised{ext}"
            suffix = 2
            while candidate in used_names:
                candidate = f"{stem}_denoised_{suffix}{ext}"
                suffix += 1
            used_names.add(candidate)
            out_path = os.path.join(out_dir, candidate)
            write_outputs(result, out_path, settings)

            m = result.metrics
            summary = (f"{display_name}: [{result.method}] {result.elapsed:.1f}秒  "
                      f"ノイズ {m['residual_noise_before']:.5f} -> {m['residual_noise_after']:.5f} "
                      f"({m.get('noise_reduction_db', 0):.1f} dB)")
            lines.append(summary)
            outputs.append(out_path)
            results.append(JobResult(display_name, out_path, summary, True))

            if i == 0:
                # Show the first file's before/after so the user can judge the
                # result visually before downloading anything.
                before_preview = _preview_png(path, out_dir, "before", auto_bright)
                after_preview = _preview_png(out_path, out_dir, "after", auto_bright)
        except Exception as exc:
            msg = f"{display_name}: 失敗 - {exc}"
            lines.append(msg)
            results.append(JobResult(display_name, None, msg, False))
            traceback.print_exc()

    if progress is not None:
        progress((1.0, 1.0), desc="完了")

    ok = sum(r.ok for r in results)
    lines.append(f"\n完了: {ok}/{len(results)} 件成功")
    return "\n".join(lines), outputs, before_preview, after_preview


def build_interface():
    import gradio as gr

    raw_extensions = [".cr2", ".cr3", ".nef", ".arw", ".raf", ".rw2", ".dng",
                     ".orf", ".pef", ".srw", ".3fr", ".iiq"]

    with gr.Blocks(title="DeNoiseRaw") as demo:
        gr.Markdown(
            "# DeNoiseRaw\n"
            "RAWファイルを選んで設定を決め、**実行**を押すだけ。"
            "学習済みモデルが無くても「古典手法」ですぐに動きます。"
        )

        with gr.Row():
            with gr.Column(scale=1):
                files = gr.File(
                    label="RAWファイルを選択(複数選択可)",
                    file_count="multiple",
                    file_types=raw_extensions,
                )

                with gr.Accordion("基本設定", open=True):
                    method = gr.Radio(
                        list(METHOD_LABELS), value=next(iter(METHOD_LABELS)), label="処理方式",
                    )
                    backend = gr.Dropdown(
                        list(BACKEND_LABELS), value="自動選択", label="古典手法バックエンド",
                        info="「処理方式」が古典手法/自動のときに使われます",
                    )
                    strength = gr.Slider(
                        0.2, 2.0, value=1.0, step=0.05, label="強さ(strength)",
                        info="1.0=推定通り。小さいほど粒状感を残し、大きいほど強く均します",
                    )
                    output_format = gr.Radio(
                        list(OUTPUT_FORMATS), value=next(iter(OUTPUT_FORMATS)), label="出力形式",
                    )

                with gr.Accordion("詳細設定", open=False):
                    checkpoint_file = gr.File(
                        label="学習済みモデル(.ckpt、任意)", file_types=[".ckpt", ".pth", ".pt"],
                    )
                    self_ensemble = gr.Checkbox(
                        label="self-ensemble(8方向平均、モデル使用時のみ・低速だが高精度)",
                        value=False,
                    )
                    profile_file = gr.File(
                        label="ノイズプロファイル JSON(任意、無ければ自動推定)",
                        file_types=[".json"],
                    )
                    iso_override = gr.Number(
                        label="ISO指定(0のままなら自動推定/ファイルのISOを使用)", value=0,
                        minimum=0, precision=0,
                    )
                    tile = gr.Slider(
                        0, 1024, value=512, step=64, label="タイルサイズ(モデル使用時、0=分割なし)",
                    )
                    demosaic = gr.Radio(
                        list(DEMOSAIC_METHODS), value="malvar", label="デモザイク方式(プレビュー用)",
                    )
                    auto_bright = gr.Checkbox(label="プレビューを自動で明るくする", value=True)
                    exposure = gr.Slider(-2.0, 4.0, value=0.0, step=0.1, label="プレビュー露出補正(段)")

                run_button = gr.Button("実行", variant="primary")

            with gr.Column(scale=1):
                log = gr.Textbox(label="処理結果", lines=10, interactive=False)
                outputs = gr.File(label="ダウンロード", file_count="multiple")
                with gr.Row():
                    before_img = gr.Image(label="処理前(1枚目)", interactive=False)
                    after_img = gr.Image(label="処理後(1枚目)", interactive=False)

        run_button.click(
            fn=process_files,
            inputs=[files, method, backend, strength, demosaic, auto_bright, exposure,
                   output_format, self_ensemble, checkpoint_file, profile_file,
                   iso_override, tile],
            outputs=[log, outputs, before_img, after_img],
        )

    return demo


def main() -> None:
    demo = build_interface()
    demo.queue().launch()


if __name__ == "__main__":
    main()
