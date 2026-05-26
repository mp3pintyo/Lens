from __future__ import annotations

import gc
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import gradio as gr
import torch

from lens import LensGptOssEncoder, LensPipeline, RESOLUTION_BUCKETS, resolve_resolution
from lens.resolution import SUPPORTED_ASPECT_RATIOS, SUPPORTED_BASE_RESOLUTIONS


ROOT = Path(__file__).resolve().parent
GALLERY_DIR = ROOT / "assets" / "gallery"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"
RESOLUTION_MODE_BUCKET = "Preset buckets"
RESOLUTION_MODE_CUSTOM = "Custom height x width"
KNOWN_CHECKPOINTS = {
    "microsoft/Lens": {"steps": 20, "cfg": 5.0},
    "microsoft/Lens-Turbo": {"steps": 4, "cfg": 1.0},
    "microsoft/Lens-Base": {"steps": 50, "cfg": 5.0},
}

_PIPELINE_LOCK = threading.Lock()
_PIPELINE_STATE = {"key": None, "pipe": None}

CSS = """
:root {
  --lens-ink: #1f1a16;
  --lens-muted: #675b51;
  --lens-panel: rgba(255, 251, 246, 0.84);
  --lens-accent: #bb5a2a;
  --lens-accent-2: #167a78;
  --lens-line: rgba(77, 50, 35, 0.12);
}

body {
  background:
    radial-gradient(circle at top left, rgba(242, 188, 124, 0.35), transparent 32%),
    radial-gradient(circle at top right, rgba(96, 187, 183, 0.22), transparent 28%),
    linear-gradient(180deg, #f8f0e6 0%, #f4ede4 42%, #efe7dc 100%);
  color: var(--lens-ink);
  font-family: Bahnschrift, "Aptos", "Segoe UI", sans-serif;
}

.gradio-container {
  max-width: 1440px !important;
}

.hero {
  padding: 28px 30px;
  border: 1px solid var(--lens-line);
  border-radius: 28px;
  background:
    linear-gradient(135deg, rgba(255, 248, 240, 0.92), rgba(249, 237, 220, 0.85)),
    radial-gradient(circle at top right, rgba(22, 122, 120, 0.15), transparent 34%);
  box-shadow: 0 30px 80px rgba(84, 53, 31, 0.10);
  overflow: hidden;
}

.hero h1 {
  margin: 0;
  font-size: 2.6rem;
  line-height: 1.02;
  font-weight: 800;
  letter-spacing: -0.04em;
}

.hero p {
  margin: 0;
}

.hero-subtitle {
  margin-top: 12px !important;
  max-width: 880px;
  color: var(--lens-muted);
  font-size: 1.02rem;
}

.eyebrow {
  margin-bottom: 12px !important;
  color: var(--lens-accent);
  font-size: 0.76rem;
  font-weight: 700;
  letter-spacing: 0.2em;
  text-transform: uppercase;
}

.hero-badges {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-top: 18px;
}

.hero-badges span {
  padding: 9px 14px;
  border-radius: 999px;
  border: 1px solid rgba(187, 90, 42, 0.18);
  background: rgba(255, 255, 255, 0.72);
  font-size: 0.9rem;
}

.panel,
.output-panel {
  border: 1px solid var(--lens-line);
  border-radius: 24px;
  background: var(--lens-panel);
  backdrop-filter: blur(10px);
  box-shadow: 0 18px 48px rgba(84, 53, 31, 0.08);
}

.panel {
  padding: 16px;
}

.output-panel {
  padding: 18px;
}

.section-note {
  color: var(--lens-muted);
  font-size: 0.92rem;
}

#generate-btn {
  background: linear-gradient(135deg, var(--lens-accent), #d67a43) !important;
  border: none !important;
  color: white !important;
}

#preset-btn {
  border-color: rgba(22, 122, 120, 0.35) !important;
  color: var(--lens-accent-2) !important;
}

.report code,
.resolution-hint code {
  background: rgba(255, 255, 255, 0.78);
  padding: 0.1rem 0.35rem;
  border-radius: 6px;
}

@media (max-width: 900px) {
  .hero {
    padding: 22px;
  }

  .hero h1 {
    font-size: 2rem;
  }
}
"""

HEADER_HTML = """
<div class="hero">
  <p class="eyebrow">Lens Studio</p>
  <h1>Gradio cockpit for Microsoft Lens</h1>
  <p class="hero-subtitle">
        Run the existing Lens pipeline with opt-in prompt batching, checkpoint presets, bucket or custom canvas sizes,
    MXFP4 and offload toggles, local or OpenAI-compatible prompt refinement, and optional disk export.
  </p>
  <div class="hero-badges">
    <span>Gallery-backed examples</span>
    <span>Checkpoint presets</span>
    <span>Reasoner + API override</span>
        <span>Opt-in batch prompts</span>
    <span>Research-only workflow</span>
  </div>
</div>
"""


def _torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _split_entries(text: str) -> List[str]:
    if not text:
        return []
    normalized = text.replace("|", "\n")
    return [line.strip() for line in normalized.splitlines() if line.strip()]


def _parse_prompt_entries(text: str, enable_batching: bool) -> List[str]:
    if enable_batching:
        return _split_entries(text)
    if not text:
        return []
    stripped = text.strip()
    return [stripped] if stripped else []


def _parse_optional_int(raw_value: Union[str, int, float, None], field_name: str) -> Optional[int]:
    if raw_value in (None, ""):
        return None
    try:
        return int(float(raw_value))
    except (TypeError, ValueError) as exc:
        raise gr.Error(f"{field_name} must be an integer.") from exc


def _validate_api_fields(api_url: str, api_key: str, api_model: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    clean_url = api_url.strip() or None
    clean_key = api_key.strip() or None
    clean_model = api_model.strip() or None
    if any((clean_url, clean_key, clean_model)) and not (clean_key and clean_model):
        raise gr.Error("API refinement requires both an API key and an API model. Base URL is optional.")
    return clean_url, clean_key, clean_model


def _resolve_dimensions(
    resolution_mode: str,
    base_resolution: int,
    aspect_ratio: str,
    custom_height: Optional[float],
    custom_width: Optional[float],
) -> Tuple[int, int, Optional[int], Optional[str]]:
    if resolution_mode == RESOLUTION_MODE_BUCKET:
        height, width = resolve_resolution(int(base_resolution), aspect_ratio)
        return height, width, int(base_resolution), aspect_ratio

    if custom_height is None or custom_width is None:
        raise gr.Error("Custom resolution mode requires both height and width.")

    height = int(custom_height)
    width = int(custom_width)
    if height < 256 or width < 256:
        raise gr.Error("Height and width must be at least 256 pixels.")
    if height % 16 != 0 or width % 16 != 0:
        raise gr.Error("Custom height and width must be divisible by 16 for the FLUX.2 VAE.")
    return height, width, None, None


def _resolution_hint(
    resolution_mode: str,
    base_resolution: int,
    aspect_ratio: str,
    custom_height: Optional[float],
    custom_width: Optional[float],
):
    bucket_visible = resolution_mode == RESOLUTION_MODE_BUCKET
    try:
        height, width, _, _ = _resolve_dimensions(
            resolution_mode, base_resolution, aspect_ratio, custom_height, custom_width
        )
        summary = (
            f"<div class='resolution-hint'><strong>Canvas:</strong> {width} x {height} px"
            f" &middot; <strong>Latents:</strong> {width // 16} x {height // 16} tiles</div>"
        )
    except Exception as exc:  # pragma: no cover - UI feedback branch
        summary = f"<div class='resolution-hint'><strong>Resolution:</strong> {exc}</div>"

    return gr.update(visible=bucket_visible), gr.update(visible=not bucket_visible), summary


def _checkpoint_defaults(repo_id: str) -> Tuple[int, float]:
    defaults = KNOWN_CHECKPOINTS.get((repo_id or "").strip(), KNOWN_CHECKPOINTS["microsoft/Lens"])
    return int(defaults["steps"]), float(defaults["cfg"])


def _build_text_encoder(repo_id: str, dtype_name: str, disable_mxfp4: bool) -> LensGptOssEncoder:
    text_encoder_kwargs = {"subfolder": "text_encoder", "dtype": _torch_dtype(dtype_name)}
    try:
        from transformers import Mxfp4Config

        text_encoder_kwargs["quantization_config"] = Mxfp4Config(
            dequantize=bool(disable_mxfp4)
        )
    except ImportError:
        pass
    return LensGptOssEncoder.from_pretrained(repo_id, **text_encoder_kwargs)


def _release_pipeline() -> None:
    pipe = _PIPELINE_STATE.get("pipe")
    if pipe is None:
        return

    _PIPELINE_STATE["pipe"] = None
    _PIPELINE_STATE["key"] = None
    try:
        pipe.maybe_free_model_hooks()
    except Exception:
        pass
    del pipe
    _flush_cuda_state()


def _flush_cuda_state() -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return

    try:
        torch.cuda.synchronize()
    except Exception:
        pass

    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass


def _get_pipeline(repo_id: str, dtype_name: str, disable_mxfp4: bool, offload: bool) -> Tuple[LensPipeline, str]:
    if not torch.cuda.is_available():
        raise gr.Error("Lens inference currently requires a CUDA-capable environment.")

    key = (repo_id, dtype_name, bool(disable_mxfp4), bool(offload))
    with _PIPELINE_LOCK:
        cached_pipe = _PIPELINE_STATE.get("pipe")
        if _PIPELINE_STATE.get("key") == key and cached_pipe is not None:
            return cached_pipe, "reused cached pipeline"

        _release_pipeline()

        text_encoder = _build_text_encoder(repo_id, dtype_name, disable_mxfp4)

        pipe = LensPipeline.from_pretrained(
            repo_id,
            text_encoder=text_encoder,
            torch_dtype=_torch_dtype(dtype_name),
        )
        if offload:
            pipe.enable_model_cpu_offload()
            try:
                pipe.vae.enable_tiling()
                pipe.vae.enable_slicing()
            except Exception:
                pass
        else:
            pipe.to("cuda")

        _PIPELINE_STATE["key"] = key
        _PIPELINE_STATE["pipe"] = pipe
        return pipe, "loaded new pipeline"


def _configure_reasoner(pipe: LensPipeline, api_url: Optional[str], api_key: Optional[str], api_model: Optional[str]) -> str:
    pipe.reasoner.openai_base_url = api_url
    pipe.reasoner.openai_api_key = api_key
    pipe.reasoner.openai_model = api_model
    pipe.reasoner._client = None
    if api_key and api_model:
        return "OpenAI-compatible API"
    return "local GPT-OSS" if pipe.reasoner.text_encoder is not None else "disabled"


def _parse_negative_prompt(
    negative_prompt: str,
    prompt_count: int,
    enable_batching: bool,
) -> Union[str, List[str]]:
    negatives = _parse_prompt_entries(negative_prompt, enable_batching)
    if not negatives:
        return ""
    if len(negatives) == 1:
        return negatives[0]
    if len(negatives) != prompt_count:
        raise gr.Error(
            "Negative prompt must be either one shared entry or the same number of entries as the prompt list."
        )
    return negatives


def _gallery_items(images: Sequence, prompts: Sequence[str], num_images_per_prompt: int):
    items = []
    image_index = 0
    for prompt_index, prompt in enumerate(prompts, start=1):
        title = prompt if len(prompt) <= 72 else f"{prompt[:69]}..."
        for sample_index in range(1, num_images_per_prompt + 1):
            label = f"P{prompt_index:02d} · S{sample_index:02d} · {title}"
            items.append((images[image_index], label))
            image_index += 1
    return items


def _refined_prompt_report(original: Sequence[str], refined: Sequence[str], reasoner_mode: str) -> str:
    changed = []
    for index, (src, dst) in enumerate(zip(original, refined), start=1):
        if src != dst:
            changed.append(f"[{index}] {src}\n-> {dst}")
    if changed:
        return "\n\n".join(changed)
    return f"No rewrite applied. Active prompt refinement path: {reasoner_mode}."


def _save_outputs(
    images: Sequence,
    prompts: Sequence[str],
    refined_prompts: Sequence[str],
    num_images_per_prompt: int,
    output_dir: str,
) -> str:
    save_root = Path(output_dir).expanduser() if output_dir.strip() else DEFAULT_OUTPUT_DIR
    run_dir = save_root / datetime.now().strftime("lens-%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    image_index = 0
    manifest_lines = []
    for prompt_index, prompt in enumerate(prompts):
        refined_prompt = refined_prompts[prompt_index] if prompt_index < len(refined_prompts) else prompt
        manifest_lines.append(f"[{prompt_index}] prompt={prompt}")
        manifest_lines.append(f"[{prompt_index}] refined={refined_prompt}")
        for sample_index in range(num_images_per_prompt):
            filename = f"p{prompt_index:03d}_s{sample_index:02d}.png"
            images[image_index].save(run_dir / filename)
            image_index += 1
    (run_dir / "prompts.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    return str(run_dir)


def _infer_gallery_aspect_ratio(width: int, height: int) -> str:
    target_ratio = width / height
    best_ratio = "1:1"
    best_distance = float("inf")
    for candidate in SUPPORTED_ASPECT_RATIOS:
        w_text, h_text = candidate.split(":", maxsplit=1)
        candidate_ratio = int(w_text) / int(h_text)
        distance = abs(candidate_ratio - target_ratio)
        if distance < best_distance:
            best_distance = distance
            best_ratio = candidate
    return best_ratio


def _build_examples() -> List[List[object]]:
    examples: List[List[object]] = []
    if not GALLERY_DIR.exists():
        return examples

    for txt_path in sorted(GALLERY_DIR.glob("*.txt"))[:12]:
        prompt = txt_path.read_text(encoding="utf-8").strip()
        try:
            _, raw_dims = txt_path.stem.split("-", maxsplit=1)
            width_text, height_text = raw_dims.split("x", maxsplit=1)
            width = int(width_text)
            height = int(height_text)
            base_resolution = 1440
            aspect_ratio = _infer_gallery_aspect_ratio(width, height)
        except ValueError:
            base_resolution, aspect_ratio = 1440, "1:1"

        defaults = _checkpoint_defaults("microsoft/Lens")
        examples.append([
            prompt,
            "",
            "microsoft/Lens",
            base_resolution,
            aspect_ratio,
            defaults[0],
            defaults[1],
            1,
            "42",
        ])
    return examples


def generate_images(
    prompt: str,
    negative_prompt: str,
    enable_prompt_batching: bool,
    repo_id: str,
    resolution_mode: str,
    base_resolution: int,
    aspect_ratio: str,
    custom_height: Optional[float],
    custom_width: Optional[float],
    steps: int,
    cfg: float,
    num_images_per_prompt: int,
    seed: str,
    dtype_name: str,
    disable_mxfp4: bool,
    offload: bool,
    enable_reasoner: bool,
    api_url: str,
    api_key: str,
    api_model: str,
    max_sequence_length: int,
    save_to_disk: bool,
    output_dir: str,
):
    prompts = _parse_prompt_entries(prompt, enable_prompt_batching)
    if not prompts:
        raise gr.Error("Provide at least one prompt.")

    seed_value = _parse_optional_int(seed, "Seed")
    height, width, resolved_base_resolution, resolved_aspect_ratio = _resolve_dimensions(
        resolution_mode, base_resolution, aspect_ratio, custom_height, custom_width
    )
    negative_prompt_value = _parse_negative_prompt(negative_prompt, len(prompts), enable_prompt_batching)
    clean_api_url, clean_api_key, clean_api_model = _validate_api_fields(api_url, api_key, api_model)

    pipe = None
    result = None
    generator = None
    try:
        pipe, cache_status = _get_pipeline(repo_id.strip(), dtype_name, disable_mxfp4, offload)
        reasoner_mode = _configure_reasoner(pipe, clean_api_url, clean_api_key, clean_api_model)

        with _PIPELINE_LOCK, torch.inference_mode():
            generator = (
                torch.Generator(device=pipe._execution_device).manual_seed(seed_value)
                if seed_value is not None
                else None
            )
            result = pipe(
                prompt=prompts,
                negative_prompt=negative_prompt_value,
                height=None if resolved_base_resolution is not None else height,
                width=None if resolved_base_resolution is not None else width,
                base_resolution=resolved_base_resolution,
                aspect_ratio=resolved_aspect_ratio,
                num_inference_steps=int(steps),
                guidance_scale=float(cfg),
                num_images_per_prompt=int(num_images_per_prompt),
                generator=generator,
                max_sequence_length=int(max_sequence_length),
                enable_reasoner=bool(enable_reasoner),
            )

        images = list(result.images)
        refined_prompts = list(getattr(pipe, "_last_refined_prompts", prompts))
        gallery_items = _gallery_items(images, prompts, int(num_images_per_prompt))
        refined_report = _refined_prompt_report(prompts, refined_prompts, reasoner_mode)

        saved_dir = None
        if save_to_disk:
            saved_dir = _save_outputs(images, prompts, refined_prompts, int(num_images_per_prompt), output_dir)

        report_lines = [
            "### Run summary",
            f"- Checkpoint: `{repo_id.strip()}`",
            f"- Model cache: {cache_status}",
            f"- Canvas: `{width} x {height}`",
            f"- Sampling: `{int(steps)}` steps, CFG `{float(cfg):.2f}`, `{int(num_images_per_prompt)}` image(s) per prompt",
            f"- Prompts: `{len(prompts)}`",
            f"- Prompt batching: `{bool(enable_prompt_batching)}`",
            f"- Dtype: `{dtype_name}`",
            f"- MXFP4 dequantization: `{bool(disable_mxfp4)}`",
            f"- CPU offload: `{bool(offload)}`",
            f"- Prompt refinement: `{reasoner_mode}`; local toggle set to `{bool(enable_reasoner)}`",
            f"- Max sequence length: `{int(max_sequence_length)}`",
            f"- Seed: `{seed_value}`" if seed_value is not None else "- Seed: random",
        ]
        if saved_dir is not None:
            report_lines.append(f"- Saved to: `{saved_dir}`")
        return gallery_items, refined_report, "\n".join(report_lines)
    finally:
        del generator
        del result
        _flush_cuda_state()


EXAMPLES = _build_examples()


with gr.Blocks(
    title="Lens Studio",
    theme=gr.themes.Soft(primary_hue="amber", secondary_hue="teal", neutral_hue="stone"),
    css=CSS,
) as demo:
    gr.HTML(HEADER_HTML)
    gr.Markdown(
        "Lens is research-only and expects the repo root layout used by the original project. "
        "This UI keeps the same loading path as `inference.py`, including the custom class registration that happens when `lens` is imported.",
        elem_classes=["section-note"],
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=7, elem_classes=["panel"]):
            prompt = gr.Textbox(
                label="Prompt",
                lines=7,
                placeholder="Write a single prompt. Enable batching below to split on new lines or '|'.",
            )
            enable_prompt_batching = gr.Checkbox(
                label="Enable prompt batching",
                info="When enabled, each non-empty line or '|' segment is treated as a separate prompt. When disabled, multiline text stays a single prompt.",
                value=False,
            )
            negative_prompt = gr.Textbox(
                label="Negative prompt",
                lines=3,
                placeholder="Optional. When batching is enabled, use one shared negative prompt or one per prompt line.",
            )
            with gr.Row():
                generate_button = gr.Button("Generate", variant="primary", elem_id="generate-btn")
                clear_button = gr.ClearButton(value="Reset", components=[prompt, negative_prompt])

            with gr.Row():
                repo_id = gr.Dropdown(
                    label="Checkpoint",
                    choices=list(KNOWN_CHECKPOINTS.keys()),
                    value="microsoft/Lens",
                    allow_custom_value=True,
                )
                preset_button = gr.Button("Apply checkpoint defaults", elem_id="preset-btn")

            resolution_mode = gr.Radio(
                label="Resolution mode",
                choices=[RESOLUTION_MODE_BUCKET, RESOLUTION_MODE_CUSTOM],
                value=RESOLUTION_MODE_BUCKET,
            )

            with gr.Group(visible=True) as bucket_group:
                with gr.Row():
                    base_resolution = gr.Dropdown(
                        label="Base resolution",
                        choices=list(SUPPORTED_BASE_RESOLUTIONS),
                        value=1440,
                    )
                    aspect_ratio = gr.Dropdown(
                        label="Aspect ratio",
                        choices=list(SUPPORTED_ASPECT_RATIOS),
                        value="1:1",
                    )

            with gr.Group(visible=False) as custom_group:
                with gr.Row():
                    custom_height = gr.Number(label="Custom height", value=1440, precision=0)
                    custom_width = gr.Number(label="Custom width", value=1440, precision=0)

            resolution_hint = gr.HTML()

            with gr.Accordion("Sampling", open=True):
                with gr.Row():
                    steps = gr.Slider(label="Steps", minimum=1, maximum=80, step=1, value=20)
                    cfg = gr.Slider(label="CFG scale", minimum=0.0, maximum=10.0, step=0.1, value=5.0)
                with gr.Row():
                    num_images = gr.Slider(label="Images per prompt", minimum=1, maximum=8, step=1, value=1)
                    seed = gr.Textbox(label="Seed", value="42", placeholder="Leave empty for random")
                max_sequence_length = gr.Slider(
                    label="Max sequence length",
                    minimum=128,
                    maximum=2048,
                    step=64,
                    value=512,
                )

            with gr.Accordion("Model and memory", open=False):
                with gr.Row():
                    dtype_name = gr.Radio(
                        label="Compute dtype",
                        choices=["bfloat16", "float16", "float32"],
                        value="bfloat16",
                    )
                with gr.Row():
                    disable_mxfp4 = gr.Checkbox(
                        label="Disable MXFP4 (dequantize text encoder)",
                        info="Dequantizes the GPT-OSS text encoder. This uses much more system RAM and is usually slower on RTX 3090; keep it off unless MXFP4 loading fails.",
                        value=False,
                    )
                    offload = gr.Checkbox(
                        label="Enable CPU offload",
                        info="Uses diffusers model CPU offload (text_encoder->transformer->vae) to reduce peak VRAM. Leave this off to keep the loaded modules on GPU.",
                        value=False,
                    )

            with gr.Accordion("Prompt reasoner", open=False):
                enable_reasoner = gr.Checkbox(
                    label="Enable local GPT-OSS prompt refinement",
                    info="Uses the already loaded Lens text encoder. It does not load a separate GPT-OSS model, and it follows the CPU offload setting above.",
                    value=False,
                )
                gr.Markdown(
                    "If API key and model are set below, the OpenAI-compatible endpoint takes precedence over the local reasoner.",
                    elem_classes=["section-note"],
                )
                api_url = gr.Textbox(label="API base URL", placeholder="Optional for OpenAI-compatible servers")
                with gr.Row():
                    api_model = gr.Textbox(label="API model", placeholder="Required when using API refinement")
                    api_key = gr.Textbox(label="API key", type="password", placeholder="Required when using API refinement")

            with gr.Accordion("Output", open=False):
                save_to_disk = gr.Checkbox(label="Save generated images to disk", value=True)
                output_dir = gr.Textbox(label="Output directory", value=str(DEFAULT_OUTPUT_DIR))

        with gr.Column(scale=5, elem_classes=["panel"]):
            gr.Markdown("### Gallery examples")
            gr.Markdown(
                "These prompts are loaded directly from `assets/gallery/*.txt`, so the examples stay aligned with the project repo.",
                elem_classes=["section-note"],
            )
            gr.Examples(
                examples=EXAMPLES,
                inputs=[prompt, negative_prompt, repo_id, base_resolution, aspect_ratio, steps, cfg, num_images, seed],
                label="Click to load a sample prompt",
                examples_per_page=6,
            )

    with gr.Row():
        with gr.Column(scale=8, elem_classes=["output-panel"]):
            gallery = gr.Gallery(
                label="Generated images",
                columns=2,
                height="auto",
                object_fit="contain",
                preview=True,
            )
        with gr.Column(scale=4, elem_classes=["output-panel"]):
            refined_prompts = gr.Textbox(
                label="Prompt rewrite report",
                lines=14,
                interactive=False,
            )
            report = gr.Markdown(elem_classes=["report"])

    update_inputs = [resolution_mode, base_resolution, aspect_ratio, custom_height, custom_width]
    resolution_mode.change(
        fn=_resolution_hint,
        inputs=update_inputs,
        outputs=[bucket_group, custom_group, resolution_hint],
    )
    base_resolution.change(
        fn=_resolution_hint,
        inputs=update_inputs,
        outputs=[bucket_group, custom_group, resolution_hint],
    )
    aspect_ratio.change(
        fn=_resolution_hint,
        inputs=update_inputs,
        outputs=[bucket_group, custom_group, resolution_hint],
    )
    custom_height.change(
        fn=_resolution_hint,
        inputs=update_inputs,
        outputs=[bucket_group, custom_group, resolution_hint],
    )
    custom_width.change(
        fn=_resolution_hint,
        inputs=update_inputs,
        outputs=[bucket_group, custom_group, resolution_hint],
    )

    preset_button.click(fn=_checkpoint_defaults, inputs=[repo_id], outputs=[steps, cfg])
    generate_button.click(
        fn=generate_images,
        inputs=[
            prompt,
            negative_prompt,
            enable_prompt_batching,
            repo_id,
            resolution_mode,
            base_resolution,
            aspect_ratio,
            custom_height,
            custom_width,
            steps,
            cfg,
            num_images,
            seed,
            dtype_name,
            disable_mxfp4,
            offload,
            enable_reasoner,
            api_url,
            api_key,
            api_model,
            max_sequence_length,
            save_to_disk,
            output_dir,
        ],
        outputs=[gallery, refined_prompts, report],
    )

    demo.load(
        fn=_resolution_hint,
        inputs=update_inputs,
        outputs=[bucket_group, custom_group, resolution_hint],
    )

demo.queue(default_concurrency_limit=1)


if __name__ == "__main__":
    demo.launch(
        server_name=os.environ.get("LENS_GRADIO_HOST", "127.0.0.1"),
        server_port=int(os.environ.get("LENS_GRADIO_PORT", "7860")),
        show_api=False,
    )
