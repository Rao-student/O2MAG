import os
from typing import Optional, Tuple

import gradio as gr
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from diffusers import DDIMScheduler

from triag.diffuser_utils import LocalBlend, McaPipeline_Replace
from triag.mca_p2p import McaControlReplace
from triag.mca_utils import regiter_attention_editor_diffusers
from triag.ptp_utils import get_equalizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 改成本地路径
MODEL_PATH = "/data1/Shared/Models/stable-diffusion-v1-5"
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
START_STEP = 4
END_STEP = NUM_INFERENCE_STEPS
NEGATIVE_PROMPT = (
    "over-exposure, under-exposure, saturated, duplicate, out of frame, "
    "lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, "
    "ugly, bad anatomy, bad proportions, "
    "intact, flawless, pristine, clean, perfect condition"
)


def _ensure_model_path() -> None:
    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(
            f"Model path '{MODEL_PATH}' not found. Set SD_MODEL_PATH to your Stable Diffusion v1-5 directory."
        )


def _maybe_load_image(path: Optional[str], mode: Optional[str] = None) -> Optional[Image.Image]:
    if not path:
        return None
    if not os.path.exists(path):
        return None
    img = Image.open(path)
    if mode:
        img = img.convert(mode)
    return img


def load_pipeline() -> McaPipeline_Replace:
    _ensure_model_path()
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    )
    dtype = torch.float32  # use float32 to avoid dtype mismatches in custom attention
    pipe = McaPipeline_Replace.from_pretrained(
        MODEL_PATH, scheduler=scheduler, safety_checker=None, torch_dtype=dtype
    )
    pipe = pipe.to(DEVICE)
    return pipe


sd_model = load_pipeline()
tokenizer = sd_model.tokenizer
MODEL_DTYPE = sd_model.unet.dtype

# TODO: 替换为你自己的示例数据路径和提示词
EXAMPLES = [
    {
        "normal": "data/000.png",               # 正常图路径
        "reference": "data/anomaly/000.png",    # 参考异常图路径
        "normal_mask": "./data/0.jpg",          # 正常图 mask 路径
        "reference_mask": "data/anomaly/000_mask.png",  # 参考图 mask 路径
        "prompt": "a photo of a hazelnut with a crack",
        "focus_word": "crack",
        "ref_prompt": "",
        "src_prompt": "",
    },
    {
        "normal": "data/001.png",               # 正常图路径
        "reference": "data/anomaly/001.png",    # 参考异常图路径
        "normal_mask": "./data/1.jpg",         # 正常图 mask 路径
        "reference_mask": "data/anomaly/001_mask.png",  # 参考图 mask 路径
        "prompt": "a photo of a bottle with contamination",
        "focus_word": "contamination",
        "ref_prompt": "",
        "src_prompt": "",
    },
    {
        "normal": "data/002.png",               # 正常图路径
        "reference": "data/anomaly/002.png",    # 参考异常图路径
        "normal_mask": "./data/2.jpg",         # 正常图 mask 路径
        "reference_mask": "data/anomaly/002_mask.png",  # 参考图 mask 路径
        "prompt": "a photo of a pill with a scratch",
        "focus_word": "scratch",
        "ref_prompt": "",
        "src_prompt": "",
    },
    {
        "normal": "data/003.png",               # 正常图路径
        "reference": "data/anomaly/003.png",    # 参考异常图路径
        "normal_mask": "./data/3.jpg",         # 正常图 mask 路径
        "reference_mask": "data/anomaly/003_mask.png",  # 参考图 mask 路径
        "prompt": "a photo of a hazelnut with a hole",
        "focus_word": "hole",
        "ref_prompt": "",
        "src_prompt": "",
    },
    {
        "normal": "data/anomaly/other/building/2.png",               # 正常图路径
        "reference": "data/anomaly/other/building/9717.jpg",    # 参考异常图路径
        "normal_mask": "data/anomaly/other/building/mask/2.png",          # 正常图 mask 路径
        "reference_mask": "data/anomaly/other/building/mask/9717.png",  # 参考图 mask 路径
        "prompt": "a photo of a building with a crack",
        "focus_word": "crack",
        "ref_prompt": "",
        "src_prompt": "",
    },
    {
        "normal": "data/anomaly/other/road/132.jpg",               # 正常图路径
        "reference": "data/anomaly/other/road/8.jpg",    # 参考异常图路径
        "normal_mask": "data/anomaly/other/road/mask/1.png",         # 正常图 mask 路径
        "reference_mask": "data/anomaly/other/road/mask/8.png",  # 参考图 mask 路径
        "prompt": "a photo of a railway with a rock on it",
        "focus_word": "rock",
        "ref_prompt": "",
        "src_prompt": "",
    },
    {
        "normal": "data/anomaly/other/solarpanel/1.jpg",               # 正常图路径
        "reference": "data/anomaly/other/solarpanel/2.jpg",    # 参考异常图路径
        "normal_mask": "data/anomaly/other/solarpanel/mask/1.png",         # 正常图 mask 路径
        "reference_mask": "data/anomaly/other/solarpanel/mask/2.png",  # 参考图 mask 路径
        "prompt": "a photo of a solarpanel with a guano",
        "focus_word": "guano",
        "ref_prompt": "",
        "src_prompt": "",
    },
    {
        "normal": "data/anomaly/other/wall/1.png",               # 正常图路径
        "reference": "data/anomaly/other/wall/4.png",    # 参考异常图路径
        "normal_mask": "data/anomaly/other/wall/mask/1.png",         # 正常图 mask 路径
        "reference_mask": "data/anomaly/other/wall/mask/4.png",  # 参考图 mask 路径
        "prompt": "a photo of a wall with a crack",
        "focus_word": "crack",
        "ref_prompt": "",
        "src_prompt": "",
    },
    {
        "normal": "data/anomaly/other/wall/3.png",               # 正常图路径
        "reference": "data/anomaly/other/wall/6.png",    # 参考异常图路径
        "normal_mask": "data/anomaly/other/wall/mask/3.png",         # 正常图 mask 路径
        "reference_mask": "data/anomaly/other/wall/mask/6.png",  # 参考图 mask 路径
        "prompt": "a photo of a wall with a crack",
        "focus_word": "crack",
        "ref_prompt": "",
        "src_prompt": "",
    },
]


def _to_model_dtype(x: torch.Tensor) -> torch.Tensor:
    return x.to(device=DEVICE, dtype=MODEL_DTYPE)


def preprocess_image(img: Image.Image) -> torch.Tensor:
    if img.mode != "RGB":
        img = img.convert("RGB")
    tensor = T.ToTensor()(img)  # [3,H,W], 0-1
    tensor = tensor.unsqueeze(0) * 2 - 1  # [-1, 1]
    tensor = F.interpolate(tensor, (512, 512), mode="bilinear", align_corners=False)
    return _to_model_dtype(tensor)


def preprocess_mask(mask: Image.Image) -> torch.Tensor:
    mask = mask.convert("L")
    tensor = T.ToTensor()(mask)  # [1,H,W], 0-1
    tensor = (tensor > 0.5).float().squeeze(0)
    tensor = F.interpolate(tensor.unsqueeze(0).unsqueeze(0), (512, 512), mode="nearest")
    tensor = tensor.squeeze(0).squeeze(0)
    return _to_model_dtype(tensor)


def _build_equalizer(prompt: str, focus_word: Optional[str]) -> torch.Tensor:
    if focus_word:
        return get_equalizer(prompt, (focus_word,), (100,), tokenizer).to(device=DEVICE, dtype=MODEL_DTYPE)
    return torch.ones(1, 77, device=DEVICE, dtype=MODEL_DTYPE)


def load_example(idx: int):
    cfg = EXAMPLES[idx]
    return (
        _maybe_load_image(cfg["normal"], mode="RGB"),
        _maybe_load_image(cfg["reference"], mode="RGB"),
        _maybe_load_image(cfg["normal_mask"], mode="L"),
        _maybe_load_image(cfg["reference_mask"], mode="L"),
        cfg["prompt"],
        cfg["focus_word"],
        cfg["ref_prompt"],
        cfg["src_prompt"],
    )


def _sanitize_pair(start: float, end: float) -> Tuple[float, float]:
    start = max(0.0, min(1.0, start))
    end = max(0.0, min(1.0, end))
    if end < start:
        start, end = end, start
    return start, end


def generate_anomaly(
    normal_image: Image.Image,
    reference_image: Image.Image,
    normal_mask: Image.Image,
    reference_mask: Image.Image,
    target_prompt: str,
    focus_word: str,
    self_start: float,
    self_end: float,
    cross_start: float,
    cross_end: float,
    start_layer: int,
    end_layer: int,
    ref_prompt: str,
    src_prompt: str,
    progress=gr.Progress(track_tqdm=True),
):
    if any(x is None for x in [normal_image, reference_image, normal_mask, reference_mask]):
        return None, "请同时上传正常图、参考异常图和两张mask。"
    self_start, self_end = _sanitize_pair(self_start, self_end)
    cross_start, cross_end = _sanitize_pair(cross_start, cross_end)
    start_layer = int(start_layer)
    end_layer = int(end_layer)
    if end_layer <= start_layer:
        return None, "END_LAYPER 需大于 START_LAYPER。"

    try:
        with torch.inference_mode():
            src_img = preprocess_image(normal_image)
            ref_img = preprocess_image(reference_image)
            src_mask = preprocess_mask(normal_mask)
            ref_mask = preprocess_mask(reference_mask)

            lbl = LocalBlend(src_mask.float())
            prompts = [target_prompt, target_prompt, target_prompt]

            ref_latent, ref_latents_list = sd_model.invert(
                ref_img,
                ref_prompt,
                guidance_scale=GUIDANCE_SCALE,
                num_inference_steps=NUM_INFERENCE_STEPS,
                return_intermediates=True,
            )
            src_latent, src_latents_list = sd_model.invert(
                src_img,
                src_prompt,
                guidance_scale=GUIDANCE_SCALE,
                num_inference_steps=NUM_INFERENCE_STEPS,
                return_intermediates=True,
            )
            ref_latent = ref_latent.expand(len(prompts), -1, -1, -1)
            src_latent = src_latent.expand(len(prompts), -1, -1, -1)

            equalizer = _build_equalizer(target_prompt, focus_word.strip())
            editor = McaControlReplace(
                prompts,
                tokenizer,
                [0, 1, 2, 3],
                (self_start, self_end),
                (cross_start, cross_end),
                equalizer,
                START_STEP,
                END_STEP,
                start_layer,
                end_layer,
                mask_s=ref_mask.float(),
                mask_t=src_mask.float(),
            )
            regiter_attention_editor_diffusers(sd_model, editor)

            image = sd_model(
                prompts,
                latents=src_latent,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                ref_intermediate_latents=[ref_latents_list, src_latents_list],
                lbl=lbl,
                neg_prompt=NEGATIVE_PROMPT,
                mask_r=ref_mask.float(),
                mask_t=src_mask.float(),
            )

            if isinstance(image, tuple):
                image = image[0]
            result = image[-1].detach().cpu().clamp(0, 1)
            return T.ToPILImage()(result), "生成完成"
    except Exception as exc:  # pragma: no cover - surface clean error
        return None, f"生成失败: {exc}"


with gr.Blocks() as demo:
    gr.Markdown(
        "# Mask-guided anomaly generation\n"
        "上传正常图、参考异常图以及对应两张mask，调节四个关键参数后生成带异常的目标图像。"
    )
    with gr.Row():
        with gr.Column():
            normal_image = gr.Image(label="正常图 (target)", type="pil", height=320)
            reference_image = gr.Image(label="参考异常图 (source)", type="pil", height=320)
        with gr.Column():
            normal_mask = gr.Image(
                label="正常图 mask (白色为生成区域)", type="pil", image_mode="L", height=320
            )
            reference_mask = gr.Image(
                label="参考异常图 mask (白色为异常区域)", type="pil", image_mode="L", height=320
            )

    with gr.Row():
        target_prompt = gr.Textbox(
            label="目标提示词 (Target Prompt)",
            value="a photo of a hazelnut with a crack",
        )
        focus_word = gr.Textbox(
            label="需要放大的token (可选，例: crack)", value="crack", placeholder="留空则不放大"
        )
    with gr.Row():
        ref_prompt = gr.Textbox(label="参考图提示词 (Reference Prompt)", value="")
        src_prompt = gr.Textbox(label="正常图提示词 (Source Prompt)", value="")

    with gr.Accordion("示例预设 (点击按钮自动填充图片和提示词)", open=False):
        gr.Markdown("请在代码中的 EXAMPLES 列表填写四组路径/提示词。点击下方按钮自动加载对应输入。")
        btns = []
        for i in range(9):
            if i==0:
                btns.append(gr.Button(f"示例 {i+1}: hazelnut with a crack", variant="secondary"))
            if i==1:
                btns.append(gr.Button(f"示例 {i+1}: bottle with a contamination", variant="secondary"))
            if i==2:
                btns.append(gr.Button(f"示例 {i+1}: pill with a scratch", variant="secondary"))
            if i==3:
                btns.append(gr.Button(f"示例 {i+1}: hazelnut with a hole in zero-shot setting", variant="secondary"))
            if i==4:
                btns.append(gr.Button(f"示例 {i+1}: building anomaly", variant="secondary"))
            if i==5:
                btns.append(gr.Button(f"示例 {i+1}: railway anomaly", variant="secondary"))
            if i==6:
                btns.append(gr.Button(f"示例 {i+1}: solarpanel anomaly", variant="secondary"))
            if i==7:
                btns.append(gr.Button(f"示例 {i+1}: wall anomaly", variant="secondary"))
            if i==8:
                btns.append(gr.Button(f"示例 {i+1}: wall with a crack in zero-shot setting", variant="secondary"))
        for i, btn in enumerate(btns):
            btn.click(
                fn=lambda idx=i: load_example(idx),
                inputs=None,
                outputs=[
                    normal_image,
                    reference_image,
                    normal_mask,
                    reference_mask,
                    target_prompt,
                    focus_word,
                    ref_prompt,
                    src_prompt,
                ],
                queue=False,
            )

    with gr.Row():
        self_start = gr.Slider(0, 1, value=0.0, step=0.01, label="self_replace_steps 起始")
        self_end = gr.Slider(0, 1, value=0.10, step=0.01, label="self_replace_steps 结束")
    with gr.Row():
        cross_start = gr.Slider(0, 1, value=0.40, step=0.01, label="cross_replace_steps 起始")
        cross_end = gr.Slider(0, 1, value=0.80, step=0.01, label="cross_replace_steps 结束")
    with gr.Row():
        start_layer = gr.Slider(0, 16, value=9, step=1, label="START_LAYPER (含)")
        end_layer = gr.Slider(1, 16, value=16, step=1, label="END_LAYPER (不含)")

    generate_btn = gr.Button("生成图像", variant="primary")
    output_image = gr.Image(label="生成结果", type="pil", height=320)
    status = gr.Textbox(label="状态", interactive=False)
    gr.ClearButton(
        [normal_image, reference_image, normal_mask, reference_mask, output_image, status],
        value="清空所有输入",
    )

    generate_btn.click(
        fn=generate_anomaly,
        inputs=[
            normal_image,
            reference_image,
            normal_mask,
            reference_mask,
            target_prompt,
            focus_word,
            self_start,
            self_end,
            cross_start,
            cross_end,
            start_layer,
            end_layer,
            ref_prompt,
            src_prompt,
        ],
        outputs=[output_image, status],
    )


if __name__ == "__main__":
    demo.queue(api_open=False).launch()
