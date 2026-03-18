#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Tuple, Iterator
import os
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms as TvT
from PIL import Image
from tqdm import tqdm, trange
from torch.optim import Adam
from torchvision import transforms
from torchvision.transforms.functional import pil_to_tensor

from diffusers import DDIMScheduler
from diffusers.utils import load_image
from triag.diffuser_utils import McaPipeline_Replace, LocalBlend
from triag.mca_utils import regiter_attention_editor_diffusers
from triag.mca_p2p import McaControlReplace

from triag.ptp_utils import get_equalizer, expand_mask_tensor
from triag.ptp_utils import MVTecBankSimple, show_overlay_pil

from triag.mask_select import best_mvtec_mask_for_gen

# prompt optimization
from triag.prompt_optimize import StableDiffusion

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ------------------------------ I/O utils ------------------------------

def load_image_512(image_path: str, device: torch.device) -> torch.Tensor:
    """RGB -> [-1,1], resize to 512x512, NCHW float32 on device."""
    image = read_image(image_path, mode=ImageReadMode.RGB)
    img = image[:3].unsqueeze(0).float() / 127.5 - 1.0
    img = F.interpolate(img, (512, 512), mode="bilinear", align_corners=False)
    return img.to(device)

def load_mask_binary(mask_path: str, device: torch.device) -> torch.Tensor:
    """L mask -> [H,W] float {0,1} on device."""
    m = Image.open(mask_path).convert("L").resize((512, 512), Image.NEAREST)
    m = TvT.ToTensor()(m).squeeze(0)          # [H,W] in [0,1]
    m = (m > 0.5).float().to(device)
    return m

def read_pairs_file(p: Path) -> List[Tuple[str, str]]:
    pairs = []
    with open(p, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "+" not in line:
                print(f"[warn] skip malformed line (no '+'): {line}")
                continue
            cls, anom = line.split("+", 1)
            cls, anom = cls.strip(), anom.strip()
            if cls and anom:
                pairs.append((cls, anom))
    return pairs

def iter_ref_and_mask_pairs(cls_dir: Path, anom: str) -> Iterator[Tuple[str, str, str]]:
    """
    Yield (ref_img_path, ref_mask_path, stem) matched by stem:
    test/{anom}/{stem}.png  <->  ground_truth/{anom}/{stem}_mask.png
    """
    test_dir = cls_dir / "test" / anom
    gt_dir   = cls_dir / "ground_truth" / anom
    if not test_dir.is_dir() or not gt_dir.is_dir():
        print(f"[warn] missing dir: {test_dir} or {gt_dir}")
        return

    img_map  = {p.stem: p for p in test_dir.glob("*.png")}
    mask_map = {p.stem.replace("_mask", ""): p for p in gt_dir.glob("*_mask.png")}
    stems = sorted(set(img_map) & set(mask_map))

    # print("missing_img:", sorted(set(mask_map)-set(img_map)))
    # print("missing_mask:", sorted(set(img_map)-set(mask_map)))

    for s in stems:
        yield str(img_map[s]), str(mask_map[s]), s


# ------------------------------ saving utils ------------------------------

def save_output_image(out, save_path: Path):
    def _to_pil_from_tensor(t: torch.Tensor) -> Image.Image:
        if t.dim() == 4:  # [B,C,H,W]
            t = t[-1]
        t = t.detach().cpu()
        # 反归一化：[-1,1] -> [0,1]
        if t.min() < 0 or t.max() > 1:
            t = (t.clamp(-1, 1) + 1) * 0.5
        t = t.clamp(0, 1)
        return TvT.ToPILImage()(t)

    if isinstance(out, Image.Image):
        pil = out
    elif torch.is_tensor(out):
        pil = _to_pil_from_tensor(out)
    elif isinstance(out, (list, tuple)) and out:
        last = out[-1]
        if isinstance(last, Image.Image):
            pil = last
        elif torch.is_tensor(last):
            pil = _to_pil_from_tensor(last)
        else:
            raise TypeError(f"Unsupported output type inside list: {type(last)}")
    else:
        raise TypeError(f"Unsupported output type: {type(out)}")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(save_path)


# ------------------------------ task runner ------------------------------

def run_one_task(device_str: str, task: Dict, model_path: str,
                 steps: int, guidance: float, negative_prompt: str):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)

    # 1) Build scheduler & pipeline once per process
    scheduler = DDIMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
        clip_sample=False, set_alpha_to_one=False
    )
    pipe = McaPipeline_Replace.from_pretrained(model_path, scheduler=scheduler).to(device)

    # (Optional) Avoid VAE half-precision causing a white cast.
    try:
        pipe.vae.to(dtype=torch.float32)
        pipe.enable_vae_slicing()
    except Exception:
        pass

    tokenizer = pipe.tokenizer

    # 2) Load data
    ref_img = load_image_512(task["ref_image"], device)
    src_img = load_image_512(task["src_image"], device)
    mask_ref = load_mask_binary(task["ref_mask"], device)
    mask_source = load_mask_binary(task["source_mask"], device)

    # 3) Inversion
    ref_start_code, ref_latents_list = pipe.invert(
        ref_img, task["ref_prompt"],
        guidance_scale=guidance, num_inference_steps=steps,
        return_intermediates=True
    )
    src_start_code, src_latents_list = pipe.invert(
        src_img, task["src_prompt"],
        guidance_scale=guidance, num_inference_steps=steps,
        return_intermediates=True
    )

    # 4) Attention controller + local blend
    lbl = LocalBlend(mask_source.float())

    # AGO Module
    bank = MVTecBankSimple(bank_path=task["bank_path"])
    diffusion_model = StableDiffusion(
        device=device,
        model_path=model_path,
        fp16=False, # TODO: use fp16
    )

    opt_embeddings = None
    # optimize target prompt to algin with reference image
    step_prompt_opt = 500
    diffusion_model.get_text_embeds([task["prompts"][2]], [negative_prompt])
    for p in diffusion_model.parameters():
        p.requires_grad_(False)
    diffusion_model.embeddings['pos'].requires_grad_(True)

    if bank.exists(task["class"], task["anomaly"], task["ref_image"]):
        emb_cpu = bank.load(task["class"], task["anomaly"], task["ref_image"])
        diffusion_model.embeddings['pos'].data.copy_(
            emb_cpu.to(diffusion_model.embeddings['pos'].device,
                    dtype=diffusion_model.embeddings['pos'].dtype)
        )
        print("[Bank] HIT")
        opt_embeddings = diffusion_model.get_embedding()
    else:
        print("[Bank] MISS → run optimization ...")
        optimizer = Adam([diffusion_model.embeddings['pos']], lr=3e-3)
        # load image
        input_image = load_image(task["ref_image"]).resize((512, 512))
        input_image = (pil_to_tensor(input_image).to(torch.float16) / 255.).to(device).unsqueeze(0)
        # Text Embedding Optimization
        pbar = trange(step_prompt_opt, desc='prompt optimization', leave=True)
        for _ in pbar:
            optimizer.zero_grad()
            loss = diffusion_model.train_step(
                input_image.clone().detach()
            )
            loss.backward(retain_graph=True)
            optimizer.step()
            pbar.set_description(f"Step A, loss: {loss.item():.3f}")            

        opt_embeddings = diffusion_model.get_embedding()
        bank.save(task["class"], task["anomaly"], task["ref_image"], diffusion_model.embeddings['pos'])
    del diffusion_model


    equalizer = get_equalizer(task["prompts"][2], (task["anomaly"],), (100,), tokenizer).to(device)
    editor = McaControlReplace(task["prompts"], tokenizer, [0,1,2,3], 
                            self_replace_steps=0.1, 
                            cross_replace_steps=(0.4, 0.8), equalizer=equalizer, 
                            start_step=4, end_step=steps,
                            start_layer=9, end_layer=16, 
                            mask_s=mask_ref.float(), mask_t=mask_source.float())
    regiter_attention_editor_diffusers(pipe, editor)

    # 5) Generate
    out = pipe(
        task["prompts"],
        latents=src_start_code.expand(len(task["prompts"]), -1, -1, -1),
        num_inference_steps=steps,
        guidance_scale=guidance,
        ref_intermediate_latents=[ref_latents_list, src_latents_list],
        lbl=lbl,
        neg_prompt=negative_prompt,
        mask_r=mask_ref.float(),
        mask_t=mask_source.float(),
        opt_embeddings=opt_embeddings,
        output_type="pt",
    )

    save_output_image(out, Path(task["save_path"]))

    # # choose to save mask-image overlay
    out = transforms.ToPILImage()(out[-1:].squeeze())
    out = show_overlay_pil(out, mask_source, title = "Contrast")
    # save_output_image(out, Path(task["save_path"]))
    out.savefig(task["overlay_path"], bbox_inches="tight")


def worker(device_str, tasks, model_path, steps, guidance, negative_prompt, progress=None):
    for t in tasks:
        try:
            run_one_task(device_str, t, model_path, steps, guidance, negative_prompt)
        except Exception as e:
            print(f"[{device_str}] Error on {t.get('save_path', '<unk>')}: {e}")
        finally:
            if progress is not None:
                progress.put(1)


# ------------------------------ build tasks ------------------------------
def build_tasks_from_pairs(root: Path, root_source: Path, pairs: List[Tuple[str, str]], bank_path: str, outputs_path, normal_path) -> List[Dict]:
    tasks: List[Dict] = []
    print("searching the best mvtec mask for generated mask")
    for cls, anom in tqdm(pairs):
        cls_dir = root / cls
        # good_dir = root_source / cls / anom / "ori"
        good_dir = Path(normal_path)
        good_dir = good_dir / cls
        # print("=============good_dir:  ", good_dir, "===================")
        source_masks_dir = root_source / cls / anom
        all_good_images = sorted([p for p in good_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
        os.makedirs(str(Path(outputs_path) / cls / anom), exist_ok=True)
        os.makedirs(str(Path(outputs_path+'_overlay') / cls / anom), exist_ok=True)
        # all_good_images = sorted([img for img in good_dir.glob("*.jpg") if img.is_file()])
        # source_masks = sorted([img for img in source_masks_dir.glob("*.jpg") if img.is_file()])
        # src_img = cls_dir / "train" / "good" / "000.png"
        for src_img in all_good_images:
            # print("=============src_img:  ", src_img)
            if not src_img.is_file():
                print(f"[warn] missing source image: {src_img}")
                continue
            
            # anomalydiffusion jpg
            source_mask = os.path.join(source_masks_dir, f"{int(os.path.splitext(src_img.name)[0])}.jpg")
            # Seas png
            # source_mask = os.path.join(source_masks_dir, f"{os.path.splitext(src_img.name)[0]}.png")
            mvtec_ano_root = cls_dir / 'ground_truth' / anom
            ref_mask_name, _ = best_mvtec_mask_for_gen(str(source_mask), mvtec_ano_root, size_ratio_min=0.7, return_basename=True)
            ref_mask = mvtec_ano_root / ref_mask_name
            ref_img_name = ref_mask_name[0:3] + ".png"
            ref_img = cls_dir / 'test' / anom / ref_img_name
            src_img = good_dir / src_img
            stem = src_img.name[:-4]

            prompts = [
                    f"a photo of a {cls} with a {anom.replace('_',' ')}",
                    f"a photo of a {cls}",
                    f"a photo of a {cls} with a {anom.replace('_',' ')}",
                ]
            tasks.append({
                "class": cls,
                "anomaly": anom,
                "stem": stem,
                "src_image": str(src_img),
                "ref_image": str(ref_img),
                "source_mask": str(source_mask),
                "ref_mask": str(ref_mask),
                "src_prompt": "",
                "ref_prompt": "",
                "prompts": prompts,
                "save_path": str(Path(outputs_path) / cls / anom / f"{int(stem):03d}_{str(ref_img)[-7:-4]}_triag.png"),
                "bank_path": bank_path,
                "overlay_path": str(Path(outputs_path+'_overlay') / cls / anom / f"{int(stem):03d}.png")
            })
        print(f"------ Already Process {cls},{anom} -----")
    return tasks


# ------------------------------ main ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./mvtec_data")
    ap.add_argument("--sourece_image_mask", default="./anomalydiffusion/generated_mask")
    ap.add_argument("--outputs_path", type=str, default="results_final_fewshot_toothbrush")
    ap.add_argument("--normal_path", type=str, default="./data_agument")
    ap.add_argument("--pairs-file", default="name-mvtec.txt",
                    help='Lines like "bottle+broken_large"')
    ap.add_argument("--embedding_file", default="./mvtec_embed_bank")
    ap.add_argument("--classes", default="all", help='Comma list or "all"')
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--devices", default="cuda:4,cuda:5,cuda:6,cuda:7", help='e.g. "cuda:1,cuda:2,cuda:3,cuda:7"')
    ap.add_argument("--model-path", default="/data1/Shared/Models/stable-diffusion-v1-5")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=7.5)
    args = ap.parse_args()

    root = Path(args.root)
    root_source = Path(args.sourece_image_mask)
    outputs_path = args.outputs_path
    normal_path = args.normal_path
    bank_path = args.embedding_file
    classes = [c.strip() for c in args.classes.split(",")] if args.classes else ["all"]

    negative_prompt = (
        # "over-exposure, under-exposure, saturated, duplicate, out of frame, "
        # "lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, "
        # "ugly, bad anatomy, bad proportions, "
        "intact shell, smooth surface, pristine, flawless, clean, "
        "no crack, no cut, no scratch, no squeeze, perfect condition"
    )

    # Build tasks
    pairs_file = Path(args.pairs_file) if args.pairs_file else None
    if pairs_file and pairs_file.is_file():
        pairs = read_pairs_file(pairs_file)
        tasks = build_tasks_from_pairs(root, root_source, pairs, bank_path, outputs_path, normal_path)

    if not tasks:
        print("No tasks found. Check dataset structure and class names.")
        return

    # Run
    if args.devices:
        devs = [d.strip() for d in args.devices.split(",") if d.strip()]
        chunks = [tasks[i::len(devs)] for i in range(len(devs))]
        mp.set_start_method("spawn", force=True)
        progress = mp.Queue()
        procs = []
        start = time.time()

        for dev, chunk in zip(devs, chunks):
            p = mp.Process(target=worker,
                           args=(dev, chunk, args.model_path, args.steps, args.guidance, negative_prompt, progress))
            p.start()
            procs.append(p)

        total = len(tasks)
        done = 0
        with tqdm(total=total, desc="Total", unit="task") as pbar:
            while done < total:
                progress.get()
                done += 1
                pbar.update(1)

        for p in procs:
            p.join()

        cost = time.time() - start
        print(f"[Done] {total} tasks finished in {cost/60:.2f} min ({cost:.1f} s).")

    else:
        start = time.time()
        for t in tqdm(tasks, desc="Total", unit="task"):
            worker(args.device, [t], args.model_path, args.steps, args.guidance, negative_prompt)
        cost = time.time() - start
        print(f"[Done] {len(tasks)} tasks finished in {cost/60:.2f} min ({cost:.1f} s).")


if __name__ == "__main__":
    main()
