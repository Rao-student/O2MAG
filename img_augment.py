#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from pathlib import Path
import cv2
import numpy as np

SRC_BASE = "./mvtec_data"
DST_BASE = "./data_agument"
CATEGORIES = [
    'bottle','cable','capsule','carpet','grid','hazelnut','leather',
    'metal_nut','pill','screw','tile','toothbrush','transistor',
    'wood','zipper'
]

TARGET_CNT = 1000
RANDOM_SEED = 2025
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------- 基础工具 ----------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def list_images(d: Path):
    return sorted([p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS])

def imread_color(p: Path):
    return cv2.imread(str(p), cv2.IMREAD_COLOR)

def imwrite_png(dst_dir: Path, idx: int, img):
    name = f"{idx:03d}.png"
    cv2.imwrite(str(dst_dir / name), img)

# ---------- 原子变换 ----------
def rotate_center(img, max_deg=15):
    if max_deg <= 0:
        return img
    h, w = img.shape[:2]
    angle = random.uniform(-max_deg, max_deg)
    if abs(angle) < 1e-3:
        return img
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)

def maybe_flip(img, allow=True):
    if not allow:
        return img
    r = random.random()
    if r < 0.5:       # 水平翻转
        return cv2.flip(img, 1)
    elif r < 0.7:     # 垂直翻转（小概率）
        return cv2.flip(img, 0)
    elif r < 0.8:     # 水平+垂直
        return cv2.flip(img, -1)
    return img

def vshift(img, max_pct=0.03):
    """仅沿Y方向平移，max_pct=0.03 表示 ±3% 高度"""
    if max_pct <= 0:
        return img
    h, w = img.shape[:2]
    dy = int(round(random.uniform(-max_pct, max_pct) * h))
    if dy == 0:
        return img
    M = np.float32([[1, 0, 0], [0, 1, dy]])
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)

# ---------- 逐类策略 ----------
DEFAULT = dict(allow_flip=True, rotate_deg=15, vshift_pct=0.0, copy_only=False)
CLASS_RULES = {
    # 只列出特殊类；未列出者用 DEFAULT
    "capsule":   dict(allow_flip=False,  rotate_deg=0,  vshift_pct=0.0, copy_only=False),  # 仅翻转
    "cable":     dict(allow_flip=False, rotate_deg=8,  vshift_pct=0.0, copy_only=False),  # 仅旋转≤8°
    "pill":      dict(allow_flip=False, rotate_deg=0,  vshift_pct=0.0, copy_only=True),   # 只复制
    "toothbrush":dict(allow_flip=False, rotate_deg=0,  vshift_pct=0.0, copy_only=True),   # 只复制
    "transistor":dict(allow_flip=False, rotate_deg=0,  vshift_pct=0.03, copy_only=False), # 仅上下平移
    "wood":      dict(allow_flip=True,  rotate_deg=8,  vshift_pct=0.0, copy_only=False),  # 旋转≤8°
    "zipper":    dict(allow_flip=True,  rotate_deg=0,  vshift_pct=0.0, copy_only=False),  # 仅翻转
}

def get_policy(cls: str):
    p = DEFAULT.copy()
    p.update(CLASS_RULES.get(cls, {}))
    return p

def augment_by_policy(img, policy):
    if policy["copy_only"]:
        return img
    # 注意顺序：先翻转→旋转→（可选）上下平移（仅 transistor）
    out = maybe_flip(img, allow=policy["allow_flip"])
    out = rotate_center(out, max_deg=policy["rotate_deg"])
    if policy["vshift_pct"] > 0:
        out = vshift(out, max_pct=policy["vshift_pct"])
    return out

# ---------- 主流程 ----------
def process_one_class(cls: str):
    src_dir = Path(SRC_BASE) / cls / "train" / "good"
    dst_dir = Path(DST_BASE) / cls
    ensure_dir(dst_dir)

    src_imgs = list_images(src_dir)
    if not src_imgs:
        print(f"[{cls}] 源目录无图：{src_dir}")
        return

    policy = get_policy(cls)
    print(f"[{cls}] policy = {policy}")

    next_idx = 0
    written = 0

    # 1) 先原样复制源图（若源图>500，只取前 500）
    base_copy = min(len(src_imgs), TARGET_CNT)
    for i in range(base_copy):
        img = imread_color(src_imgs[i])
        if img is None:
            continue
        imwrite_png(dst_dir, next_idx, img)
        next_idx += 1
        written += 1

    # 2) 再做“按规则”的增强直到 500；若 copy_only=True，则是重复拷贝（内容不变）
    need = max(0, TARGET_CNT - written)
    for _ in range(need):
        sp = random.choice(src_imgs)
        img = imread_color(sp)
        if img is None:
            continue
        aug = augment_by_policy(img, policy)
        imwrite_png(dst_dir, next_idx, aug)
        next_idx += 1
        written += 1

    print(f"[{cls}] 完成：写入 {written} 张 → {dst_dir}")

def main():
    for cls in CATEGORIES:
        process_one_class(cls)

if __name__ == "__main__":
    main()
