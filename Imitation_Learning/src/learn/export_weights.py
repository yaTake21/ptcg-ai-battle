#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""export_weights.py — 学習済み checkpoint → 提出用 weights.npz 変換器。

■ なぜ npz にするか
  提出環境（Kaggle の cabt ランタイム）に PyTorch がある保証はない。
  そこで提出用の推論は「numpy 手書き forward」で行う方針とし、
  重みを枠組み非依存の .npz（テンソル名 → float32 配列）に落とす。
  モデル設定(ModelConfig)と特徴量設定(featurize config)も JSON で同梱するので、
  このファイル1つで推論器を再構築できる。

■ 使い方
    # src/learn/ から
    python3 export_weights.py ../../runs/<run>/best.pt \\
        -o ../../runs/<run>/weights.npz
    # 検証: torch forward と numpy 再構築の一致確認は infer_numpy.py（次工程）で行う

■ 出力 npz の中身
    <state_dict の各キー> : float32 配列（例: emb_card.weight, encoder.layers.0...）
    __model_config__      : ModelConfig の JSON 文字列
    __feat_config__       : featurize.py の設定 JSON 文字列
    __metrics__           : checkpoint 保存時の val メトリクス JSON
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch


def export(ckpt_path: str, out_path: str) -> dict:
    """checkpoint を読み、全重みを float32 numpy にして npz へ保存する。"""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model"]
    arrays = {k: v.detach().cpu().numpy().astype(np.float32)
              for k, v in sd.items()}
    arrays["__model_config__"] = np.asarray(ckpt["model_config"])
    arrays["__feat_config__"] = np.asarray(ckpt["feat_config"])
    arrays["__metrics__"] = np.asarray(json.dumps(ckpt.get("metrics", {})))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    return {
        "tensors": len(sd),
        "params": int(sum(v.size for k, v in arrays.items()
                          if not k.startswith("__"))),
        "size_mb": os.path.getsize(out_path) / 1e6,
        "metrics": ckpt.get("metrics", {}),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="checkpoint → weights.npz")
    ap.add_argument("ckpt", help="train.py の best.pt / last.pt")
    ap.add_argument("-o", "--out", required=True, help="出力 .npz")
    args = ap.parse_args()
    info = export(args.ckpt, args.out)
    m = info["metrics"]
    print(f"[export] tensors={info['tensors']} params={info['params']:,} "
          f"-> {args.out} ({info['size_mb']:.1f} MB)\n"
          f"         ckpt metrics: top1={m.get('acc_top1', float('nan')):.3f} "
          f"exact={m.get('acc_exact', float('nan')):.3f}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
