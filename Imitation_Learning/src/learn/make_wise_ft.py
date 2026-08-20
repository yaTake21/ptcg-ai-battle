#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_wise_ft.py — WiSE-FT（2つの checkpoint の線形補間）で θ(α) を作る。

    θ(α) = (1−α)·θ_pre + α·θ_ft
    （Wortsman et al., CVPR 2022, arXiv:2109.01903）

段1（事前学習）と段2（特化）の best.pt を渡すと、α ごとに補間した checkpoint と
提出用 weights npz を書き出す。2026-08-09 の α=0.5 採用時はアドホックに計算したが、
value-weight A/B で同じ手順を繰り返すのでスクリプトに固定した。

■ 使い方
    # 既定の α（0.25 / 0.5 / 0.75）を作る
    python3 make_wise_ft.py \\
        --pre  ../../runs/Alakazam2/a2_vw0_s1/best.pt \\
        --ft   ../../runs/Alakazam2/a2_vw0_s2/best.pt \\
        --out  ../../runs/Alakazam2/wise_ft_vw0

    # α を指定して1点だけ
    python3 make_wise_ft.py --pre <s1>/best.pt --ft <s2>/best.pt \\
        --out <dir> --alpha 0.5

■ 出力（--out ディレクトリ）
    wise_<α>.pt          補間した checkpoint（train.py と同じ形。metrics に α を記録）
    wise_<α>.npz         提出用 numpy 重み（export_weights.py と同じ形式）

■ 前提と検査
    - 2つの checkpoint は同一アーキテクチャであること（model_config の一致を検査する）
    - state_dict のキー集合が完全一致すること（違えば止める）
    - 整数バッファ等 float でないテンソルは補間せず θ_pre 側をそのまま採る
      （このモデルには無いが、将来 buffer が増えたときに黙って壊れないように）

■ 補間したあとにやること
    提出 dir の weights.npz を差し替えて、アリーナで直接対決させる。
    ★val 指標では優劣を決めない（実測で強さと相関しなかったため）。
"""
from __future__ import annotations

import argparse
import json
import os

import torch

import export_weights


def load(path: str) -> dict:
    if not os.path.exists(path):
        raise SystemExit(f"[fatal] {path} が無い")
    return torch.load(path, map_location="cpu", weights_only=False)


def interpolate(pre: dict, ft: dict, alpha: float) -> dict:
    """θ(α) = (1−α)·θ_pre + α·θ_ft の checkpoint を組み立てる。"""
    sd_p, sd_f = pre["model"], ft["model"]
    if set(sd_p) != set(sd_f):
        only_p = sorted(set(sd_p) - set(sd_f))[:5]
        only_f = sorted(set(sd_f) - set(sd_p))[:5]
        raise SystemExit(f"[fatal] state_dict のキーが違う。pre のみ={only_p} ft のみ={only_f}")

    out = {}
    n_interp = n_copy = 0
    for k, a in sd_p.items():
        b = sd_f[k]
        if a.shape != b.shape:
            raise SystemExit(f"[fatal] {k} の形が違う: {tuple(a.shape)} vs {tuple(b.shape)}")
        if a.is_floating_point():
            out[k] = (1.0 - alpha) * a.float() + alpha * b.float()
            n_interp += 1
        else:
            out[k] = a.clone()          # 整数バッファ等は補間しない
            n_copy += 1
    print(f"  [interp] α={alpha:.2f}  補間 {n_interp} / そのまま {n_copy}")

    ck = dict(pre)                       # model_config / feat_config は段1 のものを引き継ぐ
    ck["model"] = out
    ck["metrics"] = {"epoch": f"wise{alpha:g}",
                     "alpha": alpha,
                     "src_pre": pre.get("metrics", {}),
                     "src_ft": ft.get("metrics", {})}
    return ck


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True, help="段1（事前学習）の best.pt … α=0 側")
    ap.add_argument("--ft", required=True, help="段2（特化）の best.pt … α=1 側")
    ap.add_argument("--out", required=True, help="出力ディレクトリ")
    ap.add_argument("--alpha", type=float, action="append",
                    help="補間係数。複数指定可（既定 0.25 / 0.5 / 0.75）")
    ap.add_argument("--no-npz", action="store_true", help="提出用 npz を書かない")
    a = ap.parse_args()

    alphas = a.alpha if a.alpha else [0.25, 0.5, 0.75]
    for x in alphas:
        if not 0.0 <= x <= 1.0:
            raise SystemExit(f"[fatal] α は 0〜1。与えられた値: {x}")

    pre, ft = load(a.pre), load(a.ft)
    cp, cf = pre.get("model_config"), ft.get("model_config")
    if cp != cf:
        raise SystemExit("[fatal] model_config が一致しない（別アーキテクチャ同士は補間できない）\n"
                         f"  pre: {cp}\n  ft : {cf}")

    vw_p = (pre.get("args") or {}).get("value_weight")
    vw_f = (ft.get("args") or {}).get("value_weight")
    print(f"[src] pre={a.pre}  (value_weight={vw_p})")
    print(f"[src] ft ={a.ft}   (value_weight={vw_f})")
    if vw_p != vw_f:
        print(f"[warn] 2段の value_weight が違う（{vw_p} vs {vw_f}）。意図した条件か確認すること")

    os.makedirs(a.out, exist_ok=True)
    for x in alphas:
        ck = interpolate(pre, ft, x)
        pt = os.path.join(a.out, f"wise_{x:g}.pt")
        torch.save(ck, pt)
        print(f"  [save] {pt}")
        if not a.no_npz:
            npz = os.path.join(a.out, f"wise_{x:g}.npz")
            export_weights.export(pt, npz)

    meta = os.path.join(a.out, "make_wise_ft.json")
    json.dump({"pre": os.path.abspath(a.pre), "ft": os.path.abspath(a.ft),
               "alphas": alphas, "value_weight": {"pre": vw_p, "ft": vw_f}},
              open(meta, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"[done] {len(alphas)} 点を書き出した -> {a.out}")
    print("       次: 提出 dir の weights.npz を差し替えてアリーナで直接対決（val では決めない）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
