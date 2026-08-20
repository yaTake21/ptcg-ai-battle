#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dataset.py — featurize.py の .npz を PyTorch 学習用に供給する。

■ 役割
  1. npz のロードと ragged（可変長）復元
       featurize.py は可変長トークン列を「flat 配列 + per-sample カウント」で
       保存している。ここで cumsum offset を作り、サンプル i のスライスを O(1) で
       取り出せるようにする。
  2. **episode 単位の train/val 分割**
       サンプル単位で割ると同一試合の酷似盤面が train/val 両方に入り
       精度を過大評価する（リーク）。episode_id でグループ化し試合ごとに割る。
  3. 学習曲線用のサブサンプリング（--data-fraction）
       これも **episode 単位**で間引く（25%/50%/100% の3点で val 精度を
       比較し、データ追加が効くかを判断する。§4.2）。
  4. collate: バッチ内最大長へのパディングと各種マスク生成
       tok_mask（実トークン）/ opt_mask（候補手トークン）/ label multi-hot。

■ 出力バッチ（collate の返り値。model.forward の入力契約）
    tok_type/card_id/aux_card_id/attack_id/opt_type/area : [B, Tmax] long
    numeric   : [B, Tmax, NF] float
    tok_mask  : [B, Tmax] bool（True = 実トークン）
    opt_mask  : [B, Tmax] bool（True = 候補手トークン）
    sel_type/sel_ctx : [B] long
    label_mh  : [B, Kmax] float（正解 multi-hot。Kmax = バッチ内最大候補数）
    won       : [B] float / turn: [B] long / n_options: [B] long / is_forced: [B] bool
"""
from __future__ import annotations

import json

import numpy as np
import torch
from torch.utils.data import Dataset


class PTCGFeatureData:
    """npz を読み込み、episode 分割と ragged アクセスを提供するコンテナ。

    Attributes:
        feat_cfg : featurize.py が書き出した特徴量レイアウト設定（dict）
        n        : 総サンプル数
        episodes : ユニーク episode_id（ソート済み）
    """

    TOKEN_FIELDS = ("tok_type", "card_id", "aux_card_id",
                    "attack_id", "opt_type", "area")

    def __init__(self, npz_path: str):
        z = np.load(npz_path, allow_pickle=False)
        self.feat_cfg = json.loads(str(z["config_json"]))
        self.card_static = z["card_static"]
        self.attack_static = z["attack_static"]

        # flat フィールドと offset（cumsum）を保持
        self._flat = {k: z[k] for k in self.TOKEN_FIELDS}
        self._numeric = z["numeric"]
        self.tok_count = z["tok_count"]
        self.opt_count = z["opt_count"]
        # 注意: cumsum(uint8)->uint64 と int の連結は float64 になる(numpy昇格規則)
        # ため、オフセットは int64 に明示キャストする
        self._tok_off = np.concatenate(
            [[0], np.cumsum(self.tok_count, dtype=np.int64)]).astype(np.int64)
        self.label_count = z["label_count"]
        self._label_flat = z["label_flat"]
        self._label_off = np.concatenate(
            [[0], np.cumsum(self.label_count, dtype=np.int64)]).astype(np.int64)

        # per-sample メタ
        self.won = z["won"]
        self.turn = z["turn"]
        self.sel_type = z["sel_type"]
        self.sel_ctx = z["sel_ctx"]
        self.n_options = z["n_options"]
        self.is_forced = z["is_forced"]
        self.episode_id = z["episode_id"]
        self.n = len(self.tok_count)
        self.episodes = sorted(set(self.episode_id.tolist()))

    # ------------------------------------------------------------------
    def sample(self, i: int) -> dict:
        """サンプル i の生 numpy スライスを返す（コピーなし）。"""
        a, b = self._tok_off[i], self._tok_off[i + 1]
        la, lb = self._label_off[i], self._label_off[i + 1]
        out = {k: v[a:b] for k, v in self._flat.items()}
        out["numeric"] = self._numeric[a:b]
        out["label"] = self._label_flat[la:lb]
        out["opt_count"] = int(self.opt_count[i])
        out["sel_type"] = int(self.sel_type[i])
        out["sel_ctx"] = int(self.sel_ctx[i])
        out["won"] = float(self.won[i])
        out["turn"] = int(self.turn[i])
        out["n_options"] = int(self.n_options[i])
        out["is_forced"] = bool(self.is_forced[i])
        return out

    # ------------------------------------------------------------------
    def split_by_episode(self, val_frac: float = 0.1, seed: int = 42,
                         data_fraction: float = 1.0
                         ) -> tuple[np.ndarray, np.ndarray]:
        """episode 単位で train/val のサンプル index を返す（リーク防止 §4.2）。

        Args:
            val_frac     : validation に回す episode の割合
            seed         : シャッフルの乱数シード（再現性）
            data_fraction: train episode をこの割合に間引く（学習曲線用。
                           **val は常に全量**なので曲線の比較が公平になる）
        Returns:
            (train_idx, val_idx): サンプル index の配列
        """
        rng = np.random.default_rng(seed)
        eps = np.array(self.episodes)
        rng.shuffle(eps)
        n_val = max(1, int(len(eps) * val_frac))
        val_eps = set(eps[:n_val].tolist())
        train_eps = eps[n_val:]
        if data_fraction < 1.0:
            keep = max(1, int(len(train_eps) * data_fraction))
            train_eps = train_eps[:keep]
        train_eps = set(train_eps.tolist())
        ep = self.episode_id
        train_idx = np.where(np.isin(ep, list(train_eps)))[0]
        val_idx = np.where(np.isin(ep, list(val_eps)))[0]
        return train_idx, val_idx


class PTCGDataset(Dataset):
    """PTCGFeatureData の部分集合（train or val）を torch Dataset として見せる。"""

    def __init__(self, data: PTCGFeatureData, indices: np.ndarray):
        self.data = data
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        return self.data.sample(int(self.indices[i]))


def collate(samples: list[dict]) -> dict[str, torch.Tensor]:
    """可変長サンプル列をパディングして1バッチにする（model.forward の入力契約）。

    - トークン列: バッチ内最大長 Tmax に 0 パディング（tok_type=0 は PAD、
      embedding も padding_idx=0 でゼロになる）
    - label: バッチ内最大候補数 Kmax の multi-hot に展開
    """
    B = len(samples)
    Tmax = max(len(s["tok_type"]) for s in samples)
    Kmax = max(s["opt_count"] for s in samples)
    NF = samples[0]["numeric"].shape[1]

    def zeros(dtype):
        return torch.zeros((B, Tmax), dtype=dtype)

    out = {
        "tok_type": zeros(torch.long), "card_id": zeros(torch.long),
        "aux_card_id": zeros(torch.long), "attack_id": zeros(torch.long),
        "opt_type": zeros(torch.long), "area": zeros(torch.long),
        "numeric": torch.zeros((B, Tmax, NF)),
        "tok_mask": zeros(torch.bool), "opt_mask": zeros(torch.bool),
        "sel_type": torch.zeros(B, dtype=torch.long),
        "sel_ctx": torch.zeros(B, dtype=torch.long),
        "label_mh": torch.zeros((B, Kmax)),
        "won": torch.zeros(B), "turn": torch.zeros(B, dtype=torch.long),
        "n_options": torch.zeros(B, dtype=torch.long),
        "is_forced": torch.zeros(B, dtype=torch.bool),
    }
    for b, s in enumerate(samples):
        T = len(s["tok_type"])
        K = s["opt_count"]
        for k in PTCGFeatureData.TOKEN_FIELDS:
            out[k][b, :T] = torch.from_numpy(s[k].astype(np.int64))
        out["numeric"][b, :T] = torch.from_numpy(s["numeric"])
        out["tok_mask"][b, :T] = True
        out["opt_mask"][b, T - K: T] = True   # 候補手は末尾 K 個（featurize の契約）
        out["sel_type"][b] = s["sel_type"]
        out["sel_ctx"][b] = s["sel_ctx"]
        for lab in s["label"]:
            out["label_mh"][b, int(lab)] = 1.0
        out["won"][b] = s["won"]
        out["turn"][b] = s["turn"]
        out["n_options"][b] = s["n_options"]
        out["is_forced"][b] = s["is_forced"]
    return out
