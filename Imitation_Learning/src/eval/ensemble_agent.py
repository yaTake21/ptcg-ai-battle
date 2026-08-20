#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ensemble_agent.py — 同一条件・seed 違いの N 本を「出力平均」で合成する（arena 互換）。

■ ★重み平均ではなく出力平均
  seed が違うと初期値が違う＝別の盆地に落ちるので、**重みを平均すると壊れる**
  （実測: seed 違いの重み平均は効果が無く、同一学習経路上の補間である WiSE-FT だけが効いた。
    mode connectivity が満たされるかで明暗が分かれる）。
  ここでやるのは推論時の合成:

      p_ens(候補 i) = (1/N) Σ_k softmax(logits_k)[i]      ← 確率の算術平均（既定）
      logits 平均は確率の幾何平均に相当し、1本が極端な値を出すと引きずられる。
      --ens-mode logit で切り替えられるので、どちらが良いかは実測で決める。

■ 前提: メンバーの実力が揃っていること
  08-08 の実測では実力差18pt を混ぜると悪化した（0.8159 → 0.7839）。
  seed だけ変えたメンバーは実力が揃うので、この条件を満たす。

■ 使い方（arena から）
  --ensemble <npz1> <npz2> ...   （--agent-a の deck.csv / static を土台に使う）
"""
from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
_IL_SUBMIT = os.path.join(_ROOT, "Imitation_Learning", "src", "submit")
if _IL_SUBMIT not in sys.path:
    sys.path.insert(0, _IL_SUBMIT)

from infer_numpy import NumpyPolicy, resolve_option_card_id  # noqa: E402


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


class EnsembleAgent:
    """arena の Agent と同じ顔（name / deck / act）を持つ、N本の出力平均エージェント。"""

    def __init__(self, agent_dir: str, weights: list[str], mode: str = "prob"):
        d = agent_dir if os.path.isabs(agent_dir) else os.path.join(_ROOT, agent_dir)
        if not os.path.isdir(d):
            alt = os.path.join(_ROOT, "submissions", os.path.basename(agent_dir.rstrip("/")))
            if os.path.isdir(alt):
                d = alt
        self.name = f"{os.path.basename(d.rstrip('/'))}+ens{len(weights)}"
        with open(os.path.join(d, "deck.csv"), encoding="utf-8") as f:
            self.deck = [int(x) for x in f if x.strip()][:60]
        if len(self.deck) != 60:
            raise ValueError(f"{self.name}: deck が {len(self.deck)} 枚")
        static = os.path.join(d, "static")
        self.members = []
        for w in weights:
            wp = w if os.path.isabs(w) else os.path.join(_ROOT, w)
            if not os.path.exists(wp):
                raise FileNotFoundError(wp)
            self.members.append(NumpyPolicy(wp, static))
        self.mode = mode
        if mode not in ("prob", "logit"):
            raise ValueError(f"未知の合成方式: {mode}")

    def act(self, obs: dict, my_index: int) -> list[int]:
        sel = obs.get("select") or {}
        state = obs.get("current")
        options = sel.get("option") or []
        mx = sel.get("maxCount") or 1
        mn = sel.get("minCount")
        k = min(mx if mn is None else max(mn, mx), len(options))
        if not options or state is None:
            return []
        try:
            enriched = []
            for o in options:
                oo = dict(o)
                oo["_card_id"] = resolve_option_card_id(state, sel, o, my_index)
                enriched.append(oo)
            sample = {"state": state, "select": sel,
                      "player_index": my_index, "options": enriched}
            acc = None
            for pol in self.members:
                lg, _ = pol.forward_sample(sample)
                v = _softmax(np.asarray(lg, dtype=np.float64)) if self.mode == "prob" \
                    else np.asarray(lg, dtype=np.float64)
                acc = v if acc is None else acc + v
            score = acc / len(self.members)
            return [int(i) for i in np.argsort(-score)[:k]]
        except Exception:                                    # noqa: BLE001  提出と同じ安全網
            return list(range(k))
