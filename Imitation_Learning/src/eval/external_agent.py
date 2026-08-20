#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""external_agent.py — 公開Notebook等の「素の main.py」をそのまま対戦相手として使うアダプタ。

自作エージェントは weights.npz + NumpyPolicy 前提（arena.py の Agent）だが、
公開Notebook のエージェントは `def agent(obs_dict) -> list[int]` だけを持つ素の Python。
このアダプタが両者の差を吸収し、arena から同じ形で呼べるようにする。

対象ディレクトリの要件（`opponents/<名前>/`）:
  main.py    … `def agent(obs_dict) -> list[int]` を定義（初手は deck を返す規約）
  deck.csv   … 60枚のカードID（main.py が cwd から読むので必須）

実装上の注意:
  - main.py は**専用の名前空間に exec** する。モジュールレベルのグローバル状態
    （ターン記憶・診断カウンタ等）を持つ実装が多いため、1インスタンス=1名前空間にして
    自己対戦や複数同時対戦でも状態が混ざらないようにする。
  - exec 中は **cwd をインスタンス専用の一時コピーに切り替える**（`deck.csv` を相対パスで
    読み書きする実装、import 時に deck.csv を**書き出す**実装があるため）。
    🔒 **一時コピーは必須**（2026-08-02 に本番で踏んだ）: probabilistic_expectimax は
    `Path("deck.csv").write_text(...)` を import 時に実行する。30 ワーカーが同じ
    `opponents/<名前>/deck.csv` を同時に truncate → 書き込みするので、その瞬間に読んだ
    プロセスが短いファイルや NUL 埋めを掴み、`The deck must contain 60 cards.` /
    `invalid literal for int() with base 10: '\\x00...'` でロールアウトが落ちた。
    共有ディレクトリに第三者コードを書かせない、が対策。
  - `agent()` が例外を投げても対戦を止めない（合法手の先頭を返す安全網）。
    第三者コードなので、こちらの評価が落ちないことを優先する。
  - `overrides` で main.py のモジュール変数を上書きできる（探索時間の抑制など）。

使い方:
    from external_agent import ExternalAgent
    opp = ExternalAgent("opponents/probabilistic_expectimax",
                        overrides={"SEARCH_TIME_BUDGET": 0.2})
    picks = opp.act(obs, obs["current"]["yourIndex"])
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
import types

_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))


def resolve_dir(agent_dir: str) -> str:
    """`opponents/xxx` / `xxx` / 絶対パス のどれでも受ける。"""
    if os.path.isabs(agent_dir):
        return agent_dir
    cand = [os.path.join(_ROOT, agent_dir),
            os.path.join(_ROOT, "opponents", os.path.basename(agent_dir.rstrip("/"))),
            os.path.join(_ROOT, "submissions", os.path.basename(agent_dir.rstrip("/")))]
    for c in cand:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError(f"agent dir が見つからない: {agent_dir}")


def is_external(agent_dir: str) -> bool:
    """weights.npz を持たず main.py を持つなら外部エージェント扱い。"""
    d = resolve_dir(agent_dir)
    return (not os.path.exists(os.path.join(d, "weights.npz"))
            and os.path.exists(os.path.join(d, "main.py")))


class ExternalAgent:
    """arena.py の Agent と同じ形（name / deck / act）で外部 main.py を包む。"""

    def __init__(self, agent_dir: str, overrides: dict | None = None,
                 name: str | None = None):
        self.dir = resolve_dir(agent_dir)
        self.name = name or os.path.basename(self.dir.rstrip("/"))

        self.n_calls = 0
        self.n_errors = 0
        self.total_sec = 0.0
        self.max_sec = 0.0

        # --- 🔒 インスタンス専用の作業コピーを作る（第三者コードに共有dirを書かせない）---
        #   deck.csv を import 時に書き出す実装があり、並列ワーカーが同じファイルを
        #   同時に truncate すると読み側が壊れる（docstring 参照）。数十KBなので安い。
        self._work = tempfile.TemporaryDirectory(prefix=f"extagent_{self.name}_")
        work = self._work.name
        shutil.copytree(self.dir, work, dirs_exist_ok=True)

        # --- main.py を専用名前空間に読み込む（cwd を作業コピーに固定して実行）---
        path = os.path.join(self.dir, "main.py")
        mod = types.ModuleType(f"ext_{self.name}_{id(self)}")
        mod.__file__ = path
        with open(path, encoding="utf-8") as f:
            code = compile(f.read(), path, "exec")
        cwd = os.getcwd()
        try:
            os.chdir(work)
            exec(code, mod.__dict__)  # noqa: S102  第三者コードの実行は承知の上
        finally:
            os.chdir(cwd)

        # デッキは **exec 後の作業コピー**から読む（main.py が書き換えた結果が正）
        deck_path = os.path.join(work, "deck.csv")
        if not os.path.exists(deck_path):
            deck_path = os.path.join(self.dir, "deck.csv")
        with open(deck_path, encoding="utf-8") as f:
            self.deck = [int(x) for x in f if x.strip()][:60]
        # ★ここで落とす。battle_start まで持ち越すと原因がロールアウト深部に出て追いにくい
        if len(self.deck) != 60:
            raise ValueError(f"{self.name}: deck が {len(self.deck)} 枚（60 でない）: {deck_path}")
        for k, v in (overrides or {}).items():
            if k not in mod.__dict__:
                raise KeyError(f"{self.name}: 上書き対象 {k} が main.py に無い")
            mod.__dict__[k] = v
        self.overrides = dict(overrides or {})
        self.module = mod
        fn = mod.__dict__.get("agent")
        if not callable(fn):
            raise AttributeError(f"{path} に呼び出し可能な agent が無い")
        self._fn = fn

    # arena.Agent と同じ署名に合わせる
    # （record / step は学習側の軌跡記録用。外部エージェントは学習しないので無視する）
    def act(self, obs: dict, my_index: int, record: list | None = None,
            step: int = 0) -> list[int]:
        sel = obs.get("select") or {}
        options = sel.get("option") or []
        t0 = time.perf_counter()
        try:
            picks = self._fn(obs)
            dt = time.perf_counter() - t0
            self.n_calls += 1
            self.total_sec += dt
            self.max_sec = max(self.max_sec, dt)
            if obs.get("select") is None:      # 初手のデッキ提出
                return picks
            n = len(options)
            mn = sel.get("minCount")
            if (picks and all(isinstance(i, int) and 0 <= i < n for i in picks)
                    and (mn is None or len(picks) >= mn)):
                return picks
        except Exception:  # noqa: BLE001  第三者コードが落ちても対戦は続ける
            self.n_errors += 1
        k = min(sel.get("maxCount") or 1, len(options))
        return list(range(k))

    def stats(self) -> dict:
        return {"name": self.name, "calls": self.n_calls, "errors": self.n_errors,
                "mean_sec": (self.total_sec / self.n_calls) if self.n_calls else 0.0,
                "max_sec": self.max_sec, "overrides": self.overrides}


def make_agent(agent_dir: str, overrides: dict | None = None):
    """weights.npz があれば arena.Agent、無ければ ExternalAgent を返すファクトリ。"""
    if is_external(agent_dir):
        return ExternalAgent(agent_dir, overrides=overrides)
    from arena import Agent  # 循環 import を避けるため遅延 import
    return Agent(agent_dir)


def main() -> int:
    """単体確認: 外部エージェント同士（または自作相手）で数試合まわして統計を出す。"""
    import argparse
    ap = argparse.ArgumentParser(description="外部エージェントの動作確認")
    ap.add_argument("--agent-a", required=True)
    ap.add_argument("--agent-b", required=True)
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--search-budget", type=float, default=None,
                    help="SEARCH_TIME_BUDGET を持つ実装ならこの値に上書きする")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from arena import play_one  # 同じ対戦ループを使う

    ov = {"SEARCH_TIME_BUDGET": args.search_budget} if args.search_budget else None
    def build(d):
        try:
            return make_agent(d, overrides=ov)
        except KeyError:            # その実装に無い変数だった
            return make_agent(d)
    a, b = build(args.agent_a), build(args.agent_b)
    print(f"[check] {a.name} vs {b.name}  games={args.games}", flush=True)
    wins = {a.name: 0, b.name: 0, "draw": 0}
    t0 = time.time()
    for g in range(args.games):
        r = play_one(a, b, g % 2)
        seat_of_a = g % 2
        if r == 2:
            wins["draw"] += 1
        else:
            wins[a.name if r == seat_of_a else b.name] += 1
        print(f"  game{g + 1}: result={r} ({time.time() - t0:.1f}s 累計)", flush=True)
    print(f"[result] {wins}")
    for ag in (a, b):
        if isinstance(ag, ExternalAgent):
            s = ag.stats()
            print(f"[stats] {s['name']}: calls={s['calls']} errors={s['errors']} "
                  f"mean={s['mean_sec'] * 1000:.1f}ms max={s['max_sec'] * 1000:.1f}ms "
                  f"overrides={s['overrides']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
