#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""arena.py — 2つの提出dir（weights.npz 同梱）を公式エンジンで直接対戦させ勝率を出す。

学習時の検証指標（教師一致率など）は実戦の強さを測れないため、
モデルの採用判断はこの直接対戦の勝率で行う。

実装メモ:
  - kaggle_environments を経由せず cg.game の battle_start/battle_select を直接ループ
    （visualize 経路のメモリリークを丸ごと回避する）
  - それでも libcg 本体に ≈0.6MB/試合のリークはあるが、評価規模(数百試合)では無害
  - 席順バイアス対策で seat を1試合ごとに交代
  - 各エージェントは提出dirと同じ NumpyPolicy（=提出物と完全同一の推論経路）

使い方（リポジトリ直下から。エンジンが要るので Docker）:
  docker run --rm --platform linux/amd64 -v "$PWD:/work" -w /work \\
    -e PYTHONPATH=/work/engine <エンジン同梱の公式イメージ> \\
    python Imitation_Learning/src/eval/arena.py \\
      --agent-a <mine> --agent-b <opponent> --games 1200
  # agent dir は submissions/ 配下が規約。裸の名前でも submissions/ を自動補完する
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# 🔒 BLAS のスレッド数を 1 に固定（numpy を import する前に設定する必要がある）。
#   ゲート/アリーナは 1プロセス=1試合の**プロセス並列**なので、各プロセスの中で
#   さらに BLAS がスレッドを開くと CPU の奪い合いになる。spawn の子プロセスは
#   親の environ を継承するので、ここで設定すればワーカー側にも効く。
#   （2026-08-02: 11ワーカー × 各125%CPU でクォータ13を食い合い、ゲートが異常に遅かった）
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

# NumpyPolicy（提出と同じ推論器）を IL の src/submit から import
_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
_IL_SUBMIT = os.path.join(_ROOT, "Imitation_Learning", "src", "submit")
for p in (_IL_SUBMIT,):
    if p not in sys.path:
        sys.path.insert(0, p)

from infer_numpy import NumpyPolicy  # noqa: E402
from cg.game import battle_start, battle_select, battle_finish  # noqa: E402
# 公開Notebook 等の素の main.py も相手にできるようにする（weights.npz を持たない dir）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from external_agent import ExternalAgent, is_external  # noqa: E402


class Agent:
    """提出dir（main.py 相当の推論 + 安全網）を1オブジェクトに束ねる。"""

    def __init__(self, agent_dir: str):
        d = os.path.join(_ROOT, agent_dir) if not os.path.isabs(agent_dir) else agent_dir
        if not os.path.isdir(d):
            # 提出dirは submissions/ 配下が規約。裸の名前なら補完する
            alt = os.path.join(_ROOT, "submissions", os.path.basename(agent_dir.rstrip("/")))
            if os.path.isdir(alt):
                d = alt
        self.name = os.path.basename(d.rstrip("/"))
        with open(os.path.join(d, "deck.csv"), encoding="utf-8") as f:
            self.deck = [int(x) for x in f if x.strip()][:60]
        if len(self.deck) != 60:                    # 読み込み時点で落とす（battle_start まで持ち越さない）
            raise ValueError(f"{self.name}: deck が {len(self.deck)} 枚（60 でない）: {d}/deck.csv")
        self.policy = NumpyPolicy(os.path.join(d, "weights.npz"),
                                  os.path.join(d, "static"))

    def act(self, obs: dict, my_index: int) -> list[int]:
        sel = obs.get("select") or {}
        options = sel.get("option") or []
        try:
            picks = self.policy.predict(obs, my_index)
            n = len(options)
            mn = sel.get("minCount")
            if (picks and all(0 <= i < n for i in picks)
                    and (mn is None or len(picks) >= mn)):
                return picks
        except Exception:  # noqa: BLE001  (提出 main.py と同じ安全網)
            pass
        k = min(sel.get("maxCount") or 1, len(options))
        return list(range(k))


def build_agent(agent_dir: str, overrides: dict | None = None,
                ensemble: list | None = None, ens_mode: str = "prob"):
    """weights.npz があれば Agent（自作）、無ければ ExternalAgent（公開Notebook等）。

    どちらも name / deck / act(obs, my_index) を持つので play_one からは同じに見える。
    ensemble に weights npz を複数渡すと EnsembleAgent（出力平均）になる。
    """
    if ensemble:
        from ensemble_agent import EnsembleAgent           # noqa: E402
        return EnsembleAgent(agent_dir, ensemble, mode=ens_mode)
    if is_external(agent_dir):
        return ExternalAgent(agent_dir, overrides=overrides)
    return Agent(agent_dir)


def play_one(a, b, a_seat: int, max_decisions: int = 1200) -> int:
    """1試合。Returns: 勝者 result (0/1 = seat index, 2 = draw)。

    🔒 max_decisions は**必須の安全弁**。同じデッキ同士のミラー戦は稀に膠着して
    数千決定まで暴走し、実質無限ループになる（実際に評価が止まった実例がある）。
    超えたら引き分け扱いで打ち切る。通常の試合は ~85決定なので、
    正常な対戦には一切影響しない。
    """
    decks = [a.deck, b.deck] if a_seat == 0 else [b.deck, a.deck]
    obs, sd = battle_start(decks[0], decks[1])
    if sd.errorPlayer >= 0:
        raise ValueError(f"deck error: player={sd.errorPlayer} type={sd.errorType}")
    agents = {a_seat: a, 1 - a_seat: b}
    step = 0
    try:
        while obs["current"]["result"] < 0:
            if step >= max_decisions:
                return 2                      # 膠着 → 引き分け扱いで打ち切り
            me = obs["current"]["yourIndex"]
            obs = battle_select(agents[me].act(obs, me))
            step += 1
        return obs["current"]["result"]
    finally:
        battle_finish()


def _h2h_shard(task: tuple) -> dict:
    """ワーカー1タスク: 指定された game index 群だけを対戦して集計を返す。

    ★ワーカー内でエージェントを作り直す（NumpyPolicy とエンジンはプロセス毎に持つ）。
      席は**グローバルな game index の偶奇**で決めるので、分割しても席交代は保たれる。
    """
    dir_a, dir_b, idxs = task[0], task[1], task[2]
    sopt = task[3] if len(task) > 3 else {}      # a 側の探索設定（無ければ通常エージェント）
    a = build_agent(dir_a, **sopt)
    b = build_agent(dir_b)
    out = {"a": 0, "b": 0, "draw": 0, "seat": [0, 0], "n": 0}
    for g in idxs:
        a_seat = g % 2
        res = play_one(a, b, a_seat)
        if res == 2:
            out["draw"] += 1
        else:
            out["a" if res == a_seat else "b"] += 1
            out["seat"][res] += 1
        out["n"] += 1
    return out


def h2h_parallel(dir_a: str, dir_b: str, games: int, workers: int,
                 chunk: int, recycle: int, log_every: int,
                 sopt: dict | None = None) -> dict:
    """2者対戦を複数プロセスで分担する（直列版と統計的に等価）。

    エンジンのリーク対策として maxtasksperchild でワーカーを定期的に作り直す。
    sopt は a 側に渡す追加設定（build_agent の ensemble / ens_mode）。
    """
    import multiprocessing as mp
    tasks = [(dir_a, dir_b, list(range(s, min(s + chunk, games))), sopt or {})
             for s in range(0, games, chunk)]
    agg = {"a": 0, "b": 0, "draw": 0, "seat": [0, 0], "n": 0}
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers, maxtasksperchild=recycle) as pool:
        for r in pool.imap_unordered(_h2h_shard, tasks):
            for k in ("a", "b", "draw", "n"):
                agg[k] += r[k]
            agg["seat"][0] += r["seat"][0]
            agg["seat"][1] += r["seat"][1]
            if agg["n"] % log_every < chunk:
                el = time.time() - t0
                print(f"  [{agg['n']}/{games}] a={agg['a']} b={agg['b']} "
                      f"draw={agg['draw']} ({el/max(1, agg['n']):.2f}s/game "
                      f"/ {agg['n']/max(el, 1e-9):.2f} games/s)", flush=True)
    return agg


def eval_pool(agent_dir: str, opponent_dirs: list[str], games_each: int,
              meta_weights: dict | None, log_every: int,
              workers: int = 1, chunk: int = 5, recycle: int = 40) -> dict:
    """1エージェントを複数相手に対戦させ、相手別勝率＋メタ加重集計を返す（Stage 1-2）。

    最終評価用: 自レート帯のメタ分布(meta_weights)で相手別勝率を加重平均する。
    各対戦は seat 交代。meta_weights 未指定なら一様加重。
    ★workers>=2 なら相手ごとに h2h_parallel で並列化（2026-08-15。d256 は 1,000試合の
      プール評価が直列だと 25〜40分かかり、学習より評価が長くなるため）。
    """
    agent_name = os.path.basename(agent_dir.rstrip("/"))
    per = {}
    if workers >= 2:
        for d in opponent_dirs:
            oname = os.path.basename(d.rstrip("/"))
            print(f"  [pool] {agent_name} vs {oname} ({games_each}g, "
                  f"workers={workers})", flush=True)
            agg = h2h_parallel(agent_dir, d, games_each, workers, chunk,
                               recycle, log_every)
            dec = agg["a"] + agg["b"]
            per[oname] = {"winrate": agg["a"] / dec if dec else float("nan"),
                          "wins": {"a": agg["a"], "o": agg["b"], "draw": agg["draw"]},
                          "games": games_each}
    else:
        a = build_agent(agent_dir)
        agent_name = a.name
        opps = [build_agent(d) for d in opponent_dirs]
        for opp in opps:
            wins = {"a": 0, "o": 0, "draw": 0}
            for g in range(games_each):
                a_seat = g % 2
                res = play_one(a, opp, a_seat)
                if res == 2:
                    wins["draw"] += 1
                elif res == a_seat:
                    wins["a"] += 1
                else:
                    wins["o"] += 1
                if (g + 1) % log_every == 0:
                    print(f"  [{a.name} vs {opp.name} {g+1}/{games_each}] "
                          f"{wins['a']}/{wins['o']}/{wins['draw']}", flush=True)
            dec = wins["a"] + wins["o"]
            per[opp.name] = {"winrate": wins["a"] / dec if dec else float("nan"),
                             "wins": wins, "games": games_each}
    # メタ加重集計
    mw = {n: (meta_weights or {}).get(n, 1.0) for n in per}
    tot = sum(mw.values()) or 1.0
    weighted = sum(per[n]["winrate"] * mw[n] / tot for n in per)
    return {"agent": agent_name, "per_opponent": per,
            "meta_weights": {n: round(mw[n] / tot, 4) for n in per},
            "weighted_winrate": weighted,
            "unweighted_winrate": sum(per[n]["winrate"] for n in per) / len(per) if per else float("nan")}


def main() -> int:
    ap = argparse.ArgumentParser(description="エージェント対戦評価（2者 or メタ加重プール）")
    ap.add_argument("--agent-a", required=True, help="提出dir（deck.csv/weights.npz/static）")
    ap.add_argument("--agent-b", help="2者対戦の相手（--pool 指定時は無視）")
    ap.add_argument("--pool", nargs="*", default=None,
                    help="メタ加重プール評価: agent-a を複数相手に対戦させる（Stage 1-2）")
    ap.add_argument("--meta-weights", default=None, help="MYSO メタ分布 JSON（相手加重）")
    ap.add_argument("--games", type=int, default=100, help="2者=総試合数 / プール=相手ごと試合数")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--out", default=None, help="結果 JSON の書き出し先（任意）")
    ap.add_argument("--workers", type=int, default=1,
                    help="2者対戦を並列化する（既定 1 = 従来の直列）。ゲートは 6 程度を推奨")
    ap.add_argument("--chunk", type=int, default=10, help="1タスクの試合数（並列時）")
    ap.add_argument("--recycle", type=int, default=10,
                    help="ワーカーを作り直す間隔（タスク数。libcg のリーク封じ込め）。"
                         "1,200試合の実測で 1試合あたりが 0.70→4.18s に悪化したため 40→10 に短縮")
    ap.add_argument("--ensemble", nargs="*", default=None,
                    help="★出力平均アンサンブル: 合成する weights npz を並べる"
                         "（deck.csv / static は --agent-a のものを使う）")
    ap.add_argument("--ens-mode", choices=("prob", "logit"), default="prob",
                    help="prob=確率の算術平均（既定） / logit=logits の平均")
    args = ap.parse_args()
    if args.ensemble:
        sopt = dict(ensemble=args.ensemble, ens_mode=args.ens_mode)
    else:
        sopt = {}

    # --- メタ加重プール評価（Stage 1-2）---
    if args.pool:
        mw = json.load(open(args.meta_weights)) if args.meta_weights else None
        print(f"[arena/pool] {os.path.basename(args.agent_a.rstrip('/'))} vs "
              f"{len(args.pool)} opponents, {args.games} games each", flush=True)
        r = eval_pool(args.agent_a, args.pool, args.games, mw, args.log_every,
                      workers=args.workers, chunk=args.chunk, recycle=args.recycle)
        print(f"\n[pool result] {r['agent']}")
        for n, d in r["per_opponent"].items():
            print(f"  vs {n:<20} wr={d['winrate']:.3f} "
                  f"({d['wins']['a']}/{d['wins']['o']}/{d['wins']['draw']}) "
                  f"weight={r['meta_weights'][n]}")
        print(f"  → メタ加重勝率 = {r['weighted_winrate']:.3f}  "
              f"(単純平均 {r['unweighted_winrate']:.3f})")
        if args.out:
            json.dump(r, open(args.out, "w"), indent=1)
        return 0

    # --- 2者ヘッドツーヘッド（従来）---
    if not args.agent_b:
        ap.error("--agent-b か --pool のどちらかが必要")
    # --- 並列（--workers >= 2）: ワーカー内でエージェントを作るので親では名前解決だけ ---
    if args.workers >= 2:
        na = os.path.basename(args.agent_a.rstrip("/"))
        nb = os.path.basename(args.agent_b.rstrip("/"))
        print(f"[arena] {na} vs {nb}  games={args.games} "
              f"workers={args.workers} chunk={args.chunk}", flush=True)
        agg = h2h_parallel(args.agent_a, args.agent_b, args.games,
                           args.workers, args.chunk, args.recycle, args.log_every,
                           sopt=sopt)
        n_dec = agg["a"] + agg["b"]
        wr = agg["a"] / n_dec if n_dec else float("nan")
        print(f"\n[result] {na}: {agg['a']}勝 / {nb}: {agg['b']}勝 / draw: {agg['draw']}")
        print(f"         win-rate({na}) = {wr:.3f}  "
              f"[seat0勝={agg['seat'][0]} seat1勝={agg['seat'][1]}]")
        if n_dec:
            se = (wr * (1 - wr) / n_dec) ** 0.5
            print(f"         95%CI ≈ [{wr-1.96*se:.3f}, {wr+1.96*se:.3f}]")
        if args.out:
            json.dump({"agent_a": na, "agent_b": nb, "winrate": wr,
                       "ci95": (1.96 * se) if n_dec else None,
                       "wins": {na: agg["a"], nb: agg["b"], "draw": agg["draw"]},
                       "seat_wins": agg["seat"], "games": args.games,
                       "workers": args.workers}, open(args.out, "w"), indent=1)
        return 0

    a, b = build_agent(args.agent_a), build_agent(args.agent_b)
    print(f"[arena] {a.name} vs {b.name}  games={args.games}", flush=True)

    wins = {a.name: 0, b.name: 0, "draw": 0}
    seat_wins = [0, 0]  # 先手席(0)/後手席(1)の勝ち数（seatバイアス監視）
    t0 = time.time()
    for g in range(args.games):
        a_seat = g % 2  # 席交代
        res = play_one(a, b, a_seat)
        if res == 2:
            wins["draw"] += 1
        else:
            winner = a if res == a_seat else b
            wins[winner.name] += 1
            seat_wins[res] += 1
        if (g + 1) % args.log_every == 0:
            el = time.time() - t0
            print(f"  [{g+1}/{args.games}] {a.name}={wins[a.name]} "
                  f"{b.name}={wins[b.name]} draw={wins['draw']} "
                  f"({el/(g+1):.1f}s/game)", flush=True)

    n_dec = wins[a.name] + wins[b.name]
    wr = wins[a.name] / n_dec if n_dec else float("nan")
    print(f"\n[result] {a.name}: {wins[a.name]}勝 / {b.name}: {wins[b.name]}勝 "
          f"/ draw: {wins['draw']}")
    print(f"         win-rate({a.name}) = {wr:.3f}  "
          f"[seat0勝={seat_wins[0]} seat1勝={seat_wins[1]}]")
    # 95%CI (正規近似) — 100試合なら ±0.10 程度が目安
    if n_dec:
        se = (wr * (1 - wr) / n_dec) ** 0.5
        print(f"         95%CI ≈ [{wr-1.96*se:.3f}, {wr+1.96*se:.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
