#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""featurize.py — IL教師ペア(JSONL) → 学習用テンソル(.npz) 変換器。

トークン化仕様（README §3.1）の実装。**純numpy・cg非依存**なので
Mac / Colab / Kaggle のどこでも動く（エンジン libcg.so は一切不要）。

■ 何をするか
  収集パイプラインが出力した教師ペア JSONL（1行 = 1意思決定）を読み、各サンプルを
  「エンティティトークン列」に変換する:

    [GLB] [自分ACT] [自分BE...] [手札...] [自分捨札...] [相手ACT] [相手BE...]
    [相手捨札...] [STADIUM] [LOOKING...] | [OPT_1] ... [OPT_K]

  - 状態トークンが先、option トークンが**必ず末尾 K 個**という順序を保証する
    （モデル側はこの規約を前提に「末尾 K 個の出力」を policy head に渡す）。
  - 可変長はそのまま保存（切り捨てなし）。パディングは学習時の collate で行う。

■ トークンの中身（各トークンは以下のフィールドの組）
  tok_type   : トークン種別 id（TOKEN_* 定数。位置エンベディングの素）
  card_id    : 参照カード id（無ければ 0。カード埋め込み+静的属性の素）
  aux_card_id: 補助カード id（場ポケモンの進化元。無ければ 0）
  attack_id  : ワザ id（type=ATTACK の option のみ。無ければ 0）
  opt_type   : OptionType+1（option トークンのみ。状態トークンは 0）
  area       : AreaType（option トークンのみ。無ければ 0）
  numeric    : 実数特徴ベクトル（NF次元。HP割合・付きエネ・GLBのターン情報等。
               トークン種別ごとに使うスロットが違い、残りは 0。下の NUM_* 参照）

■ 捨札の表現（設計 §3.2 の「要約」の実装）
  捨札はユニークカードごとに1トークン（numeric の NUM_COUNT に同名枚数/4）。
  Archaludon の Assemble Alloy のような「捨札のエネ枚数」が正確に伝わる。

■ 静的属性テーブル
  data/static/{cards.json, attacks.json}（dump_static_tables.py の出力）から
  card_static [n_card+1, S_CARD] / attack_static [n_atk+1, S_ATK] を構築して
  npz に同梱する。モデルはこれを凍結バッファとして参照する
  （id埋め込み=個体の記憶、静的属性=未知カードへの汎化。設計 §3.1）。

■ 出力（単一 .npz、可変長は flat 配列 + per-sample カウントで表現。pickle不使用）
  tok_type/card_id/aux_card_id/attack_id/opt_type/area : flat 配列
  numeric      : [総トークン数, NF]
  tok_count    : [N] サンプルごとのトークン数
  opt_count    : [N] サンプルごとの option 数 K（トークン列の末尾 K 個が option）
  label_flat / label_count : 教師ラベル（options への index、可変個）
  won/turn/sel_type/sel_ctx/n_options/is_forced : [N] メタ（損失・評価用）
  episode_id   : [N] 文字列（episode 単位の train/val 分割用）
  step/player_index/reward : [N] Phase 2a オフラインRL 用
                 （(episode_id, player_index) で軌跡復元・step順で隣接遷移、reward は終端 ±1）
  card_static / attack_static : 静的属性テーブル
  config_json  : 特徴量レイアウトの定義（モデル側が読む）

使い方:
  # src/learn/ から
  python3 featurize.py ../../data/samples/<samples>.jsonl \\
      -o ../../data/features/<features>.npz
  python3 featurize.py <in.jsonl> -o <out.npz> --limit 100   # スモークテスト
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

# ---------------------------------------------------------------------------
# 定数定義（モデル側と共有する契約。config_json にも書き出す）
# ---------------------------------------------------------------------------

# トークン種別 id（位置エンベディングの語彙）
TOKEN_PAD = 0
TOKEN_GLB = 1          # グローバル（ターン情報・select 情報・各種カウント）
TOKEN_MY_ACTIVE = 2    # 自分のバトル場ポケモン
TOKEN_MY_BENCH = 3     # 自分のベンチ
TOKEN_MY_HAND = 4      # 自分の手札（1枚1トークン）
TOKEN_MY_DISCARD = 5   # 自分の捨札（ユニークカードごと、枚数は numeric）
TOKEN_OPP_ACTIVE = 6
TOKEN_OPP_BENCH = 7
TOKEN_OPP_DISCARD = 8
TOKEN_STADIUM = 9
TOKEN_LOOKING = 10     # 効果解決中に見えているカード（山札サーチ等）
TOKEN_OPTION = 11      # 候補手（必ず列の末尾 K 個）
N_TOKEN_TYPES = 12

# numeric 特徴のスロット（NF 次元ベクトルの何番目に何を入れるか）
NUM_HP_FRAC = 0        # 残りHP / maxHp（場ポケモン）
NUM_DMG_FRAC = 1       # 受けているダメージ / 330
NUM_MAXHP_FRAC = 2     # maxHp / 380
NUM_ENERGY_MH = 3      # 3..13: 付きエネルギーのタイプ別枚数 / 4（11タイプ）
NUM_N_ENERGY = 14      # 付きエネ合計 / 5
NUM_TOOL = 15          # どうぐ有無
NUM_APPEAR = 16        # このターンに場に出たか
NUM_STATUS = 17        # 17..21: asleep/burned/confused/paralyzed/poisoned（active のみ）
NUM_IS_MINE = 22       # 自分側のトークンか
NUM_COUNT = 23         # 汎用カウント（捨札トークン: 同名枚数/4）
NUM_POS = 24           # エリア内 index / 5
NUM_TURN = 25          # GLB: turn / 30
NUM_FIRST_IS_ME = 26   # GLB: 先攻が自分か
NUM_ENERGY_ATTACHED = 27  # GLB: このターンエネを付けたか
NUM_SUPPORTER = 28     # GLB: サポート使用済みか
NUM_STADIUM_PLAYED = 29
NUM_RETREATED = 30
NUM_ACTION_CNT = 31    # GLB: turnActionCount / 10
NUM_MY_DECK = 32       # GLB: 自分の山札枚数 / 60
NUM_OPP_DECK = 33
NUM_MY_HAND = 34       # GLB: 手札枚数 / 20
NUM_OPP_HAND = 35
NUM_MY_PRIZE = 36      # GLB: 自分サイド残 / 6
NUM_OPP_PRIZE = 37
NUM_MIN_COUNT = 38     # GLB: select.minCount / 6
NUM_MAX_COUNT = 39     # GLB: select.maxCount / 6
NUM_REMAIN_DMG = 40    # GLB: remainDamageCounter / 10
NUM_REMAIN_ENE = 41    # GLB: remainEnergyCost / 5
NUM_OPT_INDEX = 42     # OPTION: 候補列内の位置 / 40
NUM_STADIUM_MINE = 43  # STADIUM: 自分が出したか
NF = 44

N_ENERGY_TYPES = 11    # EnergyType 0..10 (COLORLESS..RAINBOW)

# OptionType（cg/api.py と同値。option の型 id。+1 して 0 を「なし」に予約）
OPT_TYPE_VOCAB = 18    # 0=なし, 1..17 = OptionType(0..16)+1
AREA_VOCAB = 13        # 0=なし, 1..12 = AreaType
SEL_TYPE_VOCAB = 12    # SelectType 0..10 (+予備)
SEL_CTX_VOCAB = 48     # SelectContext 0..35 程度 (+予備)

# 静的属性テーブルの次元（build_static_tables を参照）
S_CARD = 1 + 7 + 12 + 13 + 13 + 1 + 7 + 4   # = 58
S_ATK = 1 + 1 + 11 + 1                        # = 14


# ---------------------------------------------------------------------------
# 静的属性テーブル
# ---------------------------------------------------------------------------

def build_static_tables(static_dir: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """data/static/ の cards.json / attacks.json から静的属性テーブルを構築する。

    Returns:
        card_static  : [max_card_id+1, S_CARD]  行 = カード id（0行目はゼロ=「なし」）
        attack_static: [max_atk_id+1, S_ATK]    行 = ワザ id（同上）
        info         : 語彙サイズ等のメタ情報

    カード行の内訳（S_CARD=58）:
        [0]      hp / 380
        [1:8]    cardType one-hot（7種: 0..6）
        [8:20]   energyType one-hot（12枠: 0..10 + 予備）
        [20:33]  weakness one-hot（12枠）+ 弱点あり flag
        [33:46]  resistance one-hot（12枠）+ 抵抗あり flag
        [46]     retreatCost / 4
        [47:54]  basic/stage1/stage2/ex/megaEx/tera/aceSpec の7フラグ
        [54:58]  ワザ数/3, 最大ダメージ/350, 最小必要エネ数/5, 進化カードか
    ワザ行の内訳（S_ATK=14）:
        [0] damage/350, [1] 必要エネ数/5, [2:13] エネタイプ別必要数/4, [13] 効果テキストあり
    """
    cards = json.load(open(os.path.join(static_dir, "cards.json")))
    attacks = json.load(open(os.path.join(static_dir, "attacks.json")))

    max_cid = max(int(k) for k in cards)
    max_aid = max(int(k) for k in attacks)
    card_static = np.zeros((max_cid + 1, S_CARD), dtype=np.float32)
    attack_static = np.zeros((max_aid + 1, S_ATK), dtype=np.float32)

    for aid_s, a in attacks.items():
        aid = int(aid_s)
        v = attack_static[aid]
        v[0] = a["damage"] / 350.0
        ens = a.get("energies") or []
        v[1] = len(ens) / 5.0
        for e in ens:
            if 0 <= e < N_ENERGY_TYPES:
                v[2 + e] += 0.25          # タイプ別必要数 / 4
        v[13] = 1.0 if (a.get("text") or "").strip() else 0.0

    for cid_s, c in cards.items():
        cid = int(cid_s)
        v = card_static[cid]
        v[0] = (c.get("hp") or 0) / 380.0
        ct = c.get("cardType")
        if ct is not None and 0 <= ct < 7:
            v[1 + ct] = 1.0
        et = c.get("energyType")
        if et is not None and 0 <= et < 12:
            v[8 + et] = 1.0
        wk = c.get("weakness")
        if wk is not None and 0 <= wk < 12:
            v[20 + wk] = 1.0
            v[32] = 1.0
        rs = c.get("resistance")
        if rs is not None and 0 <= rs < 12:
            v[33 + rs] = 1.0
            v[45] = 1.0
        v[46] = (c.get("retreatCost") or 0) / 4.0
        for j, f in enumerate(("basic", "stage1", "stage2", "ex", "megaEx", "tera", "aceSpec")):
            v[47 + j] = 1.0 if c.get(f) else 0.0
        atk_ids = c.get("attacks") or []
        v[54] = len(atk_ids) / 3.0
        if atk_ids:
            v[55] = max(attack_static[a][0] for a in atk_ids if a <= max_aid)  # 最大dmg(正規化済)
            v[56] = min(attack_static[a][1] for a in atk_ids if a <= max_aid)  # 最小エネ数(正規化済)
        v[57] = 1.0 if c.get("evolvesFrom") else 0.0

    info = {"n_card": max_cid + 1, "n_attack": max_aid + 1}
    return card_static, attack_static, info


# ---------------------------------------------------------------------------
# 1サンプル → トークン列
# ---------------------------------------------------------------------------

class TokenBuffer:
    """1サンプル分のトークンを積むだけの小さなヘルパ。"""

    def __init__(self):
        self.tok_type: list[int] = []
        self.card_id: list[int] = []
        self.aux_card_id: list[int] = []
        self.attack_id: list[int] = []
        self.opt_type: list[int] = []
        self.area: list[int] = []
        self.numeric: list[np.ndarray] = []

    def add(self, tok_type: int, card_id: int = 0, aux_card_id: int = 0,
            attack_id: int = 0, opt_type: int = 0, area: int = 0,
            numeric: np.ndarray | None = None) -> None:
        self.tok_type.append(tok_type)
        self.card_id.append(card_id)
        self.aux_card_id.append(aux_card_id)
        self.attack_id.append(attack_id)
        self.opt_type.append(opt_type)
        self.area.append(area)
        self.numeric.append(numeric if numeric is not None else np.zeros(NF, np.float32))

    def __len__(self):
        return len(self.tok_type)


def _pokemon_numeric(p: dict, is_mine: bool, pos: int,
                     status_flags: list[bool] | None) -> np.ndarray:
    """場ポケモン1体の numeric ベクトルを作る。

    Args:
        p           : state.players[i].active/bench の要素（カード dict）
        is_mine     : 自分側か
        pos         : エリア内 index（ベンチ位置）
        status_flags: [asleep,burned,confused,paralyzed,poisoned]（active のみ。bench は None）
    """
    v = np.zeros(NF, np.float32)
    max_hp = p.get("maxHp") or 0
    hp = p.get("hp") or 0
    if max_hp > 0:
        v[NUM_HP_FRAC] = hp / max_hp
        v[NUM_DMG_FRAC] = (max_hp - hp) / 330.0
        v[NUM_MAXHP_FRAC] = max_hp / 380.0
    ens = p.get("energies") or []
    for e in ens:
        if 0 <= e < N_ENERGY_TYPES:
            v[NUM_ENERGY_MH + e] += 0.25
    v[NUM_N_ENERGY] = len(ens) / 5.0
    v[NUM_TOOL] = 1.0 if (p.get("tools") or []) else 0.0
    v[NUM_APPEAR] = 1.0 if p.get("appearThisTurn") else 0.0
    if status_flags is not None:
        for j, f in enumerate(status_flags):
            v[NUM_STATUS + j] = 1.0 if f else 0.0
    v[NUM_IS_MINE] = 1.0 if is_mine else 0.0
    v[NUM_POS] = pos / 5.0
    return v


def sample_to_tokens(d: dict) -> TokenBuffer:
    """JSONL 1行（1意思決定）をトークン列に変換する。

    トークン順序の契約: 状態トークン → option トークン（末尾 K 個）。
    モデル/データセット側は opt_count=K を使って末尾 K 個を policy head に渡す。
    """
    st = d["state"]
    sel = d["select"]
    me = d["player_index"]
    opp = 1 - me
    ps_me = st["players"][me]
    ps_opp = st["players"][opp]
    buf = TokenBuffer()

    # ---- [GLB] グローバルトークン --------------------------------------
    g = np.zeros(NF, np.float32)
    g[NUM_TURN] = (st.get("turn") or 0) / 30.0
    g[NUM_FIRST_IS_ME] = 1.0 if st.get("firstPlayer") == me else 0.0
    g[NUM_ENERGY_ATTACHED] = 1.0 if st.get("energyAttached") else 0.0
    g[NUM_SUPPORTER] = 1.0 if st.get("supporterPlayed") else 0.0
    g[NUM_STADIUM_PLAYED] = 1.0 if st.get("stadiumPlayed") else 0.0
    g[NUM_RETREATED] = 1.0 if st.get("retreated") else 0.0
    g[NUM_ACTION_CNT] = (st.get("turnActionCount") or 0) / 10.0
    g[NUM_MY_DECK] = (ps_me.get("deckCount") or 0) / 60.0
    g[NUM_OPP_DECK] = (ps_opp.get("deckCount") or 0) / 60.0
    g[NUM_MY_HAND] = (ps_me.get("handCount") or 0) / 20.0
    g[NUM_OPP_HAND] = (ps_opp.get("handCount") or 0) / 20.0
    g[NUM_MY_PRIZE] = len(ps_me.get("prize") or []) / 6.0
    g[NUM_OPP_PRIZE] = len(ps_opp.get("prize") or []) / 6.0
    g[NUM_MIN_COUNT] = (sel.get("minCount") or 0) / 6.0
    g[NUM_MAX_COUNT] = (sel.get("maxCount") or 0) / 6.0
    g[NUM_REMAIN_DMG] = (sel.get("remainDamageCounter") or 0) / 10.0
    g[NUM_REMAIN_ENE] = (sel.get("remainEnergyCost") or 0) / 5.0
    buf.add(TOKEN_GLB, numeric=g)

    # ---- 場ポケモン（自分 → 相手） --------------------------------------
    def add_in_play(ps: dict, is_mine: bool, t_active: int, t_bench: int):
        # active/bench の要素はセットアップ中などに None があり得るためスキップ
        status = [bool(ps.get(k)) for k in
                  ("asleep", "burned", "confused", "paralyzed", "poisoned")]
        for p in (ps.get("active") or []):
            if not isinstance(p, dict):
                continue
            pre = (p.get("preEvolution") or [])
            buf.add(t_active, card_id=p.get("id") or 0,
                    aux_card_id=(pre[0].get("id") if pre else 0) or 0,
                    numeric=_pokemon_numeric(p, is_mine, 0, status))
        for i, p in enumerate(ps.get("bench") or []):
            if not isinstance(p, dict):
                continue
            pre = (p.get("preEvolution") or [])
            buf.add(t_bench, card_id=p.get("id") or 0,
                    aux_card_id=(pre[0].get("id") if pre else 0) or 0,
                    numeric=_pokemon_numeric(p, is_mine, i, None))

    add_in_play(ps_me, True, TOKEN_MY_ACTIVE, TOKEN_MY_BENCH)

    # ---- 自分の手札（1枚1トークン。公開情報） ---------------------------
    for i, c in enumerate(ps_me.get("hand") or []):
        if not isinstance(c, dict):
            continue
        v = np.zeros(NF, np.float32)
        v[NUM_IS_MINE] = 1.0
        v[NUM_POS] = i / 5.0
        buf.add(TOKEN_MY_HAND, card_id=c.get("id") or 0, numeric=v)

    # ---- 捨札（ユニークカードごとに1トークン、枚数は numeric） ----------
    def add_discard(ps: dict, is_mine: bool, tok: int):
        cnt = Counter((c.get("id") or 0) for c in (ps.get("discard") or [])
                      if isinstance(c, dict))
        for cid, n in cnt.most_common():
            v = np.zeros(NF, np.float32)
            v[NUM_IS_MINE] = 1.0 if is_mine else 0.0
            v[NUM_COUNT] = n / 4.0
            buf.add(tok, card_id=cid, numeric=v)

    add_discard(ps_me, True, TOKEN_MY_DISCARD)
    add_in_play(ps_opp, False, TOKEN_OPP_ACTIVE, TOKEN_OPP_BENCH)
    add_discard(ps_opp, False, TOKEN_OPP_DISCARD)

    # ---- スタジアム ------------------------------------------------------
    for s in (st.get("stadium") or []):
        if not isinstance(s, dict):
            continue
        v = np.zeros(NF, np.float32)
        v[NUM_STADIUM_MINE] = 1.0 if s.get("playerIndex") == me else 0.0
        buf.add(TOKEN_STADIUM, card_id=s.get("id") or 0, numeric=v)

    # ---- looking（効果解決中に見えているカード） -------------------------
    for i, c in enumerate(st.get("looking") or []):
        v = np.zeros(NF, np.float32)
        v[NUM_POS] = i / 5.0
        buf.add(TOKEN_LOOKING, card_id=(c.get("id") if isinstance(c, dict) else 0) or 0,
                numeric=v)

    # ---- option トークン（必ず末尾） ------------------------------------
    for i, o in enumerate(d["options"]):
        v = np.zeros(NF, np.float32)
        v[NUM_OPT_INDEX] = i / 40.0
        pidx = o.get("playerIndex")
        v[NUM_IS_MINE] = 1.0 if (pidx is None or pidx == me) else 0.0
        buf.add(TOKEN_OPTION,
                card_id=o.get("_card_id") or 0,
                attack_id=o.get("attackId") or 0,
                opt_type=(o.get("type") if o.get("type") is not None else -1) + 1,
                area=o.get("area") or 0,
                numeric=v)
    return buf


# ---------------------------------------------------------------------------
# メイン: JSONL 全体 → npz
# ---------------------------------------------------------------------------

def featurize_file(jsonl_path: str, static_dir: str, out_path: str,
                   limit: int = 0) -> dict:
    """JSONL 全体を変換して .npz に保存し、統計 dict を返す。"""
    card_static, attack_static, info = build_static_tables(static_dir)

    flat = {k: [] for k in ("tok_type", "card_id", "aux_card_id",
                            "attack_id", "opt_type", "area")}
    numeric_rows: list[np.ndarray] = []
    tok_count, opt_count = [], []
    label_flat, label_count = [], []
    won, turn, sel_type, sel_ctx, n_options, is_forced = [], [], [], [], [], []
    episode_ids: list[str] = []
    # Phase 2b (PPO) 用: 挙動方策の log π(a|s) と V(s)（IL データには無い→default 0）
    logprobs, value_preds = [], []
    # Phase 2a (オフラインRL) 用: 軌跡復元キー(step/player_index)と終端報酬
    steps, player_indices, rewards = [], [], []
    n_skip = 0

    with open(jsonl_path, encoding="utf-8") as f:
        for ln, line in enumerate(f):
            if limit and len(tok_count) >= limit:
                break
            d = json.loads(line)
            # ラベル妥当性（parser 側で保証済みだが防御的に再確認）
            K = len(d["options"])
            lab = d["label"]
            if K == 0 or not all(0 <= x < K for x in lab):
                n_skip += 1
                continue
            buf = sample_to_tokens(d)
            for k in flat:
                flat[k].extend(getattr(buf, k))
            numeric_rows.extend(buf.numeric)
            tok_count.append(len(buf))
            opt_count.append(K)
            label_flat.extend(lab)
            label_count.append(len(lab))
            won.append(1.0 if d.get("won") else 0.0)
            turn.append(d.get("turn") or 0)
            sel_type.append(min(d["select"].get("type") or 0, SEL_TYPE_VOCAB - 1))
            sel_ctx.append(min(d["select"].get("context") or 0, SEL_CTX_VOCAB - 1))
            n_options.append(K)
            is_forced.append(1 if d.get("is_forced") else 0)
            episode_ids.append(str(d.get("episode_id") or ""))
            steps.append(d.get("step") or 0)
            player_indices.append(d.get("player_index") or 0)
            rewards.append(float(d.get("reward") or 0.0))
            logprobs.append(float(d["logprob"]) if d.get("logprob") is not None else 0.0)
            value_preds.append(float(d["value"]) if d.get("value") is not None else 0.0)

    config = {
        "NF": NF, "N_TOKEN_TYPES": N_TOKEN_TYPES,
        "OPT_TYPE_VOCAB": OPT_TYPE_VOCAB, "AREA_VOCAB": AREA_VOCAB,
        "SEL_TYPE_VOCAB": SEL_TYPE_VOCAB, "SEL_CTX_VOCAB": SEL_CTX_VOCAB,
        "S_CARD": S_CARD, "S_ATK": S_ATK,
        "n_card": info["n_card"], "n_attack": info["n_attack"],
        "source_jsonl": os.path.basename(jsonl_path),
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(
        out_path,
        tok_type=np.asarray(flat["tok_type"], np.uint8),
        card_id=np.asarray(flat["card_id"], np.int16),
        aux_card_id=np.asarray(flat["aux_card_id"], np.int16),
        attack_id=np.asarray(flat["attack_id"], np.int16),
        opt_type=np.asarray(flat["opt_type"], np.uint8),
        area=np.asarray(flat["area"], np.uint8),
        numeric=np.stack(numeric_rows).astype(np.float32),
        tok_count=np.asarray(tok_count, np.int32),
        opt_count=np.asarray(opt_count, np.int32),
        label_flat=np.asarray(label_flat, np.int32),
        label_count=np.asarray(label_count, np.uint8),
        won=np.asarray(won, np.float32),
        turn=np.asarray(turn, np.int16),
        sel_type=np.asarray(sel_type, np.uint8),
        sel_ctx=np.asarray(sel_ctx, np.uint8),
        n_options=np.asarray(n_options, np.int16),
        is_forced=np.asarray(is_forced, np.uint8),
        episode_id=np.asarray(episode_ids),
        step=np.asarray(steps, np.int32),
        player_index=np.asarray(player_indices, np.uint8),
        reward=np.asarray(rewards, np.float32),
        logprob=np.asarray(logprobs, np.float32),
        value_pred=np.asarray(value_preds, np.float32),
        card_static=card_static,
        attack_static=attack_static,
        config_json=np.asarray(json.dumps(config)),
    )
    tc = np.asarray(tok_count)
    stats = {
        "samples": len(tok_count), "skipped": n_skip,
        "episodes": len(set(episode_ids)),
        "tokens_total": int(tc.sum()),
        "tokens_mean": float(tc.mean()), "tokens_max": int(tc.max()),
        "out": out_path,
        "size_mb": os.path.getsize(out_path) / 1e6,
    }
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="IL教師ペア JSONL → 学習テンソル npz")
    ap.add_argument("jsonl", help="教師ペア JSONL（1行 = 1意思決定）")
    ap.add_argument("-o", "--out", required=True, help="出力 .npz パス")
    ap.add_argument("--static", default="../../data/static",
                    help="cards.json/attacks.json のディレクトリ")
    ap.add_argument("--limit", type=int, default=0, help="先頭N件のみ（スモークテスト用）")
    args = ap.parse_args()

    stats = featurize_file(args.jsonl, args.static, args.out, args.limit)
    print(f"[featurize] samples={stats['samples']} (skip={stats['skipped']}) "
          f"episodes={stats['episodes']}\n"
          f"            tokens: total={stats['tokens_total']:,} "
          f"mean={stats['tokens_mean']:.1f} max={stats['tokens_max']}\n"
          f"            -> {stats['out']} ({stats['size_mb']:.1f} MB)",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
