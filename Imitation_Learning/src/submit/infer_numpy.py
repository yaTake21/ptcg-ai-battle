#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""infer_numpy.py — 学習済みモデルの numpy 手書き forward（提出用推論器）。

■ なぜ numpy なのか
  提出/対戦の実行環境（cabt ランタイム）に **PyTorch は入っていない**（numpy のみ。
  Docker イメージで確認済み）。そこで model.py の forward を **numpy だけ**で再実装し、
  export_weights.py が書き出した weights.npz を読んで推論する。これで torch 非依存。

■ model.py との一致
  下の forward は model.py（PTCGNet）と数値的に一致するよう実装している。
  一致は tests / __main__ の parity チェックで検証する（torch がある環境で）。
  再現している構成: 埋め込み合成 → input_proj(Linear+LayerNorm) →
  TransformerEncoderLayer×3(Pre-LN, GELU, 4head) → policy/value ヘッド。

■ 特徴量化の共有
  トークン化は featurize.sample_to_tokens をそのまま使う（学習と同一 = パリティ）。
  option の _card_id 解決だけは推論時に必要なので、cg 非依存の dict 版を同梱する
  （学習データ抽出時と同じロジック。int 定数で cg import を避け、どの環境でもテスト可）。

■ 使い方（提出の main.py から）
    from infer_numpy import NumpyPolicy
    pol = NumpyPolicy("weights.npz", "static")      # 起動時に一度ロード
    picks = pol.predict(obs_dict, my_index)         # 毎決定: options への index リスト

■ 単体実行（parity チェック / ベンチ）
    # src/submit/ から
    python3 infer_numpy.py \\
        --weights ../../runs/<run>/weights.npz \\
        --features ../../data/features/<features>.npz --check
"""
from __future__ import annotations

import json
import os
import sys
from typing import Optional

import numpy as np

# featurize（sample_to_tokens と TOKEN_*/NF 定数）を共有する。
# 提出ディレクトリでは同階層、リポジトリでは ../learn/ にあるため両対応。
try:
    import featurize as F
except ImportError:
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "learn"))
    import featurize as F

# --- AreaType / OptionType の int 値（cg/api.py と同値。cg 非依存にするため直書き） ---
A_DECK, A_HAND, A_DISCARD, A_ACTIVE, A_BENCH, A_PRIZE, A_STADIUM = 1, 2, 3, 4, 5, 6, 7
A_LOOKING = 12
O_CARD, O_TOOL_CARD, O_ENERGY_CARD, O_ENERGY = 3, 4, 5, 6
O_PLAY, O_ATTACH, O_EVOLVE, O_ABILITY, O_DISCARD = 7, 8, 9, 10, 11


# ---------------------------------------------------------------------------
# option の card_id 解決（学習データ抽出時と同一ロジックの dict 版・cg非依存）
# ---------------------------------------------------------------------------

def _get_card(state: dict, select: dict, area: int, index, pidx: int) -> Optional[dict]:
    try:
        ps = state["players"][pidx]
        if area == A_HAND:
            return ps["hand"][index]
        if area == A_DISCARD:
            return ps["discard"][index]
        if area == A_ACTIVE:
            return ps["active"][index]
        if area == A_BENCH:
            return ps["bench"][index]
        if area == A_PRIZE:
            return ps["prize"][index]
        if area == A_DECK:
            return (select.get("deck") or [])[index]
        if area == A_STADIUM:
            return state["stadium"][index]
        if area == A_LOOKING:
            return (state.get("looking") or [])[index]
    except (IndexError, TypeError, KeyError):
        return None
    return None


def resolve_option_card_id(state: dict, select: dict, opt: dict, me: int) -> Optional[int]:
    """推論時の生 option に card_id を付ける（学習データ抽出時と同一ロジック）。"""
    t = opt.get("type")
    if t == O_PLAY:
        c = _get_card(state, select, A_HAND, opt.get("index"), me)
    elif t in (O_CARD, O_TOOL_CARD, O_ENERGY_CARD, O_ENERGY,
               O_ATTACH, O_EVOLVE, O_ABILITY, O_DISCARD):
        area = opt.get("area")
        if area is None:
            return None
        pidx = opt.get("playerIndex")
        pidx = me if pidx is None else pidx
        c = _get_card(state, select, area, opt.get("index"), pidx)
    else:
        return None
    return c.get("id") if isinstance(c, dict) else None


# ---------------------------------------------------------------------------
# numpy の演算プリミティブ（torch と数値一致するよう実装）
# ---------------------------------------------------------------------------

def gelu(x: np.ndarray) -> np.ndarray:
    """nn.GELU()（厳密版, erf ベース）と一致。"""
    from math import sqrt
    # 0.5x(1+erf(x/sqrt2)) を numpy で。erf は math にしか無いので近似せず vectorize。
    import numpy as _np
    # numpy に erf は無いので tanh 近似ではなく scipy 非依存の実装:
    # erf を有理式近似（Abramowitz-Stegun 7.1.26, 誤差<1.5e-7）
    z = x / sqrt(2.0)
    sign = _np.sign(z)
    az = _np.abs(z)
    t = 1.0 / (1.0 + 0.3275911 * az)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * _np.exp(-az * az)
    erf = sign * y
    return 0.5 * x * (1.0 + erf)


def layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """nn.LayerNorm（最終次元を正規化）。"""
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def linear(x: np.ndarray, w: np.ndarray, b: Optional[np.ndarray]) -> np.ndarray:
    """nn.Linear: y = x @ w.T + b（w は [out, in]）。"""
    y = x @ w.T
    return y + b if b is not None else y


def softmax_lastdim(x: np.ndarray) -> np.ndarray:
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


# ---------------------------------------------------------------------------
# 推論器
# ---------------------------------------------------------------------------

class NumpyPolicy:
    """weights.npz を読み込み、obs_dict から選択 index を返す numpy 推論器。"""

    def __init__(self, weights_path: str, static_dir: str):
        z = np.load(weights_path, allow_pickle=False)
        self.W = {k: z[k] for k in z.files if not k.startswith("__")}
        self.cfg = json.loads(str(z["__model_config__"]))
        self.n_head = self.cfg["n_head"]
        self.n_layer = self.cfg["n_layer"]
        self.d_model = self.cfg["d_model"]
        # 静的テーブルは weights.npz に含まれる（学習時と同一を保証）
        self.card_static = self.W["card_static"]
        self.attack_static = self.W["attack_static"]

    # -- トークン列（featurize と同一） → モデル入力配列 --
    def _tokens_from_sample(self, sample: dict) -> dict:
        buf = F.sample_to_tokens(sample)
        return {
            "tok_type": np.asarray(buf.tok_type, np.int64),
            "card_id": np.asarray(buf.card_id, np.int64),
            "aux_card_id": np.asarray(buf.aux_card_id, np.int64),
            "attack_id": np.asarray(buf.attack_id, np.int64),
            "opt_type": np.asarray(buf.opt_type, np.int64),
            "area": np.asarray(buf.area, np.int64),
            "numeric": np.stack(buf.numeric).astype(np.float32),
            "n_opt": sum(1 for t in buf.tok_type if t == F.TOKEN_OPTION),
        }

    def _embed(self, t: dict, sel_type: int, sel_ctx: int) -> np.ndarray:
        """model.py の埋め込み合成 → input_proj（[T, d_model]）。"""
        W = self.W
        parts = [
            W["emb_card.weight"][t["card_id"]],
            linear(self.card_static[t["card_id"]], W["proj_card_static.weight"],
                   W["proj_card_static.bias"]),
            W["emb_card.weight"][t["aux_card_id"]],
            W["emb_attack.weight"][t["attack_id"]],
            linear(self.attack_static[t["attack_id"]], W["proj_attack_static.weight"],
                   W["proj_attack_static.bias"]),
            W["emb_tok.weight"][t["tok_type"]],
            W["emb_opt.weight"][t["opt_type"]],
            W["emb_area.weight"][t["area"]],
        ]
        T = len(t["tok_type"])
        sel_t = np.tile(W["emb_sel_type.weight"][sel_type], (T, 1))
        sel_c = np.tile(W["emb_sel_ctx.weight"][sel_ctx], (T, 1))
        x = np.concatenate(parts + [sel_t, sel_c, t["numeric"]], axis=-1)
        x = linear(x, W["input_proj.0.weight"], W["input_proj.0.bias"])
        x = layer_norm(x, W["input_proj.1.weight"], W["input_proj.1.bias"])
        return x

    def _attn(self, x: np.ndarray, li: int) -> np.ndarray:
        """nn.MultiheadAttention（self-attention, パディングなし = 1サンプル推論）。"""
        W = self.W
        p = f"encoder.layers.{li}.self_attn."
        in_w = W[p + "in_proj_weight"]      # [3d, d]
        in_b = W[p + "in_proj_bias"]        # [3d]
        d = self.d_model
        qkv = linear(x, in_w, in_b)         # [T, 3d]
        q, k, v = qkv[:, :d], qkv[:, d:2 * d], qkv[:, 2 * d:]
        h, dh = self.n_head, d // self.n_head
        T = x.shape[0]
        # [T, h, dh] -> [h, T, dh]
        q = q.reshape(T, h, dh).transpose(1, 0, 2)
        k = k.reshape(T, h, dh).transpose(1, 0, 2)
        v = v.reshape(T, h, dh).transpose(1, 0, 2)
        scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(dh)   # [h, T, T]
        attn = softmax_lastdim(scores)
        ctx = attn @ v                                       # [h, T, dh]
        ctx = ctx.transpose(1, 0, 2).reshape(T, d)           # [T, d]
        return linear(ctx, W[p + "out_proj.weight"], W[p + "out_proj.bias"])

    def _encoder_layer(self, x: np.ndarray, li: int) -> np.ndarray:
        """TransformerEncoderLayer(norm_first=True, GELU)。"""
        W = self.W
        p = f"encoder.layers.{li}."
        # Pre-LN self-attention
        h = layer_norm(x, W[p + "norm1.weight"], W[p + "norm1.bias"])
        x = x + self._attn(h, li)
        # Pre-LN FFN
        h = layer_norm(x, W[p + "norm2.weight"], W[p + "norm2.bias"])
        h = linear(h, W[p + "linear1.weight"], W[p + "linear1.bias"])
        h = gelu(h)
        h = linear(h, W[p + "linear2.weight"], W[p + "linear2.bias"])
        return x + h

    def _head(self, h: np.ndarray, name: str) -> np.ndarray:
        W = self.W
        y = gelu(linear(h, W[f"{name}.0.weight"], W[f"{name}.0.bias"]))
        return linear(y, W[f"{name}.2.weight"], W[f"{name}.2.bias"]).squeeze(-1)

    # -- forward: sample dict -> (policy_logits[K], value_logit) --
    def forward_sample(self, sample: dict) -> tuple[np.ndarray, float]:
        t = self._tokens_from_sample(sample)
        x = self._embed(t, sample["select"]["type"] or 0,
                        min(sample["select"].get("context") or 0,
                            self.cfg["sel_ctx_vocab"] - 1))
        for li in range(self.n_layer):
            x = self._encoder_layer(x, li)
        value = float(self._head(x[:1], "value_head")[0])   # [GLB]
        # option は末尾 K 個（featurize の契約）
        K = t["n_opt"]
        opt_h = x[-K:] if K > 0 else x[:0]
        logits = self._head(opt_h, "policy_head") if K > 0 else np.zeros(0)
        return np.atleast_1d(logits), value

    # -- 提出 API: obs_dict + 自分の index -> 選択 index リスト --
    def predict(self, obs_dict: dict, my_index: int) -> list[int]:
        """毎決定の推論。obs_dict は agent が受け取る dict そのもの。"""
        sel = obs_dict.get("select")
        state = obs_dict.get("current")
        if not sel or state is None:
            return []
        options = sel.get("option") or []
        if not options:
            return []
        # _card_id を解決して学習時と同じ形にする
        enriched = []
        for o in options:
            oo = dict(o)
            oo["_card_id"] = resolve_option_card_id(state, sel, o, my_index)
            enriched.append(oo)
        sample = {"state": state, "select": sel,
                  "player_index": my_index, "options": enriched}
        logits, _ = self.forward_sample(sample)
        # 必要枚数を logit 降順に選ぶ（§3.5 v1 の推論規約）
        mx = sel.get("maxCount") or 1
        mn = sel.get("minCount")
        k = mx if mn is None else max(mn, min(mx, mx))
        k = min(k, len(options))
        order = list(np.argsort(-logits))
        return [int(i) for i in order[:k]]


# ---------------------------------------------------------------------------
# parity チェック（torch がある環境のみ）
# ---------------------------------------------------------------------------

def _check_parity(weights_path: str, features_path: str, n: int = 50) -> None:
    """model.py(torch) と infer_numpy の logits 一致を検証する。"""
    import torch
    from model import ModelConfig, PTCGNet
    from dataset import PTCGFeatureData

    data = PTCGFeatureData(features_path)
    # モデル構成は weights.npz に同梱の __model_config__ から復元する
    # （d_model/n_layer/n_head を固定既定にすると d256 等の大型モデルで形が合わない）
    z = np.load(weights_path, allow_pickle=False)
    cfg = ModelConfig(**json.loads(str(z["__model_config__"])))
    net = PTCGNet(cfg, data.card_static, data.attack_static)
    sd = {k: torch.from_numpy(z[k]) for k in z.files
          if not k.startswith("__") and k in dict(net.state_dict())}
    net.load_state_dict(sd, strict=False)
    net.eval()

    pol = NumpyPolicy(weights_path, "")
    # JSONL からではなく npz サンプルを直接復元して比較するため、
    # ここでは元 JSONL を使わず、npz の1サンプルを sample dict に近い形へ戻すのは煩雑。
    # 代わりに featurize 済みトークンを両者に流して logits を比較する。
    from torch.utils.data import DataLoader
    from dataset import PTCGDataset, collate
    idx = np.arange(min(n, data.n))
    loader = DataLoader(PTCGDataset(data, idx), batch_size=1, collate_fn=collate)

    max_diff = 0.0
    for b, batch in enumerate(loader):
        with torch.no_grad():
            tl, _, _ = net(batch)
        tl = tl[0][batch["opt_mask"].sum().item() and slice(0, int(batch["n_options"][0]))]
        tl = tl[torch.isfinite(tl)].numpy()
        # numpy 側: 同じトークンを流す（batch を numpy sample に変換）
        nl = _numpy_from_batch(pol, batch)
        d = float(np.abs(tl - nl).max()) if len(nl) == len(tl) else 9.9
        max_diff = max(max_diff, d)
    print(f"[parity] samples={min(n, data.n)}  max|Δlogit|={max_diff:.2e} "
          f"({'OK' if max_diff < 1e-3 else 'NG'})")


def _numpy_from_batch(pol: "NumpyPolicy", batch) -> np.ndarray:
    """collate 済み1件バッチ（torch）を numpy forward に流して policy logits を返す。"""
    T = int(batch["tok_mask"][0].sum())
    t = {
        "tok_type": batch["tok_type"][0, :T].numpy().astype(np.int64),
        "card_id": batch["card_id"][0, :T].numpy().astype(np.int64),
        "aux_card_id": batch["aux_card_id"][0, :T].numpy().astype(np.int64),
        "attack_id": batch["attack_id"][0, :T].numpy().astype(np.int64),
        "opt_type": batch["opt_type"][0, :T].numpy().astype(np.int64),
        "area": batch["area"][0, :T].numpy().astype(np.int64),
        "numeric": batch["numeric"][0, :T].numpy().astype(np.float32),
        "n_opt": int(batch["n_options"][0]),
    }
    x = pol._embed(t, int(batch["sel_type"][0]), int(batch["sel_ctx"][0]))
    for li in range(pol.n_layer):
        x = pol._encoder_layer(x, li)
    K = t["n_opt"]
    return pol._head(x[-K:], "policy_head")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="numpy 推論器 / parity チェック")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--features", help="parity チェック用 npz")
    ap.add_argument("--static", default="../data/static")
    ap.add_argument("--check", action="store_true", help="torch との一致を検証")
    args = ap.parse_args()
    if args.check:
        _check_parity(args.weights, args.features)
    else:
        pol = NumpyPolicy(args.weights, args.static)
        print(f"[infer_numpy] loaded {args.weights} "
              f"(d={pol.d_model}, layers={pol.n_layer})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
