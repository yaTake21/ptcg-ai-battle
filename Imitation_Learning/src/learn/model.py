#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""model.py — 候補スコアリング型 State-Transformer。

■ 全体像
    トークン列（状態 + 候補手）→ 埋め込み合成 → Transformer Encoder →
      - 候補手トークンの出力 → policy head → 候補上の softmax（π）
      - 候補手トークンの出力 → Q head    → Q(s, a_i)（Phase 2a 用。Phase 1 では未使用）
      - [GLB] トークンの出力 → value head → V(s)（勝敗予測の補助学習。RL の critic 初期値）

■ トークン埋め込みの合成（§3.1〜3.3）
    各トークンの入力ベクトル = concat[
        CardVec   = Embed_card(card_id) ⊕ Linear(card_static[card_id]),   … 128
        AuxVec    = Embed_card(aux_card_id) の 64（進化元）,               … 64
        AttackVec = Embed_atk(attack_id) ⊕ Linear(attack_static[attack_id]), … 64
        Embed_tokType, Embed_optType, Embed_area,                          … 32+32+8
        Embed_selType, Embed_selCtx（全トークンに同報 = 選択文脈の条件付け）, … 16+16
        numeric（NF=44 の実数特徴）
    ] → Linear → d_model

    ※ card_static / attack_static は featurize.py が npz に同梱した凍結テーブル。
      「id 埋め込み = 個体の記憶」「静的属性 = 未知カードへの汎化」の併用（§1.3, §3.1）。

■ 出力の契約
    forward(batch) は (policy_logits [B,Kmax], value_logit [B], q_values [B,Kmax]) を返す。
    - policy_logits は非合法（パディング）位置が -inf。softmax はそのまま取れる。
    - value は logit（勝率予測）。V∈(-1,1) が欲しければ 2σ(logit)-1（§6.3）。
    - トークン列の「末尾 K 個が候補手」という featurize.py の契約を利用して
      候補手トークンの出力を抜き出す。

■ サイズ（デフォルト設定）
    d_model=128, 3層, 4head, FFN=512 → 約 2.0M パラメータ（埋め込み込み）。
    CPU 推論 数ms/決定（1手1秒制限に対し余裕。§3.6）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    """モデルの全ハイパーパラメータ。featurize.py の config_json と整合させて作る。

    Attributes:
        n_card / n_attack : カード / ワザの語彙サイズ（id 直接インデックス。0 = なし）
        nf                : numeric 特徴の次元（featurize.py の NF）
        s_card / s_atk    : 静的属性テーブルの次元
        各 *_vocab        : トークン種別 / OptionType / Area / SelectType / SelectContext の語彙
        d_model 等        : Transformer 本体の設定
    """
    n_card: int = 1268
    n_attack: int = 1557
    nf: int = 44
    s_card: int = 58
    s_atk: int = 14
    n_token_types: int = 12
    opt_type_vocab: int = 18
    area_vocab: int = 13
    sel_type_vocab: int = 12
    sel_ctx_vocab: int = 48
    d_card: int = 64          # カード id 埋め込み次元
    d_card_static: int = 64   # カード静的属性の投影次元
    d_attack: int = 32        # ワザ id 埋め込み次元
    d_attack_static: int = 32
    d_tok: int = 32           # トークン種別埋め込み
    d_opt: int = 32           # OptionType 埋め込み
    d_area: int = 8
    d_sel: int = 16           # SelectType / SelectContext 各16
    d_model: int = 128
    n_head: int = 4
    n_layer: int = 3
    d_ff: int = 512
    dropout: float = 0.1

    @classmethod
    def from_feature_config(cls, feat_cfg: dict, **overrides) -> "ModelConfig":
        """featurize.py が npz に埋めた config_json から整合の取れた設定を作る。"""
        return cls(
            n_card=feat_cfg["n_card"], n_attack=feat_cfg["n_attack"],
            nf=feat_cfg["NF"], s_card=feat_cfg["S_CARD"], s_atk=feat_cfg["S_ATK"],
            n_token_types=feat_cfg["N_TOKEN_TYPES"],
            opt_type_vocab=feat_cfg["OPT_TYPE_VOCAB"], area_vocab=feat_cfg["AREA_VOCAB"],
            sel_type_vocab=feat_cfg["SEL_TYPE_VOCAB"], sel_ctx_vocab=feat_cfg["SEL_CTX_VOCAB"],
            **overrides,
        )


class PTCGNet(nn.Module):
    """候補スコアリング型 State-Transformer（π / V / Q の3ヘッド共有バックボーン）。

    Args:
        cfg          : ModelConfig
        card_static  : [n_card, S_CARD] 静的属性テーブル（凍結バッファとして登録）
        attack_static: [n_attack, S_ATK] 同上
    """

    def __init__(self, cfg: ModelConfig,
                 card_static: np.ndarray, attack_static: np.ndarray):
        super().__init__()
        self.cfg = cfg

        # ---- 凍結の静的属性テーブル（勾配なし・checkpoint に一緒に保存される） ----
        self.register_buffer("card_static",
                             torch.from_numpy(card_static.astype(np.float32)))
        self.register_buffer("attack_static",
                             torch.from_numpy(attack_static.astype(np.float32)))

        # ---- 埋め込み（padding_idx=0 で「なし」をゼロ埋め込みに） ----
        self.emb_card = nn.Embedding(cfg.n_card, cfg.d_card, padding_idx=0)
        self.emb_attack = nn.Embedding(cfg.n_attack, cfg.d_attack, padding_idx=0)
        self.emb_tok = nn.Embedding(cfg.n_token_types, cfg.d_tok, padding_idx=0)
        self.emb_opt = nn.Embedding(cfg.opt_type_vocab, cfg.d_opt, padding_idx=0)
        self.emb_area = nn.Embedding(cfg.area_vocab, cfg.d_area, padding_idx=0)
        self.emb_sel_type = nn.Embedding(cfg.sel_type_vocab, cfg.d_sel)
        self.emb_sel_ctx = nn.Embedding(cfg.sel_ctx_vocab, cfg.d_sel)

        # ---- 静的属性の投影 ----
        self.proj_card_static = nn.Linear(cfg.s_card, cfg.d_card_static)
        self.proj_attack_static = nn.Linear(cfg.s_atk, cfg.d_attack_static)

        # ---- トークン入力の合成 → d_model ----
        d_in = (cfg.d_card + cfg.d_card_static        # CardVec
                + cfg.d_card                          # AuxVec（進化元。埋め込みのみ）
                + cfg.d_attack + cfg.d_attack_static  # AttackVec
                + cfg.d_tok + cfg.d_opt + cfg.d_area
                + cfg.d_sel * 2                       # select type/context（全トークン同報）
                + cfg.nf)
        self.input_proj = nn.Sequential(
            nn.Linear(d_in, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )

        # ---- Transformer Encoder（Pre-LN。可変長は key_padding_mask で処理） ----
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_head, dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout, batch_first=True, norm_first=True,
            activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layer)

        # ---- 3ヘッド（§3.4 / §6.3） ----
        def head():
            return nn.Sequential(nn.Linear(cfg.d_model, 64), nn.GELU(),
                                 nn.Linear(64, 1))
        self.policy_head = head()   # 候補手トークン → logit
        self.value_head = head()    # [GLB] → 勝敗 logit（Phase 1 から補助学習）
        self.q_head = head()        # 候補手トークン → Q(s,a)（Phase 2a で有効化）

    # ------------------------------------------------------------------
    def forward(self, batch: dict[str, torch.Tensor]
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """バッチを順伝播する。

        Args:
            batch: dataset.collate() の出力。
                tok_type/card_id/aux_card_id/attack_id/opt_type/area : [B, T] int
                numeric  : [B, T, NF] float
                sel_type / sel_ctx : [B] int（全トークンに同報される）
                tok_mask : [B, T] bool（True = 実トークン）
                opt_mask : [B, T] bool（True = 候補手トークン）
        Returns:
            policy_logits: [B, Kmax] 候補手のスコア（パディング位置は -inf）
            value_logit  : [B] 勝敗予測 logit
            q_values     : [B, Kmax] 行動価値（Phase 2a 用。パディング位置は 0）
            ※ Kmax = バッチ内の最大候補数。各サンプルの実候補数は opt_mask.sum(1)。
        """
        B, T = batch["tok_type"].shape
        cs = self.card_static[batch["card_id"]]          # [B,T,S_CARD]
        as_ = self.attack_static[batch["attack_id"]]     # [B,T,S_ATK]
        sel_t = self.emb_sel_type(batch["sel_type"])     # [B,d_sel]
        sel_c = self.emb_sel_ctx(batch["sel_ctx"])       # [B,d_sel]

        x = torch.cat([
            self.emb_card(batch["card_id"]),
            self.proj_card_static(cs),
            self.emb_card(batch["aux_card_id"]),
            self.emb_attack(batch["attack_id"]),
            self.proj_attack_static(as_),
            self.emb_tok(batch["tok_type"]),
            self.emb_opt(batch["opt_type"]),
            self.emb_area(batch["area"]),
            sel_t.unsqueeze(1).expand(B, T, -1),
            sel_c.unsqueeze(1).expand(B, T, -1),
            batch["numeric"],
        ], dim=-1)
        h = self.input_proj(x)                            # [B,T,d]
        h = self.encoder(h, src_key_padding_mask=~batch["tok_mask"])

        # ---- [GLB]（先頭トークン）→ value ----
        value_logit = self.value_head(h[:, 0]).squeeze(-1)          # [B]

        # ---- 候補手トークン → policy logits / Q ----
        # featurize の契約により候補手は各サンプルの末尾 K 個。opt_mask で抜き出し、
        # [B, Kmax] に右詰めではなく「候補順のまま左詰め」で並べ替える。
        opt_mask = batch["opt_mask"]                                # [B,T]
        Kmax = int(opt_mask.sum(1).max().item())
        scores = self.policy_head(h).squeeze(-1)                    # [B,T]
        qvals = self.q_head(h).squeeze(-1)                          # [B,T]

        policy_logits = scores.new_full((B, Kmax), float("-inf"))
        q_values = qvals.new_zeros((B, Kmax))
        # 各行の候補手位置を左詰めに集める（B は高々数百なのでループで十分軽い）
        for b in range(B):
            idx = opt_mask[b].nonzero(as_tuple=True)[0]
            policy_logits[b, : len(idx)] = scores[b, idx]
            q_values[b, : len(idx)] = qvals[b, idx]
        return policy_logits, value_logit, q_values

    # ------------------------------------------------------------------
    def num_parameters(self) -> int:
        """学習対象パラメータ数（凍結バッファは含まない）。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def config_json(self) -> str:
        return json.dumps(asdict(self.cfg))


def multi_positive_ce(policy_logits: torch.Tensor,
                      label_mh: torch.Tensor) -> torch.Tensor:
    """multi-positive CE（正解が複数ありうる選択のためのクロスエントロピー）。

    単一選択なら普通の CE と一致し、複数選択（label が複数 index）は
    正解集合の平均対数尤度を最大化する（順序は集合扱い。§3.5 の全数調査より
    本データに順序依存文脈は存在しないため損失なし）。

    Args:
        policy_logits: [B, Kmax]（非合法位置は -inf）
        label_mh     : [B, Kmax] 正解位置が 1 の multi-hot
    Returns:
        スカラー損失（バッチ平均）
    """
    logp = torch.log_softmax(policy_logits, dim=-1)
    # -inf の位置は label_mh=0 なので 0*(-inf)=nan を避けるため masked_fill
    logp = logp.masked_fill(label_mh == 0, 0.0)
    per_sample = -(logp * label_mh).sum(-1) / label_mh.sum(-1).clamp(min=1)
    return per_sample.mean()
