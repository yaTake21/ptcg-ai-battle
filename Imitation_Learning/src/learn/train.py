#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train.py — 模倣学習（Behavior Cloning）の学習ループ。

■ 損失
    L = multi_positive_CE(π, label)          … 主: 教師の手の模倣
      + λ_v · BCEWithLogits(V, won)          … 補助: 勝敗予測（最終レシピでは λ_v=0）
    ラベルスムージングは multi-positive との相性が悪いので使わず、
    代わりに dropout と weight decay で正則化する。

■ 評価（すべて episode 単位分割の val で計測）
    - acc_top1        : 単一選択サンプルの top-1 一致率（目標 ≥80%。相場 66〜85%）
    - acc_exact       : 全サンプルで「予測集合 == 正解集合」の完全一致率（複数選択含む）
    - acc_by_seltype  : select.type 別（MAIN/CARD/YES_NO…の診断）
    - acc_late        : turn≥5 の一致率（§3.7 記憶導入の効果判定に使う基準値）
    - value_auc       : 勝敗予測の AUC（value head の品質）
    ※ 強制手（候補1個）は常に正解になるため精度からは除外し、参考値として別掲。

■ 学習曲線（§4.2）
    --data-fraction 0.25 / 0.5 / 1.0 で3回実行し val acc を比較する。
    間引きは episode 単位・val は常に全量（公平な比較のため。dataset.py 参照）。

■ 使い方
    # ローカル/Colab 共通（GPU があれば自動使用）
    # src/learn/ から（出力は runs/<プレイヤー>_<subId>/<run名> に置く規約）
    python3 train.py --features ../../data/features/<features>.npz \\
        --out ../../runs/<run> --epochs 30
    # スモークテスト（数十秒で1周）
    python3 train.py --features <npz> --out /tmp/run --smoke

■ 出力（--out ディレクトリ）
    best.pt      : val acc_exact 最良の checkpoint（model + config + メタ）
    last.pt      : 最終 epoch
    history.json : epoch ごとの全メトリクス（学習曲線・レポート作成用）
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import PTCGFeatureData, PTCGDataset, collate
from model import ModelConfig, PTCGNet, multi_positive_ce


# ---------------------------------------------------------------------------
# 評価
# ---------------------------------------------------------------------------

def evaluate(model: PTCGNet, loader: DataLoader, device: str) -> dict:
    """val セットを1周して全メトリクスを返す（§4.2）。"""
    model.eval()
    n_single = n_single_ok = 0            # 単一選択 top-1
    n_all = n_all_ok = 0                  # 完全一致（複数選択含む・強制手除く）
    n_forced = n_forced_ok = 0            # 参考: 強制手
    n_late = n_late_ok = 0                # turn>=5
    by_type: dict[int, list[int]] = {}    # sel_type -> [ok, total]
    v_scores, v_labels = [], []
    loss_sum = n_batches = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits, v_logit, _ = model(batch)
            loss = multi_positive_ce(logits, batch["label_mh"])
            loss_sum += float(loss)
            n_batches += 1

            # 予測集合 = 「正解と同じ個数」を logit 降順で選ぶ（§3.5 v1 の推論規約）
            label_mh = batch["label_mh"]
            n_lab = label_mh.sum(-1).long()                       # [B]
            B, K = logits.shape
            for b in range(B):
                k = int(n_lab[b])
                if k == 0:
                    continue
                pred = set(torch.topk(logits[b], k).indices.tolist())
                true = set(label_mh[b].nonzero(as_tuple=True)[0].tolist())
                ok = pred == true
                forced = bool(batch["is_forced"][b])
                st = int(batch["sel_type"][b])
                if forced:
                    n_forced += 1
                    n_forced_ok += ok
                    continue
                n_all += 1
                n_all_ok += ok
                if k == 1:
                    n_single += 1
                    n_single_ok += ok
                if int(batch["turn"][b]) >= 5:
                    n_late += 1
                    n_late_ok += ok
                by_type.setdefault(st, [0, 0])
                by_type[st][0] += ok
                by_type[st][1] += 1
            v_scores += torch.sigmoid(v_logit).tolist()
            v_labels += batch["won"].tolist()

    # 順位ベースの AUC（依存ライブラリなしの実装）
    s = np.asarray(v_scores)
    y = np.asarray(v_labels)
    if len(set(y.tolist())) == 2:
        order = np.argsort(s)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(s) + 1)
        n_pos, n_neg = int(y.sum()), int((1 - y).sum())
        auc = (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    else:
        auc = float("nan")

    return {
        "val_loss": loss_sum / max(n_batches, 1),
        "acc_top1": n_single_ok / max(n_single, 1),
        "acc_exact": n_all_ok / max(n_all, 1),
        "acc_late": n_late_ok / max(n_late, 1),
        "acc_forced_ref": n_forced_ok / max(n_forced, 1),
        "acc_by_seltype": {str(t): [c[0] / max(c[1], 1), c[1]]
                           for t, c in sorted(by_type.items())},
        "value_auc": float(auc),
        "n_val_single": n_single, "n_val_all": n_all,
    }


# ---------------------------------------------------------------------------
# 学習本体
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> dict:
    """学習を実行し、最終メトリクスを返す。"""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device
    if device == "auto":
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[train] device={device}")

    # ---- データ ----
    data = PTCGFeatureData(args.features)
    train_idx, val_idx = data.split_by_episode(
        val_frac=args.val_frac, seed=args.seed, data_fraction=args.data_fraction)
    if args.smoke:  # スモーク: 少量に絞って高速に1周
        train_idx, val_idx = train_idx[:512], val_idx[:256]
        args.epochs = 1
    print(f"[train] samples: train={len(train_idx)} val={len(val_idx)} "
          f"(episodes total={len(data.episodes)}, data_fraction={args.data_fraction})")

    dl_kw = dict(batch_size=args.batch_size, collate_fn=collate,
                 num_workers=args.workers, pin_memory=(device == "cuda"))
    train_loader = DataLoader(PTCGDataset(data, train_idx), shuffle=True, **dl_kw)
    val_loader = DataLoader(PTCGDataset(data, val_idx), shuffle=False, **dl_kw)

    # ---- モデル ----
    cfg = ModelConfig.from_feature_config(
        data.feat_cfg, d_model=args.d_model, n_layer=args.n_layer,
        n_head=args.n_head, d_ff=args.d_ff, dropout=args.dropout)
    model = PTCGNet(cfg, data.card_static, data.attack_static).to(device)
    print(f"[train] params={model.num_parameters()/1e6:.2f}M "
          f"(d={cfg.d_model}, layers={cfg.n_layer})")

    # ---- 初期重み（逐次学習の2段目）----
    # optimizer / scheduler は引き継がない。段2 は新しい OneCycle を張り直す
    # （＝「事前学習した重みから、別のデータで学習し直す」という意味づけ）。
    if args.init_ckpt:
        ck = torch.load(args.init_ckpt, map_location=device, weights_only=False)
        src_cfg, dst_cfg = json.loads(ck["model_config"]), json.loads(model.config_json())
        diff = {k: (v, dst_cfg.get(k)) for k, v in src_cfg.items() if dst_cfg.get(k) != v}
        if diff:
            raise SystemExit(f"[init-ckpt] モデル構成が一致しません: {diff}")
        model.load_state_dict(ck["model"])   # strict=True: 欠損/余剰があれば例外
        src_m = ck.get("metrics") or {}
        print(f"[init-ckpt] {args.init_ckpt} から {len(ck['model'])} テンソルを読み込み "
              f"(元 epoch={src_m.get('epoch')} acc_exact={src_m.get('acc_exact')})")

    # ---- trunk 凍結（転移の測定用・frozen-trunk probe）----
    # 「別デッキで学んだ盤面の読み方(trunk)が、そのまま対象デッキに使えるか」だけを
    # 切り出して測るためのモード。policy_head 以外の勾配を止め、trunk は dropout も
    # 切って（eval 固定）純粋な特徴抽出器として扱う。※学習用ではなく測定用。
    if args.freeze_trunk:
        for name, p in model.named_parameters():
            p.requires_grad_(name.startswith("policy_head"))
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in model.parameters())
        print(f"[freeze-trunk] 学習するのは policy_head のみ: "
              f"{n_tr/1e3:.1f}K / {n_all/1e6:.2f}M ({100*n_tr/n_all:.2f}%)")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr,
                            weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.05)

    os.makedirs(args.out, exist_ok=True)
    # best の選定基準。acc_exact は暗記でも上がるため、大型モデルの比較では
    # val_loss（小さいほど良い）を使う。スコアは「大きいほど良い」に正規化する。
    history, best_score = [], -float("inf")

    # 学習前の1回評価。--init-ckpt が本当に効いているかをここで数値で確認する
    # （重みが載っていなければランダム初期値の値が出る）。history には入れない。
    if args.init_ckpt:
        m0 = evaluate(model, val_loader, device)
        print(f"[e00] (init-ckpt 読み込み直後・未学習) val: "
              f"top1={m0['acc_top1']:.3f} exact={m0['acc_exact']:.3f} "
              f"vAUC={m0['value_auc']:.3f}  ※ランダム初期値なら exact は 0.1 未満になる")

    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_trunk:
            # trunk は特徴抽出器として固定 → dropout を切る（linear probe の慣例）
            model.eval()
            model.policy_head.train()
        t0 = time.time()
        loss_pi_sum = loss_v_sum = n_b = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits, v_logit, _ = model(batch)
            loss_pi = multi_positive_ce(logits, batch["label_mh"])
            loss_v = F.binary_cross_entropy_with_logits(v_logit, batch["won"])
            loss = args.policy_weight * loss_pi + args.value_weight * loss_v
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            loss_pi_sum += float(loss_pi.detach())
            loss_v_sum += float(loss_v.detach())
            n_b += 1

        m = evaluate(model, val_loader, device)
        m.update(epoch=epoch,
                 train_loss_pi=loss_pi_sum / n_b, train_loss_v=loss_v_sum / n_b,
                 lr=sched.get_last_lr()[0], sec=round(time.time() - t0, 1))
        history.append(m)
        print(f"[e{epoch:02d}] loss_pi={m['train_loss_pi']:.4f} "
              f"val: top1={m['acc_top1']:.3f} exact={m['acc_exact']:.3f} "
              f"late={m['acc_late']:.3f} vAUC={m['value_auc']:.3f} ({m['sec']}s)")

        ckpt = {"model": model.state_dict(), "model_config": model.config_json(),
                "feat_config": json.dumps(data.feat_cfg), "metrics": m,
                "args": vars(args)}
        torch.save(ckpt, os.path.join(args.out, "last.pt"))
        if args.best_metric == "val_loss":
            score = -m["val_loss"]                       # 小さいほど良いので符号反転
        elif args.best_metric == "value_auc":
            score = m["value_auc"]                       # ★value 単体学習ではこれで選ぶ
        else:
            score = m["acc_exact"]
        if score > best_score:
            best_score = score
            torch.save(ckpt, os.path.join(args.out, "best.pt"))
        if args.snapshot_every and epoch % args.snapshot_every == 0:
            torch.save(ckpt, os.path.join(args.out, f"snap_e{epoch:02d}.pt"))

        with open(os.path.join(args.out, "history.json"), "w") as f:
            json.dump(history, f, indent=1)

    shown = -best_score if args.best_metric == "val_loss" else best_score
    print(f"[done] best val {args.best_metric}={shown:.4f} -> {args.out}/best.pt")
    return history[-1]


def main() -> int:
    ap = argparse.ArgumentParser(description="PTCG-IL Phase1 学習")
    ap.add_argument("--features", required=True, help="featurize.py の .npz")
    ap.add_argument("--out", required=True, help="checkpoint/ログの出力先")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--value-weight", type=float, default=0.5,
                    help="勝敗予測(補助)損失の重み λ_v（§4.1）")
    ap.add_argument("--policy-weight", type=float, default=1.0,
                    help="方策損失の重み λ_π。0 にすると勝敗予測だけを学ぶ"
                         "専用 value ネットになる（探索の評価役を作るとき）")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--data-fraction", type=float, default=1.0,
                    help="train episode の使用割合（学習曲線用: 0.25/0.5/1.0）")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layer", type=int, default=3)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=512,
                    help="FFN 内部次元。慣例は d_model×4（d128→512 / d256→1024）")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--best-metric", choices=("acc_exact", "val_loss", "value_auc"),
                    default="acc_exact",
                    help="best.pt の選定基準。★既定の acc_exact から変えないこと"
                         "（過去の IL / 2a / モデルサイズ A/B の全 run が acc_exact 基準。"
                         "変えると過去 run と比較できなくなる。変更はユーザーの明示的な指示があるときだけ）")
    ap.add_argument("--snapshot-every", type=int, default=0,
                    help="N epoch 毎に snap_eNN.pt を保存（0=無効）")
    ap.add_argument("--init-ckpt", default=None,
                    help="この checkpoint(best.pt) の重みから学習を始める（逐次学習の2段目用）。"
                         "optimizer/scheduler は引き継がず、新しい OneCycle を張る")
    ap.add_argument("--freeze-trunk", action="store_true",
                    help="policy_head 以外を凍結して学習する（転移量の測定用）。"
                         "trunk は dropout も切って特徴抽出器として固定する")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", help="auto/cuda/mps/cpu")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="動作確認モード（512サンプル×1epoch）")
    args = ap.parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
