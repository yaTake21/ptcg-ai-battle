"""cabt コンペ提出用 submission.tar.gz を作成する。

必須: main.py / cg(ディレクトリ) / deck.csv（コンペ指定の構成）。
任意: 学習エージェント用の追加物が deck_dir にあれば自動同梱する
  （weights.npz / static/ / featurize.py / infer_numpy.py）。
  → IL/RL 系提出dir（submissions/ 配下）ではモデル一式が入る。
使い方（リポジトリ直下から）:
  python3 Imitation_Learning/src/submit/make_submission.py [deck_dir]
既定:
  submissions/Alakazam_IL を対象に submission.tar.gz を作成。
"""
import glob
import os
import sys
import tarfile

# deck_dir 直下にあれば同梱する任意ファイル/ディレクトリ（学習エージェント用）
# ★2026-08-14: weights*.npz をワイルドカードで拾うようにした。
#   シード・アンサンブルの提出物は weights_s42.npz / weights_s0.npz のように
#   複数の重みを持つため、固定名 "weights.npz" だけだと**重みが1つも入らない tar**が
#   静かに出来上がる（実際に踏んだ。599KB の tar が生成された）。
OPTIONAL_EXTRAS = ("static", "featurize.py", "infer_numpy.py")
WEIGHTS_GLOB = "weights*.npz"

# このファイルは Imitation_Learning/src/submit/ 配下（リポジトリ直下 = 3階層上）
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", "..", ".."))


def main():
    deck_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        _ROOT, "submissions", "Alakazam_IL")
    deck_dir = os.path.abspath(deck_dir)
    # 提出dirは submissions/ 配下が規約。裸の名前で渡されたら補完する
    if not os.path.isdir(deck_dir):
        alt = os.path.join(_ROOT, "submissions", os.path.basename(deck_dir))
        if os.path.isdir(alt):
            deck_dir = alt

    main_py = os.path.join(deck_dir, "main.py")
    cg_dir = os.path.join(deck_dir, "cg")
    deck_csv = os.path.join(deck_dir, "deck.csv")
    for p in (main_py, cg_dir, deck_csv):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    out = os.path.join(deck_dir, "submission.tar.gz")

    # __pycache__ など不要物を除外
    def filt(ti):
        base = os.path.basename(ti.name)
        if "__pycache__" in ti.name or base.endswith(".pyc") or base == ".DS_Store":
            return None
        return ti

    with tarfile.open(out, "w:gz") as tar:
        tar.add(main_py, arcname="main.py")
        tar.add(cg_dir, arcname="cg", filter=filt)
        tar.add(deck_csv, arcname="deck.csv")
        # 学習エージェントの追加物（存在するものだけ）
        for name in OPTIONAL_EXTRAS:
            p = os.path.join(deck_dir, name)
            if os.path.exists(p):
                tar.add(p, arcname=name, filter=filt)
        # 重みは複数ありうる（アンサンブル）。1つも無ければ止める
        w = sorted(glob.glob(os.path.join(deck_dir, WEIGHTS_GLOB)))
        if not w:
            raise FileNotFoundError(
                f"{deck_dir} に {WEIGHTS_GLOB} が1つも無い（重み無しの tar を作らない）")
        for p in w:
            tar.add(p, arcname=os.path.basename(p), filter=filt)

    print("created:", out, f"({os.path.getsize(out)} bytes)")
    with tarfile.open(out) as tar:
        for m in tar.getmembers():
            print(f"  {m.size:>8}  {m.name}")


if __name__ == "__main__":
    main()
