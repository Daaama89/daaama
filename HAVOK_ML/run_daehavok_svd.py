# scripts / run_havok_dae.py
"""
Entry point for DAE-HAVOK execution:
- 学習, 潜在抽出, Koopman/HAVOK 同定, 予測, プロット
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from havok.havok_dae import (
    DAE_HAVOK,
    make_dataloader,
    train_dae_havok,
    encode_csv_with_dae,
    extract_K,
    compute_residual,
    build_residual_havok,
    hierarchical_predict,
    encode_array_with_dae,
)
from havok.build_havok import load_csv_data
from havok.plotting import (
    plot_matrix_and_vector_with_shared_colorbar,
    plot_timeseries_HAVOK_with_slices,
    plot_havok_reconstruction,
    plot_attractor_vertical_with_activation_slices,
)
from havok.havok_dae import (
    save_koopman_npz,
    save_havok_npz,
    load_koopman_npz,
    load_havok_npz,
)
from havok.havok_metrics import (
    antisymmetry_report,
    spectral_radius,
    koopman_residual_stats,
)


def ensure_modes_time(V_like: np.ndarray, r: int) -> np.ndarray:
    """
    V を [r, T] 形状（モード×時間）に揃える。
    想定: V_like のいずれかの次元が r。

    """
    V_like = np.asarray(V_like)
    if V_like.ndim != 2:
        raise ValueError("V must be 2D array")
    if V_like.shape[0] == r:
        return V_like
    if V_like.shape[1] == r:
        return V_like.T
    raise ValueError(f"V's shape {V_like.shape} does not match r={r} on any axis")


def build_parser():
    p = argparse.ArgumentParser(description="DAE-HAVOK Pipeline")
    p.add_argument("--csv", nargs="+", required=True)
    p.add_argument("--seq_len", type=int, default=20)
    p.add_argument("--latent_dim", type=int, default=10)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--havok_q", type=int, default=50)
    p.add_argument("--havok_r", type=int, default=None)  # 使わないなら削ってOK
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--threshold", type=float, default=0.002)
    p.add_argument("--min_jump", type=int, default=1)
    p.add_argument("--device", type=str, default="cpu")

    # ★ ここからは1回だけ定義
    p.add_argument("--dt", type=float, default=1.0)  # サンプリング間隔
    p.add_argument("--havok_rank", type=int, default=None)  # SVD保持モード数
    p.add_argument("--num_chaos", type=int, default=1)  # 強制項モード数
    p.add_argument("--out_dir", type=str, default="./artifacts")
    p.add_argument("--prefix", type=str, default="run")
    return p


def main(argv=None):
    parser = build_parser()

    if argv is None:
        import sys

        # Jupyter実行時で、--csv が無ければデフォルトを補う
        if "ipykernel" in sys.modules and not any(a == "--csv" for a in sys.argv[1:]):
            argv = ["--csv", "../dataset/lorenz/lorenz_x_emb.csv"]

    args = parser.parse_args(argv)

    # デバイス
    device = torch.device(args.device)

    # データ読み込み
    # args.csv = ["../dataset/lorenz/lorenz_x_emb.csv"]
    data_list = load_csv_data(args.csv)
    ts = data_list[0]
    if ts.ndim == 1:
        ts = ts[:, None]
    print("Loaded TS shape:", ts.shape)

    # DataLoader と モデル
    loader = make_dataloader(ts, args.seq_len)
    model = DAE_HAVOK(ts.shape[1], args.latent_dim).to(device)

    # 学習
    print("[1/6] Training DAE_HAVOK ...")
    train_dae_havok(model, loader, lr=args.lr, epochs=args.epochs, alpha=args.alpha)

    # 潜在系列
    print("[2/6] Encoding to latent ...")
    # latent_seq = encode_csv_with_dae(args.csv, model, device)
    latent_seq = encode_array_with_dae(ts, model, device)
    print("Latent seq shape:", latent_seq.shape)

    # K 抽出 & 残差
    print("[3/6] Extracting K and residuals ...")
    K = extract_K(latent_seq, model)
    print("K shape:", K.shape)
    residual_seq = compute_residual(latent_seq, K)
    print("Residual seq shape:", residual_seq.shape)

    # 残差にHAVOK同定
    print("[4/6] Building residual HAVOK ...")
    # rank が未指定なら r(=residualの次元)を使う（= data.shape[0]）
    havok_rank = args.havok_rank or residual_seq.shape[1]
    A_r, B_r, V_full, V_test, yout, u_test, t_test = build_residual_havok(
        residual_seq,
        dt=args.dt,
        delays=args.havok_q,
        rank=havok_rank,
        num_chaos=args.num_chaos,
    )
    print(
        f"Residual HAVOK A_r:{A_r.shape}, B_r:{B_r.shape}, "
        f"V_full:{V_full.shape}, V_test:{V_test.shape}, "
        f"yout:{yout.shape}, u_test:{u_test.shape}, t_test:{t_test.shape}"
    )

    # K を保存
    k_path = f"{args.out_dir}/{args.prefix}_koopman_K.npz"
    save_koopman_npz(
        K,
        k_path,
        meta=dict(
            input_dim=ts.shape[1],
            latent_dim=args.latent_dim,
            epochs=args.epochs,
            lr=args.lr,
            alpha=args.alpha,
        ),
    )
    print("Saved K ->", k_path)

    # HAVOK を保存（A,B と併せて V_test, yout, u_test, t_test も付けられます）
    havok_rank = args.havok_rank or residual_seq.shape[1]
    A_r, B_r, V_full, V_test, yout, u_test, t_test = build_residual_havok(
        residual_seq,
        dt=args.dt,
        delays=args.havok_q,
        rank=havok_rank,
        num_chaos=args.num_chaos,
    )
    h_path = f"{args.out_dir}/{args.prefix}_havok_q{args.havok_q}_r{havok_rank}.npz"
    save_havok_npz(
        A_r,
        B_r,
        h_path,
        V_full=V_full,
        V_test=V_test,
        yout=yout,
        u_test=u_test,
        t_test=t_test,
        meta=dict(
            dt=args.dt, delays=args.havok_q, rank=havok_rank, num_chaos=args.num_chaos
        ),
    )
    print("Saved HAVOK ->", h_path)

    # --- 以降の可視化は本家戻り値に合わせて形状整合 ---
    # V_test は [r, N_test]、u_test は [num_chaos, N_test]、t_test は [N_test]
    # あなたの可視化関数の前提：
    #   plot_timeseries_HAVOK_with_slices: data_original=[r,T], data=[T,r], u=[1,T], t=[T]
    #   plot_attractor_vertical_with_activation_slices: V_test=[r,T], yout=[T,r-1], u_test=[1,T], t_test=[T]

    # 5.1 A,B のヒートマップ
    plot_matrix_and_vector_with_shared_colorbar(A_r, B_r)

    K, kmeta = load_koopman_npz("./artifacts/run_koopman_K.npz")
    hav = load_havok_npz("./artifacts/run_havok_q50_r10.npz")
    A, B = hav["A"], hav["B"]

    print("rho(A) =", spectral_radius(A))
    print(antisymmetry_report(A))

    # もし潜在系列 z を持っていれば（保存していない場合は再エンコード）
    # stats = koopman_residual_stats(z, K)
    # print(stats)


if __name__ == "__main__":
    main()
