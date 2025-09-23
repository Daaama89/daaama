# scripts/run_latent_havok.py
"""
DAE を学習し、潜在系列 z_t に直接 HAVOK を適用するパイプライン
- 学習 → 潜在抽出 → HAVOK 同定 → 保存 → 可視化/指標
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from havok.havok_kAE import (
    DAE_HAVOK,
    make_dataloader,
    train_dae_havok,
    encode_array_with_dae,
    extract_K,
    build_latent_havok,
    save_koopman_npz,
    save_havok_npz,
)
from havok.build_havok import load_csv_data
from havok.havok_KAE_metrics import antisymmetry_report, spectral_radius
from havok.plotting import (
    plot_matrix_and_vector_with_shared_colorbar,
    plot_timeseries_HAVOK_with_slices,
    plot_havok_reconstruction,
    plot_attractor_vertical_with_activation_slices,
)


def _ensure_modes_time(V_like: np.ndarray, r: int) -> np.ndarray:
    """V を [r, T] に揃える（どちらかの軸が r のはず）"""
    V_like = np.asarray(V_like)
    if V_like.ndim != 2:
        raise ValueError("V must be 2D")
    if V_like.shape[0] == r:
        return V_like
    if V_like.shape[1] == r:
        return V_like.T
    raise ValueError(f"V's shape {V_like.shape} does not match r={r} on any axis")


def build_parser():
    p = argparse.ArgumentParser(description="DAE + HAVOK on latent z")
    p.add_argument("--csv", nargs="+", required=True)
    p.add_argument("--seq_len", type=int, default=20)
    p.add_argument("--latent_dim", type=int, default=10)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cpu")

    # HAVOK params
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--delays", type=int, default=100)
    p.add_argument("--rank", type=int, default=15)
    p.add_argument("--num_chaos", type=int, default=1)

    # viz
    p.add_argument("--threshold", type=float, default=0.002)
    p.add_argument("--min_jump", type=int, default=1)

    # save
    p.add_argument("--out_dir", type=str, default="./artifacts")
    p.add_argument("--prefix", type=str, default="latent")
    return p


def main(argv=None):
    parser = build_parser()
    if argv is None:
        import sys

        if "ipykernel" in sys.modules and not any(a == "--csv" for a in sys.argv[1:]):
            argv = ["--csv", "../dataset/lorenz/lorenz_x_emb.csv"]
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    # 1) data
    data_list = load_csv_data(args.csv)
    ts = data_list[0]
    if ts.ndim == 1:
        ts = ts[:, None]
    print("Loaded TS:", ts.shape)

    # 2) dataloader + model + train
    loader = make_dataloader(ts, args.seq_len)
    model = DAE_HAVOK(ts.shape[1], args.latent_dim).to(device)
    print("[1/5] Training DAE_HAVOK ...")
    train_dae_havok(model, loader, lr=args.lr, epochs=args.epochs, alpha=args.alpha)

    # 3) latent
    print("[2/5] Encoding to latent ...")
    z = encode_array_with_dae(ts, model, device)  # (T, r)
    print("latent z:", z.shape)

    # 保存（Koopman 行列自体も記録しておくと後解析に便利）
    K = extract_K(z, model)
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

    # 4) HAVOK on latent
    print("[3/5] Building HAVOK on latent ...")
    rnk = args.rank or z.shape[1]
    A, B, V_full, V_test, yout, u_test, t_test = build_latent_havok(
        z, dt=args.dt, delays=args.delays, rank=rnk, num_chaos=args.num_chaos
    )
    h_path = f"{args.out_dir}/{args.prefix}_havok_d{args.delays}_r{rnk}.npz"
    save_havok_npz(
        A,
        B,
        h_path,
        V_full=V_full,
        V_test=V_test,
        yout=yout,
        u_test=u_test,
        t_test=t_test,
        meta=dict(dt=args.dt, delays=args.delays, rank=rnk, num_chaos=args.num_chaos),
    )
    print("Saved HAVOK ->", h_path)

    # 5) viz & metrics
    print("[4/5] Visualization ...")
    plot_matrix_and_vector_with_shared_colorbar(A, B)

    V_plot = _ensure_modes_time(V_test, rnk)  # [r, T]
    t_plot = t_test - t_test[0]
    u_plot = u_test[:1, :]  # [1, T]（num_chaos>1 の場合は先頭を採用）

    # plot_timeseries_HAVOK_with_slices(
    #     label="Latent HAVOK (test)",
    #     data_original=V_plot,
    #     data=yout,
    #     u=u_plot,
    #     t=t_plot,
    #     r=rnk,
    #     threshold=args.threshold,
    #     min_jump_dist=args.min_jump,
    # )

    # plot_attractor_vertical_with_activation_slices(
    #     V_test=V_plot,
    #     yout=yout,
    #     u_test=u_plot,
    #     t_test=t_plot,
    #     threshold=args.threshold,
    #     min_jump_dist=args.min_jump,
    #     label="Latent HAVOK Attractor (test)",
    # )

    # fig, axs = plot_havok_reconstruction(
    #     t_rec=t_plot,
    #     V=V_plot,
    #     yout_pred=yout,
    #     u_pred=u_plot.flatten(),
    #     r=rnk,
    #     idx_train_end=0,
    # )
    # plt.show()

    # metrics
    print("[5/5] Metrics ...")
    print("rho(A) =", spectral_radius(A))
    print(antisymmetry_report(A))


if __name__ == "__main__":
    main()
