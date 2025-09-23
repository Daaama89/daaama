import matplotlib.pyplot as plt
from havok.build_havok import load_csv_data
from havok.havok_lstm import havok_lstm_one_step_pipeline
from havok.plotting import plot_havok_reconstruction


def main():
    data = load_csv_data(["../dataset/lorenz/lorenz_x_emb.csv"])[0]
    if data.ndim == 1:
        data = data[None, :]

    # # パラメータ
    # dt, delays, rank, num_chaos = 0.001, 100, 15, 1
    # seq_len = 40
    # lstm_units = 64
    # epochs = 50

    # # HAVOK-LSTM パイプライン実行
    # res = havok_lstm_pipeline_pytorch(
    #     data,
    #     dt,
    #     delays,
    #     rank,
    #     num_chaos,
    #     seq_len=seq_len,
    #     lstm_units=lstm_units,
    #     epochs=epochs,
    # )

    # history = res["history"]
    # losses = history["train_loss"]
    # epochs = len(losses)

    # plt.figure(figsize=(6, 4))
    # plt.plot(history["train_loss"], label="Train")
    # plt.plot(history["val_loss"], label="Val")
    # plt.yscale("log")  # 収束の様子を見やすく
    # plt.xlabel("Epoch")
    # plt.ylabel("MSE Loss")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # # 実データ vs HAVOK 予測 vs HAVOK-LSTM 予測
    # fig, axs = plot_havok_reconstruction(
    #     t_rec=res["t_pred"],
    #     V=res["V_test"],
    #     yout_pred=res["yout_pred"],
    #     u_pred=res["u_pred"],
    #     r=rank,
    #     idx_train_end=0,
    # )
    # plt.show()

    # 2) パイプライン実行
    res = havok_lstm_one_step_pipeline(
        data,
        dt=0.001,
        delays=40,
        rank=11,
        num_chaos=1,
        seq_len=40,  # 入力系列長
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        lr=1e-3,
        epochs=50,
        batch_size=32,
        patience=10,
        val_split=0.2,
        device=None,  # 自動選択
    )

    # 結果取り出し
    history = res["history"]
    u_pred = res["u_pred"]
    t_pred = res["t_pred"]
    yout_pred = res["yout_pred"]
    V_test = res["V_test"]
    u_true = res["u_true"]

    # 3) 学習／検証損失のプロット
    plt.figure(figsize=(6, 4))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val   Loss")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 4) 強制項 v_r の比較プロット
    #    テストデータの先頭 seq_len を除いた部分が予測対象
    seq_len = 40
    u_true = u_true.flatten()[seq_len:]
    u_pred = u_pred.flatten()[seq_len:]
    t_true = t_pred[seq_len:]  # t_pred はすでに offset 調整済み
    plt.figure(figsize=(8, 3))
    plt.plot(t_true, u_true, label="True  forcing $v_r$")
    plt.plot(t_true, u_pred, "--", label="Predicted forcing $v_r$")
    plt.xlabel("Time")
    plt.ylabel("$v_r$")
    plt.legend()
    plt.title("Forcing term: True vs One-step LSTM Prediction")
    plt.tight_layout()
    plt.show()

    # 5) モード1 (v1) 再構成の比較
    #    plot_havok_reconstruction を使って真値 vs 再構成値を描画
    fig, axs = plot_havok_reconstruction(
        t_rec=t_pred,
        V=V_test,
        yout_pred=yout_pred,
        u_pred=u_pred,
        r=11,
        idx_train_end=0,  # V_test はすでにテスト期間分
        fig=None,
    )
    plt.show()


if __name__ == "__main__":
    main()
