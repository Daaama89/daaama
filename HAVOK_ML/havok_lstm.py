import numpy as np
import torch
import torch.nn as nn
from scipy.signal import StateSpace, lsim
from torch.utils.data import DataLoader, TensorDataset

from .build_havok import build_havok_model


# def create_lstm_sequences(u, seq_len):
#     """
#     強制項時系列 u = [u1, u2, ..., uT]を入力データX、目標データyに変換

#     ---各種パラメータ---
#     u: (T,) or (1, T)
#     seq_len: LSTM に渡すシーケンス長
#     X: (N, seq_len, 1)
#     y: (N, 1)
#     """
#     u = u.flatten()
#     N = len(u) - seq_len
#     X = np.stack([u[i : i + seq_len] for i in range(N)], axis=0)
#     y = np.array([u[i + seq_len] for i in range(N)])
#     # PyTorch Tensor に変換
#     X_t = torch.from_numpy(X).float().unsqueeze(-1)  # (N, seq_len, 1)
#     y_t = torch.from_numpy(y).float().unsqueeze(-1)  # (N, 1)
#     return X_t, y_t


def create_multichannel_sequences(V_modes, u_force, seq_len):
    """
    V_modes: ndarray, shape=(r-1, T)
    seq_len: int
    Returns X:(N,seq_len,r-1), y:(N,1)  # 予測対象は forcing mode v_r
    """
    data = V_modes.T  # (T, r-1)
    u = u_force.flatten()  # (T, )
    N = len(u) - seq_len

    X = np.stack([data[i : i + seq_len] for i in range(N)], axis=0)
    y = np.array([u[i + seq_len] for i in range(N)])

    return (
        torch.from_numpy(X).float(),  # (N, seq_len, r-1)
        torch.from_numpy(y).float().unsqueeze(-1),  # (N, 1)
    )


# ============================================================================================== #


# --- 2) モデル定義 を input_size 可変に ---
class LSTMForcingPredictor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # 最終ステップの出力だけ使う


# ============================================================================================== #


# --- 3) build_and_train_lstm を入力 X,y 受け取り型に ---
def build_and_train_lstm(
    X, y, hidden_units=50, epochs=50, lr=1e-3, batch_size=32, val_split=0.2, device=None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # DataLoader
    dataset = TensorDataset(X, y)
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # モデル
    input_size = X.shape[-1]
    model = LSTMForcingPredictor(input_size, hidden_units).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= n_train
        history["train_loss"].append(train_loss)

        # Val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item() * xb.size(0)
        val_loss /= n_val
        history["val_loss"].append(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train {train_loss:.3e} | Val {val_loss:.3e}")

    return model, history


# ============================================================================================== #


def predict_future_forcing(model, u_initial, predict_len, seq_len, device=None):
    """
    LSTM を使って、u_initial の末端から predict_len 先の強制項を逐次予測。
    ----
    model       : 学習済み LSTM モデル (torch.nn.Module)
    u_initial   : (T0,) または (1, T0) の Tensor or ndarray
    predict_len : 予測ステップ数
    seq_len     : シーケンス長
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # 入力を Tensor 化しデバイスへ
    if not isinstance(u_initial, torch.Tensor):
        u_initial = torch.tensor(u_initial, dtype=torch.float32)
    u_initial = u_initial.to(device).flatten()

    # リスト化して逐次予測用のシーケンスを保持
    seq = u_initial.detach().cpu().tolist()
    preds = []

    for _ in range(predict_len):
        x = (
            torch.tensor(seq[-seq_len:], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(-1)
            .to(device)
        )
        with torch.no_grad():
            u_next = model(x).item()
        preds.append(u_next)
        seq.append(u_next)

    return np.array(preds, dtype=np.float32)


def havok_lstm_pipeline_pytorch(
    data,
    dt,
    delays,
    rank,
    num_chaos,
    seq_len=40,
    lstm_units=64,
    epochs=100,
    lr=1e-3,
    batch_size=64,
):
    # 1) HAVOK モデルを構築・シミュレーション
    A, B, V_full, V_test, yout, u_test, t_test = build_havok_model(
        data, dt, delays, rank, num_chaos
    )

    # 2) LSTM の学習
    N = V_full.shape[1]
    split_idx = int(N * 0.8)
    V_modes_train = V_full[: rank - num_chaos, :split_idx]
    u_train = V_full[rank - num_chaos, :split_idx]
    X, y = create_multichannel_sequences(V_modes_train, u_train, seq_len)

    model, history = build_and_train_lstm(
        X, y, hidden_units=lstm_units, epochs=epochs, lr=lr, batch_size=batch_size
    )

    # 3) テスト領域での強制項予測
    u_initial = u_test.flatten()[:seq_len]
    predict_len = len(u_test.flatten()) - seq_len
    u_pred = predict_future_forcing(model, u_initial, predict_len, seq_len)

    # 4) HAVOK再構成
    r = rank
    x0 = V_test[: r - num_chaos, 0]
    sys = StateSpace(A, B, np.eye(r - num_chaos), np.zeros((r - num_chaos, num_chaos)))
    t_pred = t_test[seq_len:]
    _, yout_pred, _ = lsim(sys, u_pred.reshape(-1, 1), t_pred - t_pred[0], x0)

    return {
        "A": A,
        "B": B,
        "V_test": V_test,
        "yout": yout,
        "u_test": u_test,
        "u_pred": u_pred,
        "t_test": t_test,
        "t_pred": t_pred,
        "yout_pred": yout_pred,
        "lstm_model": model,
        "history": history,
    }


# --- データ生成: マルチチャネル入力, １ステップ先予測 ---
def create_multichannel_one_step(V_modes, u_force, seq_len):
    """
    V_modes: ndarray (r-1, T), u_force: ndarray (T,)
    seq_len: 入力長 → 予測は「次の 1 ステップ v_r」
    """
    data = V_modes.T  # shape (T, r-1)
    u = u_force.flatten()
    N = len(u) - seq_len
    X = np.stack([data[i : i + seq_len] for i in range(N)], axis=0)  # (N, seq_len, r-1)
    y = np.array([u[i + seq_len] for i in range(N)])  # (N,)
    return (torch.from_numpy(X).float(), torch.from_numpy(y).float().unsqueeze(-1))


# --- モデル定義: 多層 LSTM + Dropout ---
class OneStepLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        h = out[:, -1, :]  # 最終時刻の隠れ状態
        h = self.drop(h)
        return self.fc(h)  # (batch,1)


# --- 学習関数に Early-Stopping を追加 ---
def train_one_step_lstm(
    X,
    y,
    hidden_size=64,
    num_layers=2,
    dropout=0.2,
    lr=1e-3,
    epochs=200,
    batch_size=32,
    val_split=0.2,
    patience=10,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Dataset / Split
    dataset = TensorDataset(X, y)
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = OneStepLSTM(
        input_size=X.shape[-1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    wait = 0
    history = {"train_loss": [], "val_loss": []}

    for ep in range(1, epochs + 1):
        # --- train ---
        model.train()
        tl = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item() * xb.size(0)
        tl /= n_train

        # --- val ---
        model.eval()
        vl = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                vl += loss_fn(pred, yb).item() * xb.size(0)
        vl /= n_val

        history["train_loss"].append(tl)
        history["val_loss"].append(vl)

        # Early-Stopping
        if vl < best_val:
            best_val = vl
            wait = 0
            best_state = model.state_dict()
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {ep}")
                break

        if ep == 1 or ep % 10 == 0:
            print(f"Epoch {ep:03d} | Train {tl:.3e} | Val {vl:.3e}")

    # 最良モデルに戻す
    model.load_state_dict(best_state)
    return model, history


# --- パイプライン統合 ---
def havok_lstm_one_step_pipeline(
    data,
    dt,
    delays,
    rank,
    num_chaos,
    seq_len=40,
    hidden_size=64,
    num_layers=2,
    dropout=0.2,
    lr=1e-3,
    epochs=50,
    batch_size=32,
    patience=10,
    val_split=0.2,
    device=None,
):
    # (1) HAVOK モデル構築
    A, B, V_full, V_test, yout, u_test, t_test = build_havok_model(
        data, dt, delays, rank, num_chaos
    )

    # (2) 訓練データ準備
    idx_train = int(0.8 * V_full.shape[1])
    V_modes_train = V_full[: rank - num_chaos, :idx_train]
    u_train = V_full[rank - num_chaos, :idx_train]
    X_train, y_train = create_multichannel_one_step(V_modes_train, u_train, seq_len)

    # (3) One-step モデル学習
    model, history = train_one_step_lstm(
        X_train,
        y_train,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        val_split=val_split,
        patience=patience,
        device=device,
    )

    # ——— ここを再帰から「直接テスト領域回帰」へ書き換え ———
    # テスト領域のスライディングウィンドウを作成
    idx_test = int(0.9 * V_full.shape[1])
    V_modes_test = V_full[: rank - num_chaos, idx_train:idx_test]
    u_test_force = V_full[rank - num_chaos, idx_train:idx_test]
    X_test, y_test = create_multichannel_one_step(V_modes_test, u_test_force, seq_len)

    # テスト領域での予測
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        u_pred = model(X_test.to(device)).cpu().numpy().flatten()
    # テストの真値
    u_true = y_test.numpy().flatten()
    t_true = t_test[seq_len:]  # シーケンス長だけオフセット

    # (4) HAVOK 再構成
    x0 = V_test[: rank - num_chaos, 0]
    sys = StateSpace(
        A, B, np.eye(rank - num_chaos), np.zeros((rank - num_chaos, num_chaos))
    )
    t_pred = t_true - t_true[0]
    _, yout_pred, _ = lsim(sys, u_pred.reshape(-1, 1), t_pred, x0)

    return {
        "model": model,
        "history": history,
        "u_pred": u_pred,
        "u_true": u_true,
        "t_true": t_true,
        "t_pred": t_pred,
        "yout_pred": yout_pred,
        "V_test": V_test,
        "X_train": X_train,
        "X_test": X_test,
    }
