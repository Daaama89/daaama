import random
import numpy as np
import matplotlib.pyplot as plt
from environment import Environment, Action

class Agent():
    def __init__(self, env):
        self.actions = env.actions

    def policy(self, state):
        return random.choice(self.actions)


def plot_maze_with_trajectory(grid, trajectory, rewards):
    """エージェントの軌跡をMatplotlibで図示（セルの区切りと中心を通る軌跡）"""
    grid = np.array(grid)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xticks(np.arange(grid.shape[1] + 1), minor=False)
    ax.set_yticks(np.arange(grid.shape[0] + 1), minor=False)
    ax.set_xticks(np.arange(grid.shape[1]) + 0.5, minor=True)
    ax.set_yticks(np.arange(grid.shape[0]) + 0.5, minor=True)
    ax.grid(which="major", color="black", linestyle='-', linewidth=2)
    ax.grid(which="minor", color="gray", linestyle='--', linewidth=1)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    
    # 描画用の色設定
    colors = {0: "white", 1: "green", -1: "red", 9: "black"}
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            cell_value = grid[row, col]
            ax.add_patch(plt.Rectangle((col, grid.shape[0] - 1 - row), 1, 1, 
                                       color=colors.get(cell_value, "white"), edgecolor='black'))
            text = str(cell_value)
            if cell_value in rewards:
                text += f"\n({rewards[cell_value]})"  # 報酬を追加
            ax.text(col + 0.5, grid.shape[0] - 1 - row + 0.5, text,
                    ha='center', va='center', fontsize=10, color='black')
    
    # 軌跡をプロット（セルの中心を通るように調整）
    trajectory = np.array(trajectory)
    y_coords = grid.shape[0] - 1 - trajectory[:, 0] + 0.5  # y軸を反転させ、中心を通るように調整
    x_coords = trajectory[:, 1] + 0.5  # x軸も中心を通るように調整
    ax.plot(x_coords, y_coords, marker="o", color="blue", linestyle="-")
    
    plt.show()


def main():
    # 迷路の設定 (数字はセル番号を表す)
    grid = [
        [9, 10, 11, 12],
        [5, 6, 7, 8],
        [1, 2, 3, 4]
    ]
    
    # 各セルの報酬設定（例: 12番は +1、8番は -1、6番は障害物）
    rewards = {
        12: 1,  # ゴールの報酬
        8: -1,  # 罠の報酬
        6: None  # 障害物（通過不可）
    }
    
    env = Environment(grid, rewards)  # 環境に報酬設定を渡す
    agent = Agent(env)

    # 10エピソードのシミュレーション
    for i in range(10):
        state = env.reset()
        total_reward = 0
        done = False
        trajectory = [(state.row, state.column)]  # エージェントの移動履歴を (row, column) 形式で保存

        while not done:
            action = agent.policy(state)
            next_state, reward, done = env.step(action)

            # next_state が None の場合はスキップ
            if next_state is None:
                continue
            
            # 6番セルが障害物の場合、移動をキャンセル
            if grid[next_state.row][next_state.column] == 6:
                continue  # 通過せずにスキップ

            total_reward += reward
            trajectory.append((next_state.row, next_state.column))  # 軌跡を (row, column) 形式で記録
            state = next_state
        
        # エージェントの移動履歴をセル番号で表示
        trajectory_numbers = [grid[row][col] for row, col in trajectory]
        print("Episode {}: Agent gets {} reward.".format(i, total_reward))
        print("Trajectory: " + " → ".join(map(str, trajectory_numbers)))
        plot_maze_with_trajectory(grid, trajectory, rewards)  # Matplotlibで軌跡を図示


if __name__ == "__main__":
    main()
