import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1. 計算条件と物理パラメータの設定
L = 10.0      # 空間の範囲 (-5.0 から 5.0)
T = 30.0       # シミュレーションを行う全時間
D = 1.0       # 拡散係数

Nx = 150      # 空間の分割数
Nt = 20000      # 時間の分割数

dx = L / Nx
dt = T / Nt

# 差分法の安定性条件 (CFL条件: α <= 0.5) の確認
alpha = D * dt / (dx**2)
if alpha > 0.5:
    raise ValueError(f"警告: 計算が不安定です (alpha = {alpha:.3f} > 0.5)。")

# 空間グリッドと初期条件の生成
x = np.linspace(-L/2, L/2, Nx+1)
u_initial = np.exp(-10 * x**2)  # 初期状態: 中央に急峻な山

# 2. グラフ描画の初期設定
fig, ax = plt.subplots(figsize=(8, 5))
line, = ax.plot(x, u_initial, color='darkblue', linewidth=2)
title_text = ax.set_title('', fontsize=12)

ax.set_xlim(-L/2, L/2)
ax.set_ylim(-0.1, 1.1)
ax.set_xlabel("Position (x)")
ax.set_ylabel("Concentration (c)")
ax.grid(True, linestyle='--')

# 3. アニメーション更新関数の定義
u = u_initial.copy()

def update(frame):
    global u
    # 1フレームごとに2ステップ時間を発展させて描画を高速化
    for _ in range(2):
        u_next = u.copy()
        # 境界条件は両端固定 (u=0)
        u_next[1:Nx] = u[1:Nx] + alpha * (u[2:Nx+1] - 2*u[1:Nx] + u[0:Nx-1])
        u = u_next
    
    current_time = frame * 2 * dt
    line.set_ydata(u)
    title_text.set_text(f"1D Diffusion Simulation")
    return line, title_text

fps = 100            # 1秒あたりのフレーム数
steps = Nt + 1        # 総フレーム数
interval_ms = 1000 / fps                    # 1コマあたりのミリ秒

# 4. アニメーションの実行と表示
# frames: 描画する総フレーム数, interval: 更新間隔（ミリ秒）
anim = FuncAnimation(fig, update, frames=steps, interval=interval_ms, blit=True)

savefile = './mp4/diffusion_eq_ani.mp4'
anim.save(savefile, writer='ffmpeg', fps=fps, extra_args=['-r', '30'])

plt.show()