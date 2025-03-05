import re
import numpy as np
import matplotlib.pyplot as plt

def parse_extxyz(filename):
    """
    extxyzファイルを読み込み、各構成のエネルギーと
    各構成における原子ごとの力からRMS値を計算して返す関数です。
    """
    energies = []
    forces_rms = []
    
    with open(filename, 'r') as f:
        while True:
            header = f.readline()
            if not header:
                break  # ファイルの終端
            try:
                natoms = int(header.strip())
            except ValueError:
                break
            # コメント行からエネルギーの値を抽出
            comment = f.readline().strip()
            energy_match = re.search(r'energy\s*=\s*([-\d.]+)', comment)
            if energy_match:
                energy = float(energy_match.group(1))
            else:
                energy = None
            energies.append(energy)
            
            # 各原子のデータ行（座標や力など）が続く
            forces = []
            for _ in range(natoms):
                line = f.readline().strip().split()
                # 例：1列目が原子種、その後にx,y,z, そしてfx,fy,fzと想定
                # 力の成分は末尾の3要素として抽出
                if len(line) < 7:
                    continue  # 不足している場合はスキップ
                fx, fy, fz = map(float, line[-3:])
                forces.append([fx, fy, fz])
            forces = np.array(forces)
            if forces.size > 0:
                # 各原子の力のノルムの二乗平均平方根（RMS）を計算
                rms = np.sqrt(np.mean(np.sum(forces**2, axis=1)))
            else:
                rms = None
            forces_rms.append(rms)
    
    return np.array(energies), np.array(forces_rms)

# ファイル名は適宜変更してください
dft_energy, dft_force = parse_extxyz('dft.extxyz')
mlff_energy, mlff_force = parse_extxyz('mlff.extxyz')

# エネルギーのパリティプロット（DFT vs MLFF）
plt.figure(figsize=(6,6))
plt.scatter(dft_energy, mlff_energy, label='データ点')
# y=xの線を描画して理想の一致を確認
min_val = min(np.min(dft_energy), np.min(mlff_energy))
max_val = max(np.max(dft_energy), np.max(mlff_energy))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
plt.xlabel('DFT Energy')
plt.ylabel('MLFF Energy')
plt.title('エネルギーのパリティプロット')
plt.legend()
plt.grid(True)
plt.show()

# 各構成ごとのエネルギー誤差（MLFF - DFT）のプロット
energy_diff = mlff_energy - dft_energy
plt.figure(figsize=(6,4))
plt.plot(energy_diff, 'o-', label='Energy Error')
plt.xlabel('構成番号')
plt.ylabel('エネルギー差 (MLFF - DFT)')
plt.title('各構成ごとのエネルギー誤差')
plt.grid(True)
plt.legend()
plt.show()

# 各構成ごとのRMS力誤差（MLFF - DFT）のプロット
force_diff = mlff_force - dft_force
plt.figure(figsize=(6,4))
plt.plot(force_diff, 'o-', color='purple', label='Force RMS Error')
plt.xlabel('構成番号')
plt.ylabel('RMS力差 (MLFF - DFT)')
plt.title('各構成ごとのRMS力誤差')
plt.grid(True)
plt.legend()
plt.show()
