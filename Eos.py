import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 3次Birch–Murnaghan EOSのエネルギー式
def BM_energy(V, E0, V0, B0, B0p):
    # η = (V0/V)^(2/3) - 1
    eta = (V0/V)**(2/3) - 1
    return E0 + (9 * V0 * B0 / 16) * (B0p * eta**3 + (6 - 4*(V0/V)**(2/3)) * eta**2)

# 対応する圧力式 (エネルギー微分から)
def BM_pressure(V, V0, B0, B0p):
    # 第3式： P(V) = (3B0/2)[(V0/V)^(7/3) - (V0/V)^(5/3)] * {1 + (3/4)(B0'-4)[(V0/V)^(2/3)-1]}
    return (3 * B0 / 2) * (((V0/V)**(7/3)) - ((V0/V)**(5/3))) * (1 + (3/4)*(B0p - 4)*(((V0/V)**(2/3))-1))

# データの読み込み（ファイルは同じディレクトリにある前提）
data = np.loadtxt('Energy_volume.dat')
energy = data[:, 0]    # エネルギー [eV]
volume = data[:, 1]    # 体積 [Å³]
pressure = data[:, 2]  # 圧力 [bar]

# エネルギーデータのフィッティング
# 初期値: E0は最小エネルギー, V0は最小エネルギーに対応する体積, B0は0.5 eV/Å³程度, B0'は通常4前後
p0 = [min(energy), volume[np.argmin(energy)], 0.5, 4]
popt, pcov = curve_fit(BM_energy, volume, energy, p0=p0)
E0_fit, V0_fit, B0_fit, B0p_fit = popt

# 体積弾性率 B0 の単位変換
# BM_energy式での B0 の単位は eV/Å³ であり、1 eV/Å³ ≈ 160.21766208 GPa
B0_GPa = B0_fit * 160.21766208

print("フィッティング結果:")
print("E0 =", E0_fit, "eV")
print("V0 =", V0_fit, "Å³")
print("B0 =", B0_fit, "eV/Å³  →  {:.2f} GPa".format(B0_GPa))
print("B0' =", B0p_fit)

# フィッティング曲線を描くため、体積の範囲を細かく設定
V_fit = np.linspace(min(volume), max(volume), 200)
E_fit = BM_energy(V_fit, *popt)

# 圧力は BM_pressure で eV/Å³ 単位になるので bar に変換
# 1 eV/Å³ = 1.60218e6 bar (1 eV/Å³ = 1.60218e11 Pa, 1 bar = 1e5 Pa)
P_fit = BM_pressure(V_fit, V0_fit, B0_fit, B0p_fit) * 1.60218e6

# プロット (ツイン軸を用いて1枚の図にエネルギーと圧力のフィッティング結果を表示)
fig, ax1 = plt.subplots(figsize=(8, 6))

# 左側軸：エネルギー
ax1.scatter(volume, energy, color='blue', label='Energy data (eV)')
ax1.plot(V_fit, E_fit, color='blue', linestyle='-', label='BM energy fit')
ax1.set_xlabel('Volume (Å³)')
ax1.set_ylabel('Energy (eV)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# 右側軸：圧力
ax2 = ax1.twinx()
ax2.scatter(volume, pressure, color='red', marker='x', label='Pressure data (bar)')
ax2.plot(V_fit, P_fit, color='red', linestyle='--', label='BM pressure fit')
ax2.set_ylabel('Pressure (bar)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# タイトルに体積弾性率の結果を表示
plt.title('Birch–Murnaghan EOS Fit\nBulk modulus = {:.2f} GPa'.format(B0_GPa))

# 凡例を両軸のハンドルを合わせて表示
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')

plt.tight_layout()
plt.show()
