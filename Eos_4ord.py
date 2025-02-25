import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 4次のBirch–Murnaghan EOSエネルギー式 (エネルギー単位: hartree, 体積単位: Å³)
def BM_energy_4th(V, E0, V0, B0, B0p, B0pp):
    # x = (V0/V)^(2/3)  ,  η = x - 1
    x = (V0 / V)**(2/3)
    eta = x - 1
    # 4次項を追加： ((B0'-4)^2 + B0''*V0 - 12)/8 * eta^4
    return E0 + (9 * V0 * B0 / 16) * (B0p * eta**3 + (6 - 4*x) * eta**2 + (((B0p - 4)**2 + B0pp * V0 - 12) / 8) * eta**4)

# 4次のBirch–Murnaghan EOS圧力式 (エネルギー微分から導出)
def BM_pressure_4th(V, V0, B0, B0p, B0pp):
    x = (V0 / V)**(2/3)
    eta = x - 1
    return (3 * B0 / 2) * (((V0 / V)**(7/3)) - ((V0 / V)**(5/3))) * \
           (1 + (3/4) * (B0p - 4) * eta + (3/16) * (((B0p - 4)**2 + B0pp * V0)) * eta**2)

# データ読み込み
# Energy_volume.dat の各列：1列目 エネルギー[hartree]、2列目 体積[Å³]、3列目 圧力[bar]
data = np.loadtxt('Energy_volume.dat')
energy = data[:, 0]    # エネルギー [hartree]
volume = data[:, 1]    # 体積 [Å³]
pressure = data[:, 2]  # 圧力 [bar]

# フィッティングの初期値
# E0: 最小エネルギー, V0: 最小エネルギーに対応する体積,
# B0: 0.5 hartree/Å³ 程度, B0': 4, B0'': 0 (初期値として)
p0 = [min(energy), volume[np.argmin(energy)], 0.5, 4, 0]

# curve_fit により4次BM EOSでフィッティング
popt, pcov = curve_fit(BM_energy_4th, volume, energy, p0=p0)
E0_fit, V0_fit, B0_fit, B0p_fit, B0pp_fit = popt

# 体積弾性率 B0 の換算：B0は hartree/Å³ 単位
# 1 hartree/Å³ = 27.211386 eV/Å³, 1 eV/Å³ ≒ 160.21766208 GPa → 1 hartree/Å³ ≒ 4359 GPa
conv_B0 = 27.211386 * 160.21766208
B0_GPa = B0_fit * conv_B0

print("フィッティング結果:")
print("E0    =", E0_fit, "hartree")
print("V0    =", V0_fit, "Å³")
print("B0    =", B0_fit, "hartree/Å³  →  {:.2f} GPa".format(B0_GPa))
print("B0'   =", B0p_fit)
print("B0''  =", B0pp_fit)

# フィッティング曲線描画用に体積の範囲を設定
V_fit = np.linspace(min(volume), max(volume), 200)
E_fit = BM_energy_4th(V_fit, *popt)

# BM_pressure_4th で算出される圧力は hartree/Å³ 単位 → bar に変換
# 1 hartree/Å³ = 27.211386*1.60218e6 bar ≒ 4.36e7 bar
conv_P = 27.211386 * 1.60218e6
P_fit = BM_pressure_4th(V_fit, V0_fit, B0_fit, B0p_fit, B0pp_fit) * conv_P

# プロット (ツイン軸を用いてエネルギーと圧力のフィッティング結果を1枚の図に表示)
fig, ax1 = plt.subplots(figsize=(8, 6))

# 左軸：エネルギー (hartree)
ax1.scatter(volume, energy, color='blue', label='Energy data (hartree)')
ax1.plot(V_fit, E_fit, color='blue', linestyle='-', label='BM 4th order energy fit')
ax1.set_xlabel('Volume (Å³)')
ax1.set_ylabel('Energy (hartree)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# 右軸：圧力 (bar)
ax2 = ax1.twinx()
ax2.scatter(volume, pressure, color='red', marker='x', label='Pressure data (bar)')
ax2.plot(V_fit, P_fit, color='red', linestyle='--', label='BM 4th order pressure fit')
ax2.set_ylabel('Pressure (bar)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# タイトルに体積弾性率（換算後）を表示
plt.title('4th Order Birch–Murnaghan EOS Fit\nBulk modulus = {:.2f} GPa'.format(B0_GPa))

# 両軸の凡例を合わせる
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

plt.tight_layout()
plt.show()
