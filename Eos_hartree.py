import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 3次Birch–Murnaghan EOSのエネルギー式（単位：energy in hartree, volume in Å³）
def BM_energy(V, E0, V0, B0, B0p):
    # η = (V0/V)^(2/3) - 1
    eta = (V0 / V)**(2/3) - 1
    return E0 + (9 * V0 * B0 / 16) * (B0p * eta**3 + (6 - 4*(V0 / V)**(2/3)) * eta**2)

# 圧力式 (エネルギー微分から導出)
def BM_pressure(V, V0, B0, B0p):
    return (3 * B0 / 2) * (((V0 / V)**(7/3)) - ((V0 / V)**(5/3))) * \
           (1 + (3/4) * (B0p - 4) * (((V0 / V)**(2/3)) - 1))

# データの読み込み (Energy_volume.dat の1列目：エネルギー[hartree]、2列目：体積[Å³]、3列目：圧力[bar])
data = np.loadtxt('Energy_volume.dat')
energy = data[:, 0]    # energy in hartree
volume = data[:, 1]    # volume in Å³
pressure = data[:, 2]  # pressure in bar

# フィッティング (初期値: E0は最小エネルギー、V0は最小エネルギーに対応する体積、B0は0.5 hartree/Å³、B0'は4)
p0 = [min(energy), volume[np.argmin(energy)], 0.5, 4]
popt, pcov = curve_fit(BM_energy, volume, energy, p0=p0)
E0_fit, V0_fit, B0_fit, B0p_fit = popt

# 体積弾性率の換算：B0は hartree/Å³ 単位
# 1 hartree/Å³ = 27.211386 eV/Å³, 1 eV/Å³ = 160.21766208 GPa → 1 hartree/Å³ ≒ 27.211386*160.21766208 GPa
B0_GPa = B0_fit * 27.211386 * 160.21766208

print("フィッティング結果:")
print("E0 =", E0_fit, "hartree")
print("V0 =", V0_fit, "Å³")
print("B0 =", B0_fit, "hartree/Å³  →  {:.2f} GPa".format(B0_GPa))
print("B0' =", B0p_fit)

# フィッティング曲線用の体積範囲を設定
V_fit = np.linspace(min(volume), max(volume), 200)
E_fit = BM_energy(V_fit, *popt)

# BM_pressureで算出される圧力は hartree/Å³ 単位なので、barに変換
# 1 hartree/Å³ = 27.211386 eV/Å³, 1 eV/Å³ = 1.60218e6 bar
pressure_conversion = 27.211386 * 1.60218e6
P_fit = BM_pressure(V_fit, V0_fit, B0_fit, B0p_fit) * pressure_conversion

# プロット (ツイン軸を用いてエネルギーと圧力を1枚の図に表示)
fig, ax1 = plt.subplots(figsize=(8, 6))

# 左軸：エネルギー (hartree)
ax1.scatter(volume, energy, color='blue', label='Energy data (hartree)')
ax1.plot(V_fit, E_fit, color='blue', linestyle='-', label='BM energy fit')
ax1.set_xlabel('Volume (Å³)')
ax1.set_ylabel('Energy (hartree)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# 右軸：圧力 (bar)
ax2 = ax1.twinx()
ax2.scatter(volume, pressure, color='red', marker='x', label='Pressure data (bar)')
ax2.plot(V_fit, P_fit, color='red', linestyle='--', label='BM pressure fit')
ax2.set_ylabel('Pressure (bar)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# タイトルに体積弾性率を表示
plt.title('Birch–Murnaghan EOS Fit\nBulk modulus = {:.2f} GPa'.format(B0_GPa))

# 両軸の凡例を合わせる
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')

plt.tight_layout()
plt.show()
