from ase.io import read
from ase.eos import EquationOfState
from ase.calculators.emt import EMT
import numpy as np
import matplotlib.pyplot as plt

# 単一構造を読み込む（ファイル内の最初の構造）
atoms = read('bulk.extxyz', index=0)

# Calculatorを設定（ここではEMTを使用）
atoms.calc = EMT()

# スケーリング因子の範囲（例えば0.94～1.06の範囲で10段階）
scale_factors = np.linspace(0.94, 1.06, 10)

volumes = []
energies = []

for s in scale_factors:
    # オリジナル構造をコピーし、一様にスケーリング
    scaled_atoms = atoms.copy()
    # セルと原子位置を同時にスケール（scale_atoms=Trueで原子位置も自動でスケーリング）
    scaled_atoms.set_cell(scaled_atoms.get_cell() * s, scale_atoms=True)
    
    # 各構造の体積とエネルギーを計算
    volumes.append(scaled_atoms.get_volume())
    energies.append(scaled_atoms.get_potential_energy())

volumes = np.array(volumes)
energies = np.array(energies)

# EquationOfStateによるフィッティング
eos = EquationOfState(volumes, energies)
v0, e0, B = eos.fit()

print("Equilibrium Volume (v0): {:.3f} Å³".format(v0))
print("Minimum Energy (e0): {:.3f} eV".format(e0))
print("Bulk Modulus (B): {:.3f} eV/Å³".format(B))

# 単位変換：1 eV/Å³ ≒ 160.21766208 GPa
B_GPa = B * 160.21766208
print("Bulk Modulus (B): {:.3f} GPa".format(B_GPa))

# フィッティング結果のEoS曲線をプロット
volumes_fit = np.linspace(volumes.min(), volumes.max(), 100)
energies_fit = eos.calc_energy(volumes_fit)

plt.figure()
plt.plot(volumes, energies, 'o', label='Calculated data')
plt.plot(volumes_fit, energies_fit, '-', label='EoS fit')
plt.xlabel('Volume (Å³)')
plt.ylabel('Energy (eV)')
plt.legend()
plt.show()
