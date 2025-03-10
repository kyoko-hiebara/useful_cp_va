import numpy as np
import matplotlib.pyplot as plt
from larch import Interpreter
from larch.xafs.feff8l import feff8l

# Larch のインタープリタを作成
larch = Interpreter()

# FEFF8l を実行
# ※ 'feff.inp' が入力ファイル名です。実行結果として 'chi.dat' 等のファイルが生成される前提です。
feff8l(larch.symtable, filename='feff.inp')

# FEFF8l の実行後、出力されたデータファイル（例: 'chi.dat'）からデータを読み込む
# 読み込むファイル名は実際の出力に合わせてください
data = np.loadtxt('chi.dat')
k = data[:, 0]      # k 値 (Å⁻¹)
chi = data[:, 1]    # chi(k)

# k³·chi(k) の値を計算
chi_k3 = k**3 * chi

# グラフを描画 (横軸: k [Å⁻¹], 縦軸: k³·chi(k))
plt.figure()
plt.plot(k, chi_k3, marker='o', linestyle='-', label=r'$k^3\chi(k)$')
plt.xlabel(r'$k\ (\AA^{-1})$')
plt.ylabel(r'$k^3\chi(k)$')
plt.title('FEFF8l 実行結果')
plt.legend()
plt.grid(True)
plt.show()

# k と chi(k) の値を 2 列のテキストファイルとして保存
output = np.column_stack((k, chi))
np.savetxt('feff8l_output.dat', output, fmt='%12.6f', header='k chi(k)')
