import random
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.data import atomic_numbers
from dscribe.descriptors import SOAP
import umap.umap_ as umap
import chemiscope

# --- Ti原子の周囲のN, O 配位割合から色を計算する関数 ---
def compute_coordination_color(atoms, ti_index, cutoff=3.0):
    """
    対象Ti原子（ti_index）の周囲(cutoff内)のN, Oの数をカウントし、
    Nの割合とOの割合からRGBを (frac_O, 0, frac_N) として返す。
    もしNもOもなければグレー(0.5,0.5,0.5)を返す。
    """
    ti_pos = atoms.positions[ti_index]
    count_N = 0
    count_O = 0
    for i, pos in enumerate(atoms.positions):
        if i == ti_index:
            continue
        if np.linalg.norm(pos - ti_pos) < cutoff:
            if atoms[i].symbol == "N":
                count_N += 1
            elif atoms[i].symbol == "O":
                count_O += 1
    total = count_N + count_O
    if total == 0:
        return (0.5, 0.5, 0.5)
    frac_N = count_N / total
    frac_O = count_O / total
    # 色は (frac_O, 0, frac_N) として、Oのみ→赤、Nのみ→青、混合→紫系となる
    return (frac_O, 0.0, frac_N)

# --- RGBタプルを16進数カラー文字列に変換する関数 ---
def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(
        int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)
    )

# --- シミュレーションデータの読み込み ---
all_structures = read("simulation.extxyz", index=":")

# --- 系に含まれる元素の指定 ---
symbols = ["Ti", "N", "O", "H"]
species = [atomic_numbers[s] for s in symbols]

# --- SOAPディスクリプターの設定 (DScribe 2.1.x 用: 位置引数で species, periodic, cutoff, nmax, lmax, sigma) ---
soap = SOAP(species, False, 5.0, 8, 6, 0.5)

# --- 各構造ごとに SOAP と配位色を計算 ---
soap_descriptors = []       # 各構造のSOAP特徴ベクトル
color_list = []             # 各構造の配位色 (RGB タプル)
structures_used = []        # Chemiscope用に採用する構造 (ASE Atoms オブジェクト)

for atoms in all_structures:
    # 構造中にTi原子があるかチェック
    ti_indices = [i for i, atom in enumerate(atoms) if atom.symbol == "Ti"]
    if ti_indices:
        # 存在するならランダムに1つ選ぶ
        selected_index = random.choice(ti_indices)
        descriptor = soap.create(atoms, positions=[selected_index], n_jobs=1)[0]
        col = compute_coordination_color(atoms, selected_index, cutoff=3.0)
    else:
        # Ti原子が存在しない場合：次元数は soap.get_number_of_features() を利用
        descriptor = np.zeros(soap.get_number_of_features())
        col = (0.5, 0.5, 0.5)
    soap_descriptors.append(descriptor)
    color_list.append(col)
    structures_used.append(atoms)

soap_descriptors = np.array(soap_descriptors)

# --- UMAP による次元削減 ---
reducer = umap.UMAP(n_neighbors=5, min_dist=0.05, random_state=42)
embedding = reducer.fit_transform(soap_descriptors)

# --- Matplotlib による UMAP 散布図プロット ---
plt.figure(figsize=(8, 6))
plt.scatter(embedding[:, 0], embedding[:, 1], c=color_list, s=50, edgecolors='k')
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.title("SOAP-UMAP Visualization for ML Potential")
plt.show()

# --- Chemiscope 用データの作成 ---
# プロパティとして UMAP 座標と配位色 (16進数文字列) を設定
umap1 = embedding[:, 0].tolist()
umap2 = embedding[:, 1].tolist()
color_hex = [rgb_to_hex(c) for c in color_list]

properties = {
    "UMAP Dimension 1": {
        "target": "structure",
        "values": umap1,
        "description": "First dimension of UMAP embedding"
    },
    "UMAP Dimension 2": {
        "target": "structure",
        "values": umap2,
        "description": "Second dimension of UMAP embedding"
    },
    "Ti Coordination Color": {
        "target": "structure",
        "values": color_hex,
        "description": "Hex color code representing the Ti coordination environment: blue for Ti-N, red for Ti-O, purple for mixed; grey if no Ti is present."
    },
}

meta = {
    "name": "SOAP UMAP Chemiscope Dataset for ML Potential",
    "description": "Visualization of SOAP descriptors reduced by UMAP. The color of each structure indicates the Ti coordination environment, chosen randomly from available Ti atoms. Grey indicates no Ti present.",
    "authors": ["Your Name"],
    "references": ["Relevant papers / DOI"]
}

# Chemiscope 0.8.3 では frames に ASE の Atoms オブジェクトのリストをそのまま渡すことができます
chemiscope.write_input(
    path="chemiscope_data.json",
    frames=structures_used,
    meta=meta,
    properties=properties
)
