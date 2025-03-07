import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.data import atomic_numbers
from dscribe.descriptors import SOAP
import umap.umap_ as umap
import chemiscope

# --- 対象 Ti 原子の周囲の N, O 配位割合から色を計算する関数 ---
def compute_coordination_color(atoms, ti_index, cutoff=3.0):
    """
    対象 Ti 原子（ti_index）の周囲（cutoff 内）の N, O の数をカウントし、
    N の割合と O の割合から RGB を (frac_O, 0, frac_N) として返す。
    もしどちらも 0 ならグレー (0.5,0.5,0.5) を返す。
    """
    ti_pos = atoms.positions[ti_index]
    count_N = 0
    count_O = 0
    for i, pos in enumerate(atoms.positions):
        if i == ti_index:
            continue
        dist = np.linalg.norm(pos - ti_pos)
        if dist < cutoff:
            if atoms[i].symbol == "N":
                count_N += 1
            elif atoms[i].symbol == "O":
                count_O += 1
    total = count_N + count_O
    if total == 0:
        return (0.5, 0.5, 0.5)  # どちらもない場合はグレー
    frac_N = count_N / total
    frac_O = count_O / total
    # N のみ → 青 (0,0,1), O のみ → 赤 (1,0,0), 両方 → 混合 (例：50/50 → (0.5,0,0.5))
    return (frac_O, 0.0, frac_N)

# --- RGB タプルを 16 進数カラー文字列に変換する関数 ---
def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(
        int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)
    )

# --- シミュレーションデータの読み込み ---
# ※ "simulation.extxyz" は各スナップショットの構造を含む Extxyz ファイル
all_structures = read("simulation.extxyz", index=":")

# --- 系に含まれる元素の指定 ---
symbols = ["Ti", "N", "O", "H"]
# DScribe 2.1.x 用に、species は原子番号のリストで渡すと安定する場合が多い
species = [atomic_numbers[s] for s in symbols]

# --- 対象 Ti 原子の指定 ---
selected_ti_index = 0  # extxyz 内で解析対象とする Ti 原子のインデックス

# --- SOAPディスクリプターの設定 ---
# DScribe 2.1.x では位置引数で渡すのが確実です（順番: species, periodic, cutoff, nmax, lmax, sigma）
soap = SOAP(species, False, 5.0, 8, 6, 0.5)

# --- 各スナップショットごとに SOAP と Ti の配位色を計算 ---
soap_descriptors = []       # SOAP の特徴量（各構造に対して）
color_list = []             # 各構造の配位環境に応じた RGB (タプル)
structures_used = []        # Chemiscope に渡す構造（ASE Atoms オブジェクト）

for atoms in all_structures:
    # 対象インデックスが構造内に存在し、かつ対象原子が Ti であるかチェック
    if selected_ti_index >= len(atoms):
        continue
    if atoms[selected_ti_index].symbol != "Ti":
        print(f"Warning: このスナップショット内の原子 {selected_ti_index} は Ti ではありません。")
        continue

    # SOAP ディスクリプター計算：対象 Ti 原子の局所環境（全構造から情報取得）
    descriptor = soap.create(atoms, positions=[selected_ti_index], n_jobs=1)[0]
    soap_descriptors.append(descriptor)

    # Ti の周囲にある N, O の配位環境から色を計算
    col = compute_coordination_color(atoms, selected_ti_index, cutoff=3.0)
    color_list.append(col)
    
    # Chemiscope 用にこの構造を採用
    structures_used.append(atoms)

soap_descriptors = np.array(soap_descriptors)

# --- UMAP による次元削減 ---
# 局所的な差分を強調するため n_neighbors, min_dist を調整（例: n_neighbors=5, min_dist=0.05）
reducer = umap.UMAP(n_neighbors=5, min_dist=0.05, random_state=42)
embedding = reducer.fit_transform(soap_descriptors)

# --- matplotlib による UMAP 散布図プロット ---
# 色は計算した RGB (タプル) をそのまま利用
plt.figure(figsize=(8, 6))
plt.scatter(embedding[:, 0], embedding[:, 1], c=color_list, s=50, edgecolors='k')
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.title("SOAP UMAP for Ti Coordination Environment")
plt.colorbar(plt.cm.ScalarMappable(cmap="coolwarm"), label="dummy")  # カラーバーは参考用（必要に応じて削除）
plt.show()

# --- Chemiscope 用データの作成 ---
# Chemiscope 0.8.3 では、frames には ASE の Atoms オブジェクトのリストをそのまま渡せます。
# properties として、UMAP 座標と配位色（16進数文字列）を設定
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
        "description": "Hex color code derived from Ti coordination environment (blue: Ti-N, red: Ti-O, purple: mixed)"
    },
}

meta = {
    "name": "SOAP UMAP Chemiscope Dataset for ML Potential",
    "description": "Visualization of SOAP descriptors reduced by UMAP. The color of each structure indicates the coordination environment of a selected Ti atom: blue for Ti-N, red for Ti-O, purple for mixed.",
    "authors": ["Your Name"],
    "references": ["Relevant papers / DOI"]
}

# Chemiscope の JSON ファイルを書き出し
chemiscope.write_input(
    path="chemiscope_data.json",
    frames=structures_used,
    meta=meta,
    properties=properties
)
