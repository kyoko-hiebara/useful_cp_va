import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from dscribe.descriptors import SOAP
import umap.umap_ as umap
import json

# --- Chemiscope用にASEのAtomsオブジェクトを辞書に変換する関数 ---
def get_structure_dict(atoms):
    """
    ChemiscopeのJSONフォーマットに合わせるため、ASEのAtomsオブジェクトから
    必要な情報（positions, numbers, cell）を抽出して辞書にします。
    """
    structure = {}
    # 座標（リスト形式）
    structure["positions"] = atoms.positions.tolist()
    # 元素番号（原子番号）リスト
    structure["numbers"] = [atom.number for atom in atoms]
    # 周期境界条件が有効ならcell情報を追加
    try:
        cell = atoms.get_cell()
        # cellがNoneでなければリストに変換
        structure["cell"] = cell.tolist()
    except Exception:
        structure["cell"] = None
    return structure

# --- シミュレーションデータの読み込み ---
structures = read("simulation.extxyz", index=":")

# --- 系に含まれる元素の指定 ---
species = ["Ti", "N", "O", "H"]

# --- 対象Ti原子の指定 ---
selected_ti_index = 0  # extxyz中の対象Ti原子のインデックス（必要に応じて変更）

# --- SOAPディスクリプターの設定 ---
soap = SOAP(
    species=species,
    rcut=5.0,
    nmax=8,
    lmax=6,
    sigma=0.5,
    periodic=False
)

# --- 高度な配位数計算の関数 ---
def coordination_number(atoms, ti_index, r_cut=3.0, d=0.2, water_oh_cutoff=1.2):
    """
    対象Ti原子と周囲原子との間で、滑らかな重み付けによる配位数を計算します。
    
    ・r_cut: 評価するカットオフ距離
    ・d: スムージングパラメータ（カットオフ近傍での重みの変化幅）
    ・water_oh_cutoff: H原子が水分子内のOに属しているか判定するためのO-H結合距離の閾値
    
    H原子の場合、同一水分子の酸素原子がTi原子からr_cut以内にある場合は重複寄与を除外します。
    """
    ti_pos = atoms.positions[ti_index]
    cn = 0.0
    for i, pos in enumerate(atoms.positions):
        if i == ti_index:
            continue
        r = np.linalg.norm(pos - ti_pos)
        if r > r_cut:
            continue
        # 距離rがカットオフに近づくと重みが0に滑らかに減少するシグモイド関数
        weight = 1.0 / (1.0 + np.exp((r - r_cut) / d))
        
        # H原子の場合、同じ水分子に属しているかチェック
        if atoms[i].symbol == "H":
            skip_H = False
            for j, other in enumerate(atoms):
                if other.symbol == "O":
                    if np.linalg.norm(atoms.positions[j] - pos) < water_oh_cutoff:
                        if np.linalg.norm(atoms.positions[j] - ti_pos) < r_cut:
                            skip_H = True
                            break
            if skip_H:
                weight = 0.0
        cn += weight
    return cn

# --- 各スナップショットごとにSOAPディスクリプターと配位数を計算 ---
soap_descriptors = []   # 各スナップショットでのSOAP特徴ベクトル
coord_numbers = []      # 高度な配位数
structures_chemiscope = []  # Chemiscope用に変換した構造情報

for atoms in structures:
    # 指定した原子番号が存在し、その原子がTiであるか確認
    if selected_ti_index >= len(atoms):
        continue
    if atoms[selected_ti_index].symbol != "Ti":
        print(f"Warning: スナップショット内の原子 {selected_ti_index} はTiではありません。")
        continue

    # SOAPディスクリプター計算（対象原子のみ）
    descriptor = soap.create(atoms, positions=[selected_ti_index], n_jobs=1)[0]
    soap_descriptors.append(descriptor)
    
    # 高度な配位数の計算
    cn = coordination_number(atoms, selected_ti_index, r_cut=3.0, d=0.2, water_oh_cutoff=1.2)
    coord_numbers.append(cn)
    
    # Chemiscope用の構造辞書に変換
    structures_chemiscope.append(get_structure_dict(atoms))

soap_descriptors = np.array(soap_descriptors)

# --- UMAPによる次元削減 ---
reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(soap_descriptors)

# --- UMAPプロット（確認用） ---
plt.figure(figsize=(8, 6))
sc = plt.scatter(embedding[:, 0], embedding[:, 1], c=coord_numbers, cmap="viridis")
plt.colorbar(sc, label="Coordination Number")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.title("UMAP Projection of SOAP Descriptors for Selected Ti Atom")
plt.show()

# --- Chemiscope用データの作成 ---
data = {
    "structures": structures_chemiscope,
    "properties": {
        # 配位数をスカラーデータとして登録
        "Coordination Number": {"values": coord_numbers}
    },
    "descriptors": {
        # UMAPによる2次元埋め込みを散布図の座標として登録
        "UMAP": {"values": embedding.tolist()}
    },
    "metadata": {
        "description": "UMAP projection of SOAP descriptors and advanced coordination numbers for a selected Ti atom"
    }
}

# JSONファイルとして書き出し
with open("chemiscope_data.json", "w") as f:
    json.dump(data, f, indent=2)
