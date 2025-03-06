import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.geometry import distance_array
import umap

# --- パラメータ設定 ---
filename = "data.extxyz"       # 読み込むextxyzファイルのパス
target_index = 0               # 特徴量を計算する対象原子のインデックス（例：最表面のTi原子）
coordination_cutoff = 3.5      # 配位数計算用のカットオフ距離 (Angstrom)
max_radius = 5.0               # ヒストグラムで使用する最大距離 (Angstrom)
nbins = 50                     # ヒストグラムのビン数

# --- extxyzファイルから全フレーム読み込み ---
frames = read(filename, index=':')

# --- 各フレームで特徴量（距離ヒストグラム）と配位数を計算 ---
features = []    # 各フレームの特徴ベクトル（ヒストグラム）
coord_nums = []  # 各フレームの配位数

# ヒストグラム用のビンの定義
bins = np.linspace(0, max_radius, nbins+1)

for frame in frames:
    pos = frame.get_positions()
    target_pos = pos[target_index]
    
    # 周期境界条件がある場合はASEのdistance_arrayを利用
    if frame.get_pbc().any():
        dists = distance_array([target_pos], pos, cell=frame.get_cell(), pbc=frame.get_pbc())[0]
    else:
        dists = np.linalg.norm(pos - target_pos, axis=1)
    
    # 自身との距離は除く
    mask = dists > 1e-8
    dists = dists[mask]
    
    # 配位数: coordination_cutoff以内にある原子の数をカウント
    coord = np.sum(dists < coordination_cutoff)
    coord_nums.append(coord)
    
    # 特徴量: ヒストグラム（隣接原子との距離分布）を計算
    hist, _ = np.histogram(dists, bins=bins)
    features.append(hist)

features = np.array(features)
coord_nums = np.array(coord_nums)

# --- UMAPで次元削減 ---
reducer = umap.UMAP(n_components=2, random_state=42)
embedding = reducer.fit_transform(features)

# --- プロット ---
plt.figure(figsize=(8,6))
sc = plt.scatter(embedding[:,0], embedding[:,1], c=coord_nums, cmap='viridis')
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP projection of atomic environment features")
plt.colorbar(sc, label="Coordination Number")
plt.show()
