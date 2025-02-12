#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XYZファイルから格子情報を読み込み、周期境界条件下での水の接触可能表面（solvent accessible surface; SAS）の面積を
グリッド法＋marching cubesにより求め、3次元プロットで可視化するサンプルコードです。

使い方:
    python script.py input.xyz
"""

import numpy as np
import re
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure

def read_xyz(filename):
    """
    XYZファイルを読み込み、原子数、格子情報（3x3行列）、元素記号リスト、原子座標（Cartesian, Å）を返す。
    2行目は、 lattice="x1 x2 x3 y1 y2 y3 z1 z2 z3" の形式とする。
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    natoms = int(lines[0].strip())
    
    # 2行目から格子情報を抽出
    lattice_line = lines[1].strip()
    m = re.search(r'lattice\s*=\s*\"([^\"]+)\"', lattice_line)
    if m:
        lattice_values = list(map(float, m.group(1).split()))
        if len(lattice_values) != 9:
            raise ValueError("lattice情報は9個の数値で指定してください。")
        lattice = np.array(lattice_values).reshape(3, 3)
    else:
        raise ValueError("2行目にlattice情報が見つかりません。")
    
    elements = []
    coords = []
    # 3行目以降は、"Element x y z"となっているとする
    for line in lines[2:]:
        if line.strip() == "":
            continue
        parts = line.split()
        element = parts[0]
        x, y, z = map(float, parts[1:4])
        elements.append(element)
        coords.append([x, y, z])
    coords = np.array(coords)
    return natoms, lattice, elements, coords

def get_effective_radii(elements, probe_radius=1.4):
    """
    各元素に対してvan der Waals半径（Å）を定義し、水プローブ半径を加えた有効半径を返す。
    見つからなければ、デフォルト値1.5 Åを用いる。
    """
    vdw = {
        'H': 1.2,
        'C': 1.7,
        'N': 1.55,
        'O': 1.52,
        'S': 1.8,
        'P': 1.8
    }
    radii = []
    for e in elements:
        r = vdw.get(e, 1.5)
        radii.append(r + probe_radius)
    return np.array(radii)

def compute_accessibility_field(lattice, coords, radii, Ngrid=50):
    """
    格子（周期境界条件）内のfractional座標空間 [0,1]^3 上に3次元グリッドを生成し、
    各グリッド点がいずれかの原子球（有効半径以内）に入っている場合1、そうでなければ0のフィールドを作成する。
    
    ※周期境界条件はfractional座標上で、差を計算後 np.rint() で最小像をとることで考慮する。
    """
    # fractional座標でのグリッドを作成
    f_lin = np.linspace(0, 1, Ngrid)
    fx, fy, fz = np.meshgrid(f_lin, f_lin, f_lin, indexing='ij')
    grid_frac = np.stack([fx, fy, fz], axis=-1)  # shape: (Ngrid, Ngrid, Ngrid, 3)
    
    # 原子座標をfractional座標へ変換
    L_inv = np.linalg.inv(lattice)
    coords_frac = np.dot(coords, L_inv.T)  # shape: (n_atoms, 3)
    
    field = np.zeros((Ngrid, Ngrid, Ngrid), dtype=np.float32)
    
    # 各原子について、周期境界条件を考慮しながら、グリッド点と原子中心との距離を計算
    for i in range(len(coords_frac)):
        f_atom = coords_frac[i]  # 原子のfractional座標
        r_eff = radii[i]
        # 各グリッド点とのfractional差を計算
        diff = grid_frac - f_atom  # shape: (Ngrid, Ngrid, Ngrid, 3)
        # 周期境界条件：差が[-0.5, 0.5]になるように補正
        diff = diff - np.rint(diff)
        # Cartesian座標での差: diff_cart = lattice @ diff
        diff_cart = np.tensordot(diff, lattice, axes=([3],[0]))  # shape: (Ngrid, Ngrid, Ngrid, 3)
        dist = np.linalg.norm(diff_cart, axis=-1)
        # 有効半径以内なら1（内部）とする
        field[dist < r_eff] = 1.0
    return field, f_lin

def compute_surface_area(vertices, faces):
    """
    marching cubesで得られたメッシュ（三角形分割）の各三角形の面積を合算して、総面積を返す。
    vertices: (N,3) のCartesian座標、faces: (M,3) の各三角形頂点インデックス
    """
    triangles = vertices[faces]  # shape: (M, 3, 3)
    vec1 = triangles[:, 1] - triangles[:, 0]
    vec2 = triangles[:, 2] - triangles[:, 0]
    cross_prod = np.cross(vec1, vec2)
    tri_areas = 0.5 * np.linalg.norm(cross_prod, axis=1)
    total_area = tri_areas.sum()
    return total_area

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py input.xyz")
        sys.exit(1)
    filename = sys.argv[1]
    
    # XYZファイル読み込み
    natoms, lattice, elements, coords = read_xyz(filename)
    print("原子数:", natoms)
    print("格子情報（lattice）:\n", lattice)
    
    # 各原子の有効半径（van der Waals半径 + 水プローブ半径）を求める
    radii = get_effective_radii(elements, probe_radius=1.4)
    
    # 3次元グリッド（fractional空間）上で、各点が原子球内部か否かを判定するフィールドを作成
    Ngrid = 50  # 格子分解能（必要に応じて変更可）
    field, f_lin = compute_accessibility_field(lattice, coords, radii, Ngrid=Ngrid)
    
    # marching cubesにより、フィールドの等値面（level=0.5）から面を抽出する
    # グリッドはfractional空間なので、spacingは f_lin[1]-f_lin[0] とする
    spacing = (f_lin[1]-f_lin[0], f_lin[1]-f_lin[0], f_lin[1]-f_lin[0])
    verts_frac, faces, normals, values = measure.marching_cubes(field, level=0.5, spacing=spacing)
    
    # 得られた頂点はfractional座標なので、Cartesian座標へ変換
    verts_cart = np.dot(verts_frac, lattice)
    
    # メッシュから面積を計算
    area = compute_surface_area(verts_cart, faces)
    print("水の接触可能表面積: {:.2f} Å²".format(area))
    
    # 3次元プロット（plot_trisurf）で水の接触可能表面を表示
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(verts_cart[:, 0], verts_cart[:, 1], verts_cart[:, 2],
                           triangles=faces, cmap='viridis', lw=0.5, edgecolor='none', alpha=0.8)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")
    ax.set_title("Water Accessible Surface")
    
    plt.show()

if __name__ == "__main__":
    main()
