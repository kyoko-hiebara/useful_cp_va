#!/usr/bin/env python3
import numpy as np
import re
import sys

def read_xyz(filename):
    """
    XYZファイルを読み込み、原子数、ラティス情報、原子種と座標を返す。
    ファイルの2行目に、例 "Lattice="a11 a12 a13 a21 a22 a23 a31 a32 a33"" の形式でラティス情報がある前提。
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    natoms = int(lines[0].strip())
    # 2行目からラティス情報を抽出
    lattice_line = lines[1].strip()
    lattice_match = re.search(r'Lattice="([^"]+)"', lattice_line)
    if lattice_match:
        lattice_str = lattice_match.group(1)
        lattice_vals = list(map(float, lattice_str.split()))
        if len(lattice_vals) != 9:
            raise ValueError(f"{filename} のラティス情報は9個の数値である必要があります。")
        lattice = np.array(lattice_vals).reshape((3, 3))
    else:
        raise ValueError(f"{filename} の2行目からラティス情報が見つかりません。")
    
    atoms = []
    positions = []
    # 3行目以降に原子種と座標（x,y,z）があると仮定
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        atoms.append(parts[0])
        positions.append(list(map(float, parts[1:4])))
    positions = np.array(positions)
    if positions.shape[0] != natoms:
        raise ValueError("原子数と実際の座標数が一致しません。")
    return natoms, lattice, atoms, positions

def write_xyz(filename, natoms, lattice, atoms, positions):
    """
    XYZ形式で出力する。1行目に原子数、2行目にラティス情報（Lattice="..."）、
    その後に各原子種と座標を出力する。
    """
    with open(filename, 'w') as f:
        f.write(f"{natoms}\n")
        # ラティス情報を1行に出力（数値は空白区切り）
        lattice_flat = " ".join(map(str, lattice.flatten()))
        f.write(f'Lattice="{lattice_flat}"\n')
        for atom, pos in zip(atoms, positions):
            # 小数点以下8桁で出力（必要に応じてフォーマットを変更）
            pos_str = " ".join(f"{x:.8f}" for x in pos)
            f.write(f"{atom} {pos_str}\n")

def interpolate_positions(pos1, pos2, lattice, n_images):
    """
    初期座標pos1と最終座標pos2（いずれもshape=(N,3)のndarray）を用いて
    n_images個の中間画像を作成する関数。
    
    補間はまずラティスに基づいてfractional座標に変換し、
    最小画像規則を適用して周期境界条件下で補間後、
    再びCartesian座標に変換して返す。
    
    戻り値は、初期構造～最終構造を含む(n_images+2)個のリスト（各要素はshape=(N,3)のndarray）です。
    """
    # ラティス行列の逆行列（※ここではラティスの行がセルベクトルと仮定）
    lattice_inv = np.linalg.inv(lattice)
    frac1 = pos1.dot(lattice_inv)
    frac2 = pos2.dot(lattice_inv)
    
    # 最小画像規則：差分を -0.5～0.5 の範囲に収める
    dfrac = frac2 - frac1
    dfrac = (dfrac + 0.5) % 1.0 - 0.5
    
    images = []
    # 補間画像は初期と最終を含めて (n_images + 2) 個
    for i in range(n_images + 2):
        t = i / (n_images + 1)  # t=0で初期, t=1で最終
        frac_interp = frac1 + t * dfrac
        # fractional座標を [0,1) に戻す
        frac_interp = frac_interp % 1.0
        cart_interp = frac_interp.dot(lattice)
        images.append(cart_interp)
    return images

def main():
    if len(sys.argv) != 4:
        print("Usage: python script.py initial.xyz final.xyz n_interpolations")
        sys.exit(1)
    
    initial_file = sys.argv[1]
    final_file = sys.argv[2]
    try:
        n_interp = int(sys.argv[3])
    except ValueError:
        print("n_interpolations は整数で指定してください。")
        sys.exit(1)
    
    # ファイルの読み込み
    natoms1, lattice1, atoms1, pos1 = read_xyz(initial_file)
    natoms2, lattice2, atoms2, pos2 = read_xyz(final_file)
    
    if natoms1 != natoms2:
        raise ValueError("2つのファイルの原子数が一致していません。")
    if atoms1 != atoms2:
        raise ValueError("2つのファイルの原子種の並びが一致していません。")
    
    # ラティス情報が異なる場合は警告（NEBでは同一セルが前提の場合が多い）
    if not np.allclose(lattice1, lattice2):
        print("Warning: 2つのファイルでラティス情報が異なっています。初期ファイルのラティスを使用します。")
    lattice = lattice1
    
    # 補間構造の作成（初期～最終を含む）
    images = interpolate_positions(pos1, pos2, lattice, n_interp)
    
    # それぞれの構造をXYZファイルとして出力（例: image0.xyz, image1.xyz, ...）
    for i, pos in enumerate(images):
        out_filename = f"image{i}.xyz"
        write_xyz(out_filename, natoms1, lattice, atoms1, pos)
        print(f"Wrote {out_filename}")

if __name__ == "__main__":
    main()
