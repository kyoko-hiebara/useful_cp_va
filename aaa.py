import numpy as np

def read_xyz(filename):
    """
    training_base.xyz 形式のファイルを読み込み、
    原子数、格子ベクトル(a,b,c)、原子名＆座標リストを返す関数
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # 1行目: 原子数
    num_atoms = int(lines[0].strip())

    # 2行目: Lattice=... のパース
    # 例: Lattice="1.4934 -8.2186 0.0 1.4934 8.2186 0.0 0.0 0.0 4.1939"
    #    -> [1.4934, -8.2186, 0.0, 1.4934, 8.2186, 0.0, 0.0, 0.0, 4.1939]
    lattice_line = lines[1].strip()
    # Lattice="..." の中身を取り出す
    # まず "Lattice=\"" とその後の "\"" を除去
    prefix = 'Lattice="'
    start_index = lattice_line.find(prefix) + len(prefix)
    end_index = lattice_line[start_index:].find('"') + start_index
    lattice_str = lattice_line[start_index:end_index]
    # 数値リストとしてパース
    lattice_values = list(map(float, lattice_str.split()))
    # a, b, c ベクトルに分割
    a_vec = np.array(lattice_values[0:3], dtype=float)
    b_vec = np.array(lattice_values[3:6], dtype=float)
    c_vec = np.array(lattice_values[6:9], dtype=float)

    # 3行目以降: 原子の要素名と座標
    atoms = []
    coords = []
    for i in range(num_atoms):
        line = lines[2 + i].strip()
        parts = line.split()
        # parts[0]: 原子名, parts[1..3]: x, y, z
        atoms.append(parts[0])
        xyz = np.array(list(map(float, parts[1:4])), dtype=float)
        coords.append(xyz)

    return num_atoms, a_vec, b_vec, c_vec, atoms, coords

def create_rotation_matrix(a_vec, b_vec):
    """
    汎用的に “aベクトルを x軸に揃える” ような回転行列を作る例。
    ただし右手系を保つために a×b が正しく z軸方向を向くようにする。
    """
    # e1 = a / |a|
    e1 = a_vec / np.linalg.norm(a_vec)
    # e3 = (a × b) / |a × b|
    cross_ab = np.cross(a_vec, b_vec)
    e3 = cross_ab / np.linalg.norm(cross_ab)
    # e2 = e3 × e1
    e2 = np.cross(e3, e1)

    # 回転行列 R を作る
    # ここでは「旧座標 -> 新座標」へ変換する場合を想定。
    # 各列に e1, e2, e3 を並べた行列とする(行単位でも可だが掛け方に注意)
    R = np.column_stack((e1, e2, e3))

    return R

def transform_coordinates(a_vec, b_vec, c_vec, coords):
    """
    格子ベクトル(a,b,c)と各原子の座標を、ある回転(例: aをx軸に)で変換した結果を返す。
    """
    # 回転行列を作成 (例: aを x軸に、かつ右手系)
    R = create_rotation_matrix(a_vec, b_vec)

    # 新しい格子ベクトル
    new_a = R.T @ a_vec  # Rが「旧->新」なので、座標を新系へ行くには v' = R^T v の場合も
    new_b = R.T @ b_vec
    new_c = R.T @ c_vec

    # 各原子の新しい座標
    new_coords = []
    for r in coords:
        r_new = R.T @ r
        new_coords.append(r_new)

    return new_a, new_b, new_c, new_coords

def write_xyz(filename, num_atoms, new_a, new_b, new_c, atoms, new_coords):
    """
    変換後の格子ベクトルと原子座標を "training.xyz" の形式で書き出す
    """
    with open(filename, 'w') as f:
        # 1行目: 原子数
        f.write(f"{num_atoms}\n")
        # 2行目: Lattice="x1 x2 x3 y1 y2 y3 z1 z2 z3"
        lattice_str = (
            f'Lattice="{new_a[0]:.4f} {new_a[1]:.4f} {new_a[2]:.4f} '
            f'{new_b[0]:.4f} {new_b[1]:.4f} {new_b[2]:.4f} '
            f'{new_c[0]:.4f} {new_c[1]:.4f} {new_c[2]:.4f}"'
        )
        f.write(lattice_str + "\n")

        # 原子座標
        for atom, coord in zip(atoms, new_coords):
            x, y, z = coord
            f.write(f"{atom} {x:.4f} {y:.4f} {z:.4f}\n")

def main():
    # 入出力ファイル名
    input_file = "training_base.xyz"
    output_file = "training.xyz"

    # 1) 読み込み
    num_atoms, a_vec, b_vec, c_vec, atoms, coords = read_xyz(input_file)

    # 2) 変換 (例: aを x軸に回転する & 右手系を保つ)
    new_a, new_b, new_c, new_coords = transform_coordinates(a_vec, b_vec, c_vec, coords)

    # 3) 書き出し
    write_xyz(output_file, num_atoms, new_a, new_b, new_c, atoms, new_coords)

if __name__ == "__main__":
    main()
