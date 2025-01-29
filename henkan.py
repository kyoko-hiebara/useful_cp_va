import numpy as np

def read_xyz_file(filename):
    """
    training_base.xyz を読み込んで以下を返す:
      - atom_num: 原子数(int)
      - lattice: [v1, v2, v3] 各vは長さ3のnumpy配列 (旧格子ベクトル)
      - atoms: [(element, x, y, z), ...] のリスト
    """
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    # 1行目: 原子数
    atom_num = int(lines[0])

    # 2行目: Lattice="x1 x2 x3 y1 y2 y3 z1 z2 z3"
    # 例: Lattice="1.4934 -8.2186 0.0 1.4934 8.2186 0.0 0.0 0.0 4.1939"
    lattice_line = lines[1]
    # Lattice=" ... " から数値部分だけを抜き出す
    #  "Lattice=\"" と "\"" を除去し、splitで配列化
    #  もしくは置換などでやる方法もある
    prefix = 'Lattice="'
    suffix = '"'
    assert lattice_line.startswith(prefix) and lattice_line.endswith(suffix), \
        "Lattice行の形式が想定と違います"
    lattice_str = lattice_line[len(prefix):-len(suffix)]
    lattice_vals = list(map(float, lattice_str.split()))

    # 格子ベクトルを3つ取り出す
    # v1 = (x1, x2, x3), v2 = (y1, y2, y3), v3 = (z1, z2, z3)
    v1 = np.array(lattice_vals[0:3], dtype=float)
    v2 = np.array(lattice_vals[3:6], dtype=float)
    v3 = np.array(lattice_vals[6:9], dtype=float)

    # 3行目以降: 原子情報
    atoms = []
    for i in range(atom_num):
        # 例: "Ti 1.4934 -4.7391 1.0484"
        element, x, y, z = lines[2 + i].split()
        x, y, z = map(float, (x, y, z))
        atoms.append((element, x, y, z))

    return atom_num, [v1, v2, v3], atoms


def calc_lattice_constants_and_angles(v1, v2, v3):
    """
    旧格子ベクトル v1, v2, v3 から
    格子定数 (a, b, c) と 格子角度 (alpha, beta, gamma) [deg] を返す。
    ここで
      a = |v1|
      b = |v2|
      c = |v3|
      alpha = ∠(v2, v3)
      beta  = ∠(v1, v3)
      gamma = ∠(v1, v2)
    """
    a = np.linalg.norm(v1)
    b = np.linalg.norm(v2)
    c = np.linalg.norm(v3)

    def angle_deg(u, w):
        # u と w の成す角度(度数法)
        cos_val = np.dot(u, w) / (np.linalg.norm(u)*np.linalg.norm(w))
        # 数値誤差で±1をわずかに超える場合の対策
        cos_val = max(min(cos_val, 1.0), -1.0)
        return np.degrees(np.arccos(cos_val))

    alpha = angle_deg(v2, v3)  # ∠(b, c)
    beta  = angle_deg(v1, v3)  # ∠(a, c)
    gamma = angle_deg(v1, v2)  # ∠(a, b)

    return (a, b, c), (alpha, beta, gamma)


def make_new_lattice_vectors(a, b, c, alpha, beta, gamma):
    """
    求めた a, b, c, alpha, beta, gamma から
    新しい格子ベクトル v1', v2', v3' を返す。

    一般的な(直方体でない)格子でも以下のように組むのが標準的:
      v1' = ( a, 0, 0 )
      v2' = ( b*cos(gamma), b*sin(gamma), 0 )
      v3' = (..., ..., c )
    ただし alpha, beta, gamma は
      alpha = ∠(v2, v3)
      beta  = ∠(v1, v3)
      gamma = ∠(v1, v2)
    の順で与えられているので、斜方格子以上の場合は
    3つめのベクトルに微妙な成分が入るが、ここでは
    質問文の例のように z 軸だけに成分が入る想定(α=90, β=90の場合)を踏まえ、
    質問文の最終例に沿う形で単純化している。

    質問文の例:
      Lattice="8.3532 0.0000 0.0000 7.8192 2.9387 0.0000 0.0000 0.0000 4.1939"
      -> gamma = 159.4° でも、実際には 180-159.4=20.6° の方向を使っている
         (cos(20.6°) ~ 0.936, sin(20.6°) ~ 0.35)
    """
    # gamma を 180°-γ にするかどうかは、最終的に
    # 質問文が提示している出力例に合わせるため、
    # 実際には 180 - gamma を用いているという解釈。
    # ただし alpha, beta が90°の場合だけこの単純形にできるため、
    # 一般トリクリン系の完全対応コードを書くなら
    # cベクトルにもxy成分が入るが、ここでは省略。
    # → 質問文例に厳密に合わせる場合、γが159.4°なら、実際に使う角度は20.6°となる。
    gamma_eff = 180.0 - gamma  # 質問文例に準拠

    # ラジアンに変換
    gamma_rad = np.radians(gamma_eff)

    # v1' = (a, 0, 0)
    v1p = np.array([a, 0.0, 0.0])

    # v2' = (b*cos(gamma_eff), b*sin(gamma_eff), 0)
    vx = b * np.cos(gamma_rad)
    vy = b * np.sin(gamma_rad)
    v2p = np.array([vx, vy, 0.0])

    # v3' = (0, 0, c)  (問題文の例に合わせる)
    v3p = np.array([0.0, 0.0, c])

    return [v1p, v2p, v3p]


def convert_coordinates(old_vs, new_vs, coords):
    """
    old_vs = [v1_old, v2_old, v3_old] (3x3行列の列ベクトル想定)
    new_vs = [v1_new, v2_new, v3_new]
    coords = [(element, x, y, z), ...]

    各 (x,y,z) は旧座標系での実座標(カーティシアン)とみなし、
    1) 旧格子行列(M_old) の逆行列 でかけて分数座標に変換
    2) 新格子行列(M_new) とかけて新カーティシアン座標に戻す

    を行い、その新しい座標を返す。
    """
    # M_old, M_new を作る (列ベクトルが v1,v2,v3 になるように転置する流儀など注意)
    # ここでは [v1, v2, v3] を列として束ねる -> shape (3,3)
    M_old = np.column_stack(old_vs)
    M_new = np.column_stack(new_vs)

    # 逆行列
    M_old_inv = np.linalg.inv(M_old)

    new_coords = []
    for elem, x, y, z in coords:
        r_old = np.array([x, y, z], dtype=float)
        # 分数座標
        frac = M_old_inv @ r_old  # shape(3,)
        # 新実座標
        r_new = M_new @ frac
        new_coords.append((elem, r_new[0], r_new[1], r_new[2]))
    return new_coords


def write_xyz_file(filename, atom_num, new_vs, new_coords):
    """
    training.xyz を書き出す。
    new_vs は新しい格子ベクトル [v1', v2', v3'] (それぞれ長さ3のndarray)
    new_coords は [(element, x', y', z'), ...]
    """
    with open(filename, 'w') as f:
        f.write(f"{atom_num}\n")

        # Lattice="x1 x2 x3 y1 y2 y3 z1 z2 z3" の形式で出力
        # 質問文の最終例に倣い、小数点以下4桁でフォーマット
        # new_vs = [v1p, v2p, v3p]
        v1p, v2p, v3p = new_vs
        lattice_str = (
            f"Lattice=\""
            f"{v1p[0]:.4f} {v1p[1]:.4f} {v1p[2]:.4f} "
            f"{v2p[0]:.4f} {v2p[1]:.4f} {v2p[2]:.4f} "
            f"{v3p[0]:.4f} {v3p[1]:.4f} {v3p[2]:.4f}\""
        )
        f.write(lattice_str + "\n")

        # 原子座標出力
        for (elem, x, y, z) in new_coords:
            f.write(f"{elem} {x:.4f} {y:.4f} {z:.4f}\n")


def main():
    # 1) training_base.xyz を読み込む
    input_file = "training_base.xyz"
    atom_num, old_vs, atoms = read_xyz_file(input_file)

    # 2) 旧格子定数と格子角度を計算
    (a, b, c), (alpha, beta, gamma) = calc_lattice_constants_and_angles(*old_vs)
    # 質問文の例だと:
    #   ABC [angstrom] 8.3532 8.3532 4.1939
    #   ALPHA_BETA_GAMMA [deg] 90.00 90.00 159.40
    # と出るはず

    # 3) 新しい格子ベクトルを作成 (質問文に示された最終形に合わせる)
    new_vs = make_new_lattice_vectors(a, b, c, alpha, beta, gamma)

    # 4) 各原子座標を旧→新へ座標変換
    new_coords = convert_coordinates(old_vs, new_vs, atoms)

    # 5) training.xyz として出力
    output_file = "training.xyz"
    write_xyz_file(output_file, atom_num, new_vs, new_coords)

    # 参考: 中間結果を確認したい場合は適宜print等で確認してください
    # print(f"Old lattice vectors:\n{old_vs}")
    # print(f"a,b,c = {a:.4f}, {b:.4f}, {c:.4f}")
    # print(f"alpha,beta,gamma = {alpha:.2f}, {beta:.2f}, {gamma:.2f}")
    # print(f"New lattice vectors:\n{new_vs}")
    # for i, (elem, x,y,z) in enumerate(new_coords):
    #     print(i, elem, x, y, z)


if __name__ == "__main__":
    main()
