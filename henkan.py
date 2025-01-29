import numpy as np
import math

def read_xyz_with_lattice(filename):
    """
    training_base.xyz を読み込み、
    戻り値:
      num_atoms: 原子数（int）
      old_cell: 3x3 の numpy 配列（列が格子ベクトル）
      atoms: [(element, x, y, z), ...] のリスト (座標はfloat, 要素はstr)
    """
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    # 1行目: 原子数
    num_atoms = int(lines[0])

    # 2行目: Lattice="x1 x2 x3 y1 y2 y3 z1 z2 z3"
    # 例: Lattice="1.4934 -8.2186 0.0 1.4934 8.2186 0.0 0.0 0.0 4.1939"
    lattice_line = lines[1]
    # "Lattice=" を取り除き、数字部分だけ抜き取る
    # 形式: Lattice="... ...
    lattice_str = lattice_line.split('=')[1].strip()  # "..." となる想定
    lattice_str = lattice_str.strip('"')  # "" を除去
    vals = lattice_str.split()           # [x1, x2, x3, y1, y2, y3, z1, z2, z3]
    vals = list(map(float, vals))

    # old_cell を 3x3 (列が a, b, c) に変形
    # a = (x1, x2, x3), b = (y1, y2, y3), c = (z1, z2, z3)
    old_cell = np.array([
        [vals[0], vals[3], vals[6]],
        [vals[1], vals[4], vals[7]],
        [vals[2], vals[5], vals[8]]
    ], dtype=float)

    # 3行目以降: 原子情報
    atoms = []
    # num_atoms 個だけ読む
    for i in range(num_atoms):
        # 例: Ti 1.4934 -4.7391 1.0484
        line = lines[2 + i]
        parts = line.split()
        elem = parts[0]
        x, y, z = map(float, parts[1:4])
        atoms.append((elem, x, y, z))

    return num_atoms, old_cell, atoms

def calc_lattice_parameters(cell):
    """
    3x3 行列 (列が格子ベクトル) から、
    A, B, C と α, β, γ (度数) を返す。
      cell[:,0] = aベクトル
      cell[:,1] = bベクトル
      cell[:,2] = cベクトル
    """
    a_vec = cell[:, 0]
    b_vec = cell[:, 1]
    c_vec = cell[:, 2]

    A = np.linalg.norm(a_vec)
    B = np.linalg.norm(b_vec)
    C = np.linalg.norm(c_vec)

    # cosθ = (u·v)/(|u||v|)
    def angle_deg(u, v):
        dot_ = np.dot(u, v)
        norm_ = np.linalg.norm(u)*np.linalg.norm(v)
        # 計算上の誤差があるので、-1 ~ 1 にクリップ
        cost = np.clip(dot_/norm_, -1.0, 1.0)
        return math.degrees(math.acos(cost))

    alpha = angle_deg(b_vec, c_vec)  # b-c 間
    beta  = angle_deg(a_vec, c_vec)  # a-c 間
    gamma = angle_deg(a_vec, b_vec)  # a-b 間

    return A, B, C, alpha, beta, gamma

def make_new_cell(a, b, c, alpha, beta, gamma):
    """
    A, B, C, alpha, beta, gamma (度数) から
    新しい 3x3 セル行列 (列が a', b', c') を生成。
    質問文の出力例に倣い、もし角度 > 90 なら 180 - 角度 で再定義する簡易処理を行う。
    """
    # もし 90 度超える場合は補角へ (質問文例が 159.4->20.6 のようにしているため)
    if alpha > 90.0:
        alpha = 180.0 - alpha
    if beta > 90.0:
        beta = 180.0 - beta
    if gamma > 90.0:
        gamma = 180.0 - gamma

    # ラジアンに変換
    alpha_r = math.radians(alpha)
    beta_r  = math.radians(beta)
    gamma_r = math.radians(gamma)

    # トリクリニック系の標準的な構築
    ax = a
    ay = 0.0
    az = 0.0

    bx = b*math.cos(gamma_r)
    by = b*math.sin(gamma_r)
    bz = 0.0

    cx = c*math.cos(beta_r)
    # 分母が 0 に近い場合の対策として微小量のクリップ等する手もある
    cy = c*(math.cos(alpha_r) - math.cos(gamma_r)*math.cos(beta_r)) / max(math.sin(gamma_r), 1e-12)
    cz = c*math.sqrt(
        1.0
        + 2.0*math.cos(alpha_r)*math.cos(beta_r)*math.cos(gamma_r)
        - math.cos(alpha_r)**2
        - math.cos(beta_r)**2
        - math.cos(gamma_r)**2
    ) / max(math.sin(gamma_r), 1e-12)

    new_cell = np.array([
        [ax, bx, cx],
        [ay, by, cy],
        [az, bz, cz]
    ], dtype=float)

    return new_cell

def transform_coordinates(old_cell, new_cell, atoms):
    """
    古い直交座標 -> (old_cell)^-1 -> 分数座標 -> new_cell -> 新しい直交座標
    atoms: [(element, x, y, z), ...]
    戻り値: [(element, new_x, new_y, new_z), ...]
    """
    old_inv = np.linalg.inv(old_cell)
    new_atoms = []

    for elem, x, y, z in atoms:
        old_cart = np.array([x, y, z], dtype=float)
        frac = old_inv.dot(old_cart)         # 分数座標
        new_cart = new_cell.dot(frac)        # 新しい直交座標
        new_atoms.append((elem, new_cart[0], new_cart[1], new_cart[2]))

    return new_atoms

def write_xyz_with_lattice(filename, num_atoms, cell, atoms):
    """
    training.xyz を書き出す:
      1行目: num_atoms
      2行目: Lattice="..."
      以降: 原子情報
    """
    # cell[:,0], cell[:,1], cell[:,2] が a,b,c
    # それぞれ (x, y, z)
    ax, ay, az = cell[:, 0]
    bx, by, bz = cell[:, 1]
    cx, cy, cz = cell[:, 2]

    with open(filename, 'w') as f:
        # 1行目
        f.write(f"{num_atoms}\n")
        # 2行目: Lattice="ax ay az bx by bz cx cy cz"
        f.write('Lattice="')
        f.write(f"{ax:.4f} {ay:.4f} {az:.4f} ")
        f.write(f"{bx:.4f} {by:.4f} {bz:.4f} ")
        f.write(f"{cx:.4f} {cy:.4f} {cz:.4f}")
        f.write('"\n')

        # 原子座標
        for elem, x, y, z in atoms:
            f.write(f"{elem} {x:.4f} {y:.4f} {z:.4f}\n")

def main():
    # 1. training_base.xyz の読み込み
    base_file = "training_base.xyz"
    num_atoms, old_cell, atoms = read_xyz_with_lattice(base_file)

    # 2. 格子定数と角度の計算 (確認用に print する)
    A, B, C, alpha, beta, gamma = calc_lattice_parameters(old_cell)
    print("Old Lattice parameters:")
    print(f"  A, B, C = {A:.4f}, {B:.4f}, {C:.4f}")
    print(f"  alpha, beta, gamma = {alpha:.2f}, {beta:.2f}, {gamma:.2f} (deg)")

    # 例としては
    #  A,B,C = 8.3532, 8.3532, 4.1939
    #  alpha, beta, gamma = 90.0, 90.0, 159.4
    #
    # gamma>90 なので 180-gamma= 20.6 の扱いになる（下記 make_new_cell 内の簡易処理）

    # 3. 新しい格子行列 new_cell の生成
    new_cell = make_new_cell(A, B, C, alpha, beta, gamma)

    # 4. 原子座標の変換 (旧直交座標 -> 分数 -> 新直交座標)
    new_atoms = transform_coordinates(old_cell, new_cell, atoms)

    # 5. "training.xyz" に書き出し
    out_file = "training.xyz"
    write_xyz_with_lattice(out_file, num_atoms, new_cell, new_atoms)

    print(f"\n変換完了: '{out_file}' に書き込みました。")

if __name__ == "__main__":
    main()
