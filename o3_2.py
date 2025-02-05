#!/usr/bin/env python3
import random, math
import numpy as np

# --- パラメータ ---
# Ti, Oの半径（見た目の球サイズ）
r_Ti = 1.47   # Å
r_O  = 0.74   # Å

# 配置すべき原子数（Ti:O = 1:2）
N_Ti = 10        # 必要に応じて変更してください
N_O  = 2 * N_Ti

# 距離条件
TiO_min = 1.8   # Ti-O距離下限 (Å)
TiO_max = 2.2   # Ti-O距離上限 (Å)
TiTi_min = 3.0  # 橋渡しするTi–Ti距離下限 (Å)
TiTi_max = 3.6  # 橋渡しするTi–Ti距離上限 (Å)
O_O_min  = 2.2  # あるOの最も近い別のOとの距離 (Å)

# Ti同士は球が重ならない：最低距離 = 2×r_Ti = 2.94Å
min_dist_TiTi = 2 * r_Ti

# --- 補助関数 ---
def random_vector_in_cube(L):
    """中心を原点とする一辺 L の立方体内から3次元ベクトルを返す"""
    return np.array([random.uniform(-L/2, L/2) for _ in range(3)])

def generate_Ti_positions(N, L):
    """
    L [Å] 辺の立方体内でTi原子（非重なり）をランダム配置する。
    非重なり条件：既存原子との距離が min_dist_TiTi 以上。
    """
    positions = []
    attempts = 0
    max_attempts = 100
    while len(positions) < N and attempts < max_attempts:
        pos = random_vector_in_cube(L)
        if all(np.linalg.norm(pos - p) >= min_dist_TiTi for p in positions):
            positions.append(pos)
        attempts += 1
    return positions if len(positions) == N else None

def bridging_O_position(posA, posB):
    """
    Ti原子 posA, posB のペアから，両Tiから距離 r = 2.0 (Å)となる
    橋渡し酸素の候補位置をランダムな方向（Ti–Ti軸に垂直な面上）で返す。
    なお，このTi–Ti距離が TiTi_min〜TiTi_max にあるか確認する。
    """
    d = np.linalg.norm(posB - posA)
    if d < TiTi_min or d > TiTi_max:
        return None
    # ここでは対称的に r = 2.0 とする（Ti-O距離 2.0は条件内）
    r = 2.0
    # 球の交わり円の半径
    try:
        R_circle = math.sqrt(r**2 - (d/2)**2)
    except ValueError:
        return None
    midpoint = (posA + posB) / 2.0
    # Ti–Ti軸に垂直な方向を求める
    direction = posB - posA
    if np.linalg.norm(direction) == 0:
        return None
    direction_unit = direction / np.linalg.norm(direction)
    arbitrary = np.array([1, 0, 0])
    if np.allclose(arbitrary, direction_unit) or np.allclose(arbitrary, -direction_unit):
        arbitrary = np.array([0, 1, 0])
    perp = np.cross(direction_unit, arbitrary)
    perp = perp / np.linalg.norm(perp)
    # perp と direction_unitと垂直なベクトル（perp2）を用いて平面内任意の方向を作る
    perp2 = np.cross(direction_unit, perp)
    angle = random.uniform(0, 2*math.pi)
    new_direction = perp * math.cos(angle) + perp2 * math.sin(angle)
    O_pos = midpoint + new_direction * R_circle
    return O_pos

def generate_bridging_O_positions(Ti_positions):
    """
    配置されたTi原子のすべてのペアについて，
    Ti–Ti距離が条件内ならば橋渡し酸素候補を生成する。
    結果は ((Ti_index1, Ti_index2), O_position) のタプルリスト。
    """
    bridging_candidates = []
    n = len(Ti_positions)
    for i in range(n):
        for j in range(i+1, n):
            posA = Ti_positions[i]
            posB = Ti_positions[j]
            d = np.linalg.norm(posB - posA)
            if d >= TiTi_min and d <= TiTi_max:
                O_pos = bridging_O_position(posA, posB)
                if O_pos is not None:
                    bridging_candidates.append(((i, j), O_pos))
    return bridging_candidates

def select_bridging_oxygens(candidates, target_count):
    """
    橋渡し候補リストからランダムに選び，
    追加するOどうしの距離が O_O_min 以上となるものを target_count 個選出する。
    選出できなければ None を返す。
    """
    random.shuffle(candidates)
    selected = []
    for cand in candidates:
        pos = cand[1]
        if all(np.linalg.norm(pos - s[1]) >= O_O_min for s in selected):
            selected.append(cand)
            if len(selected) == target_count:
                break
    return selected if len(selected) == target_count else None

def terminal_O_position(Ti_pos):
    """
    指定したTi原子の周りに，ランダムな方向へ距離2.0Åの端末型酸素を配置する。
    """
    r = 2.0
    theta = math.acos(random.uniform(-1, 1))
    phi = random.uniform(0, 2*math.pi)
    x = math.sin(theta) * math.cos(phi)
    y = math.sin(theta) * math.sin(phi)
    z = math.cos(theta)
    direction = np.array([x, y, z])
    return Ti_pos + r * direction

def fill_terminal_oxygens(Ti_positions, existing_O_positions, target_count):
    """
    橋渡し候補などで既に得られている酸素配置に加え，
    target_count 個になるように端末型酸素を追加する。
    追加する際は，既存のOとの間が O_O_min 以上であることをチェックする。
    """
    O_positions = list(existing_O_positions)  # 既に得られているOリスト（位置のみ）
    attempts = 0
    max_attempts = 1000
    while len(O_positions) < target_count and attempts < max_attempts:
        Ti_pos = random.choice(Ti_positions)
        new_O = terminal_O_position(Ti_pos)
        if all(np.linalg.norm(new_O - op) >= O_O_min for op in O_positions):
            O_positions.append(new_O)
        attempts += 1
    return O_positions if len(O_positions) == target_count else None

def generate_structure(N_Ti, L=8.0):
    """
    全体の構造（Ti, O座標）を生成する。
    1. 立方体内に Ti を非重なり条件で配置し，
    2. そのうち Ti–Ti 距離が TiTi_min〜TiTi_max となるペアから橋渡しO候補を作成，
       候補から target_count = N_O 個選出（O–O間の非重なりもチェック）。
    3. 候補が target_count 個未満の場合は端末型Oを補充。
    返り値は (Ti_positions, O_positions) 。
    """
    Ti_positions = generate_Ti_positions(N_Ti, L)
    if Ti_positions is None:
        return None, None
    bridging_candidates = generate_bridging_O_positions(Ti_positions)
    bridging_selected = select_bridging_oxygens(bridging_candidates, N_O)
    if bridging_selected is not None:
        O_positions = [cand[1] for cand in bridging_selected]
    else:
        # 候補が足りなかった場合，既存候補＋端末型Oで補充
        O_positions = [cand[1] for cand in bridging_candidates]
        O_positions = fill_terminal_oxygens(Ti_positions, O_positions, N_O)
        if O_positions is None:
            return Ti_positions, None
    return Ti_positions, O_positions

def check_structure(Ti_positions, O_positions):
    """
    以下の条件をチェックする：
      - 各酸素は少なくとも一つのTiと TiO_min〜TiO_max の距離にある。
      - 酸素同士の最短距離が O_O_min 以上。
      - もし酸素が2つ以上のTiと結合していれば，その Ti–O–Ti 角が 90〜155° 内である。
      - 同種原子（Ti同士）は球が重ならない（距離 >= 2*r_Ti）。
    条件をすべて満たせば True を返す。
    """
    # (1) 各酸素について Ti–O距離チェック
    for O in O_positions:
        distances = [np.linalg.norm(O - Ti) for Ti in Ti_positions]
        if not any(TiO_min <= d <= TiO_max for d in distances):
            return False
    # (2) O–O 非重なりチェック
    for i, O in enumerate(O_positions):
        for j, O2 in enumerate(O_positions):
            if i != j:
                if np.linalg.norm(O - O2) < O_O_min:
                    return False
    # (3) 橋渡しOの Ti–O–Ti 角チェック
    for O in O_positions:
        bonded_Ti = [Ti for Ti in Ti_positions if TiO_min <= np.linalg.norm(O - Ti) <= TiO_max]
        if len(bonded_Ti) >= 2:
            for i in range(len(bonded_Ti)):
                for j in range(i+1, len(bonded_Ti)):
                    vec1 = bonded_Ti[i] - O
                    vec2 = bonded_Ti[j] - O
                    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1)*np.linalg.norm(vec2))
                    angle = math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0)))
                    if angle < 90 or angle > 155:
                        return False
    # (4) Ti同士の重なり（非重なり条件）チェック
    for i, Ti in enumerate(Ti_positions):
        for j, Ti2 in enumerate(Ti_positions):
            if i < j:
                if np.linalg.norm(Ti - Ti2) < 2 * r_Ti:
                    return False
    return True

def write_xyz(frames, filename="training.xyz"):
    """
    frames は各試行結果のタプル (Ti_positions, O_positions, valid, L) のリスト。
    XYZ形式で全試行結果を書き出します。  
    有効な構造の場合はコメント行にLattice情報を出力し、
    無効な場合は "Invalid" と出力します。
    """
    with open(filename, "w") as f:
        for (Ti_positions, O_positions, valid, L) in frames:
            if Ti_positions is None or O_positions is None:
                continue
            natoms = len(Ti_positions) + len(O_positions)
            f.write(f"{natoms}\n")
            if valid:
                comment = f'Lattice="{L:.1f} 0.0 0.0 0.0 {L:.1f} 0.0 0.0 0.0 {L:.1f}"'
            else:
                comment = "Invalid"
            f.write(comment + "\n")
            for pos in Ti_positions:
                f.write(f"Ti {pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}\n")
            for pos in O_positions:
                f.write(f"O {pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}\n")

def write_xyz_valid(frames, filename="training_ok.xyz"):
    """
    frames の中から valid==True のものだけを抽出し、
    XYZ形式でファイルに書き出します。  
    コメント行は有効な構造の場合のLattice情報となります。
    """
    with open(filename, "w") as f:
        for (Ti_positions, O_positions, valid, L) in frames:
            if not valid:
                continue
            natoms = len(Ti_positions) + len(O_positions)
            f.write(f"{natoms}\n")
            comment = f'Lattice="{L:.1f} 0.0 0.0 0.0 {L:.1f} 0.0 0.0 0.0 {L:.1f}"'
            f.write(comment + "\n")
            for pos in Ti_positions:
                f.write(f"Ti {pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}\n")
            for pos in O_positions:
                f.write(f"O {pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}\n")

def main():
    frames = []
    # Lを8.0から10.0まで0.1刻みで試行
    L_values = np.arange(5.5, 5.8 + 0.1, 0.1)
    N_attempts_per_L = 1000  # 各Lでの試行回数
    total_trials = len(L_values) * N_attempts_per_L

    for L in L_values:
        print(f"L = {L:.1f} の試行中...")
        for _ in range(N_attempts_per_L):
            Ti_positions, O_positions = generate_structure(N_Ti, L)
            if Ti_positions is None or O_positions is None:
                valid = False
            else:
                valid = check_structure(Ti_positions, O_positions)
            if valid:
                print(f"Valid structure found at L = {L:.1f}")
            frames.append((Ti_positions, O_positions, valid, L))
    
    write_xyz(frames, "training.xyz")
    write_xyz_valid(frames, "training_ok.xyz")
    print(f"計 {total_trials} 回の試行結果を training.xyz に、")
    print("有効な構造のみを training_ok.xyz に書き込みました。")

if __name__ == '__main__':
    main()
