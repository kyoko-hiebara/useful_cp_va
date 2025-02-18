import numpy as np

def read_xyz(filename):
    """
    XYZファイルを読み込み、原子記号と座標（numpy array）、コメント行を返す関数
    """
    with open(filename, "r") as f:
        lines = f.readlines()
    natoms = int(lines[0].strip())
    comment = lines[1].strip()
    symbols = []
    coords = []
    for line in lines[2:2+natoms]:
        parts = line.split()
        symbols.append(parts[0])
        coords.append([float(x) for x in parts[1:4]])
    return symbols, np.array(coords), comment

def write_xyz(filename, structures):
    """
    複数の構造をXYZ形式で書き込む関数
    structuresは (symbols, coords, comment) のタプルのリスト
    """
    with open(filename, "w") as f:
        for sym, coords, comment in structures:
            f.write(f"{len(sym)}\n")
            f.write(comment + "\n")
            for s, (x, y, z) in zip(sym, coords):
                f.write(f"{s} {x: .8f} {y: .8f} {z: .8f}\n")

def modify_angle(structure, idxA, idxB, idxC, angle_list_deg):
    """
    構造(structure)内の3原子 A, B, C（Bが頂点）について、
    B–Cの方向を変えることで A–B–C の角度を指定した値（deg）に変更する。
    残りの原子は固定したまま、原子Cのみ座標変更します。
    
    structure: (symbols, coords, comment) のタプル
    idxA, idxB, idxC: 対象原子のインデックス（0-indexed）
    angle_list_deg: 角度のリスト（度単位）　例: [0, 10, 20, ..., 180]
    
    戻り値: 各角度設定ごとの構造リスト（XYZ書式出力用）
    """
    symbols, coords, comment = structure
    A = coords[idxA]
    B = coords[idxB]
    C = coords[idxC]
    
    # B–Cの長さ（固定とする）
    r = np.linalg.norm(C - B)
    
    # A–B方向を基準にする（Bが頂点なので、BA = A - B）
    u = A - B
    norm_u = np.linalg.norm(u)
    if norm_u == 0:
        raise ValueError("原子AとBが重なっています。")
    u = u / norm_u

    # 現在のB–Cベクトル
    v = C - B
    # u方向成分を引いて、uに垂直な成分を抽出
    v_parallel = np.dot(v, u) * u
    v_perp = v - v_parallel
    norm_v_perp = np.linalg.norm(v_perp)
    if norm_v_perp < 1e-8:
        # もし v が u と平行ならば（角度が0°または180°）任意の垂直方向をとる
        # ここでは、uと直交する単位ベクトルとして [0,0,1]（ただし、uが[0,0,±1]の場合は別のもの）
        if np.allclose(u, [0,0,1]) or np.allclose(u, [0,0,-1]):
            v_perp = np.array([0,1,0])
        else:
            v_perp = np.array([0,0,1])
        norm_v_perp = 1.0
    else:
        v_perp = v_perp / norm_v_perp

    new_structures = []
    for deg in angle_list_deg:
        theta = np.deg2rad(deg)
        # 新たなB–Cベクトル：u方向成分を r*cos(theta)、垂直成分を r*sin(theta) に設定
        new_v = r * (np.cos(theta) * u + np.sin(theta) * v_perp)
        new_coords = coords.copy()
        new_coords[idxC] = B + new_v
        new_comment = f"Angle A-B-C set to {deg} degrees"
        new_structures.append((symbols, new_coords, new_comment))
    return new_structures

if __name__ == "__main__":
    # 入力・出力ファイル名（必要に応じて変更してください）
    input_file = "input.xyz"
    output_file = "output.xyz"
    
    # 入力XYZファイルを読み込み
    symbols, coords, comment = read_xyz(input_file)
    structure = (symbols, coords, comment)
    
    # 対象とする3原子のインデックス（0-indexed: 例として A=0, B=1, C=2）
    idxA = 0
    idxB = 1
    idxC = 2
    
    # 変更後の角度リスト（例：0°から180°まで10°刻み）
    angle_list_deg = list(range(0, 181, 10))
    
    # 各角度に対して原子Cの位置を再計算した構造リストを作成
    new_structures = modify_angle(structure, idxA, idxB, idxC, angle_list_deg)
    
    # すべての構造を1つのXYZファイルに書き出す
    write_xyz(output_file, new_structures)
