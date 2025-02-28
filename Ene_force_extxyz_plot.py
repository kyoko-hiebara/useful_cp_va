import matplotlib.pyplot as plt
import re

# ファイル名を指定（適宜変更してください）
filename = "sample.extxyz"

energy_list = []
total_force_list = []

with open(filename, "r") as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    # 空行はスキップ
    if lines[i].strip() == "":
        i += 1
        continue

    # 1行目：原子数（例: "32"）
    try:
        num_atoms = int(lines[i].strip())
    except ValueError:
        raise ValueError(f"{i+1}行目が原子数として認識できません: {lines[i].strip()}")
    
    # 2行目：メタデータ行。例："energy = -123.456 その他の情報..."
    metadata = lines[i+1].strip()
    energy = None
    # まずトークンごとに調べ、"energy"が含まれるものから抽出
    for token in metadata.split():
        if "energy" in token:
            # tokenが "energy=-123.456" のような形式の場合
            if "=" in token:
                parts = token.split("=")
                if len(parts) == 2 and parts[1] != "":
                    try:
                        energy = float(parts[1])
                    except ValueError:
                        pass
            # もし energy と "=" が別々の場合、後続トークンに数値があるかチェック
            if energy is None:
                idx = metadata.split().index(token)
                # 次のトークンが "=" であれば、その次が数値になる場合も
                tokens = metadata.split()
                if idx+1 < len(tokens) and tokens[idx+1] == "=" and idx+2 < len(tokens):
                    try:
                        energy = float(tokens[idx+2])
                    except ValueError:
                        pass
            if energy is not None:
                break
    # 正規表現でも抽出を試みる
    if energy is None:
        m = re.search(r"energy\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", metadata)
        if m:
            energy = float(m.group(1))
        else:
            raise ValueError(f"エネルギー情報が見つかりませんでした: {metadata}")
    energy_list.append(energy)
    
    # 3行目以降：原子ごとのデータ（原子数分の行があると仮定）
    total_force = 0.0
    for j in range(i+2, i+2+num_atoms):
        atom_line = lines[j].strip()
        if atom_line == "":
            continue
        cols = atom_line.split()
        # 力の成分が4列目～6列目（1-indexed）として、
        # Pythonの0-indexedでは cols[3], cols[4], cols[5] と仮定します。
        try:
            fx = float(cols[3])
            fy = float(cols[4])
            fz = float(cols[5])
        except IndexError:
            raise IndexError(f"原子データ行の列数が不足しています: {atom_line}")
        total_force += (fx + fy + fz)
    total_force_list.append(total_force)
    
    # 次のブロックへ移動（1行目＋メタデータ＋原子数分の行）
    i += 2 + num_atoms

# プロット作成
plt.figure(figsize=(12, 5))

# エネルギープロット
plt.subplot(1, 2, 1)
plt.plot(energy_list, marker='o')
plt.xlabel("ステップ")
plt.ylabel("エネルギー")
plt.title("ステップごとのエネルギー")

# 力の合計値プロット
plt.subplot(1, 2, 2)
plt.plot(total_force_list, marker='o', color='red')
plt.xlabel("ステップ")
plt.ylabel("力の合計")
plt.title("ステップごとの力の合計")

plt.tight_layout()
plt.show()
