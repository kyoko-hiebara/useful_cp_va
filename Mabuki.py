#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
このスクリプトは、extxyz形式の軌道データから、任意の間隔でフレームを抽出（間引き）し、
新たなextxyzファイルとして保存します。

【使い方】
    python thin_trajectory.py input.extxyz output.extxyz 10
上記の例では、input.extxyzから10フレームごとに1フレームを抽出してoutput.extxyzに保存します。
"""

import argparse

def thin_trajectory(input_file, output_file, interval):
    """
    extxyzファイルを読み込み、指定された間隔でフレームを間引きして出力ファイルに保存する関数
    
    Parameters:
        input_file (str): 入力extxyzファイルのパス
        output_file (str): 出力extxyzファイルのパス
        interval (int): 何フレームごとに1フレームを抽出するか（例: 10なら10フレームごとに抽出）
    """
    # 入力ファイルを全行読み込み
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    frames = []
    i = 0
    # extxyzファイルは各フレームの最初に原子数（整数）が記載され、
    # 次の行にコメント、その後に原子ごとのデータが記述されていると仮定する。
    while i < len(lines):
        # 空行はスキップ
        if not lines[i].strip():
            i += 1
            continue
        
        try:
            # 現在の行に原子数が記述されている
            natoms = int(lines[i].strip())
        except ValueError:
            print(f"エラー: 行 {i+1} に整数が見つかりませんでした。内容: {lines[i].strip()}")
            break
        
        # フレーム全体は、原子数の行、コメント行、および各原子のデータ行（natoms行）で構成される
        frame_lines = lines[i : i + 2 + natoms]
        if len(frame_lines) < 2 + natoms:
            print("不完全なフレームが見つかりました。処理を終了します。")
            break
        frames.append(frame_lines)
        i += 2 + natoms
    
    # 指定された間隔ごとにフレームを抽出（例: interval=10なら0,10,20,...番目のフレームを抽出）
    thinned_frames = frames[::interval]
    
    # 抽出したフレームを出力ファイルに書き込む
    with open(output_file, "w", encoding="utf-8") as f:
        for frame in thinned_frames:
            f.write("".join(frame))
    
    print(f"間引かれた軌道が {output_file} に保存されました。抽出されたフレーム数: {len(thinned_frames)}")

def main():
    parser = argparse.ArgumentParser(
        description="extxyz形式の軌道データを指定した間隔で間引きするスクリプト"
    )
    parser.add_argument("input", help="入力extxyzファイルのパス")
    parser.add_argument("output", help="出力extxyzファイルのパス")
    parser.add_argument("interval", type=int, help="間引くフレームの間隔（例: 10なら10フレームごとに抽出）")
    args = parser.parse_args()
    
    thin_trajectory(args.input, args.output, args.interval)

if __name__ == "__main__":
    main()
