#!/usr/bin/env python3
import argparse

def calculate_surface_charge(N_ads_slab, N_ads_ads, N_iso_slab, N_iso_ads, area_angstrom2):
    """
    Bader解析結果から表面の帯電状態を解析します。
    
    Parameters:
      N_ads_slab  : 吸着系におけるスラブの電子数
      N_ads_ads   : 吸着系における吸着分子の電子数
      N_iso_slab  : 孤立系におけるスラブの電子数
      N_iso_ads   : 孤立系における吸着分子の電子数
      area_angstrom2 : 表面積 [Å²]
      
    Returns:
      結果を辞書型で返します（また、画面に各値を表示します）
    """
    # 定数：電気素量 (C)
    e0 = 1.602176634e-19

    # 各系での電子数の差（スラブ側・吸着分子側）
    delta_N_slab = N_ads_slab - N_iso_slab
    delta_N_ads  = N_ads_ads  - N_iso_ads

    # 全体の電子数は保存されるはずなので、理想的には delta_N_slab + delta_N_ads ≈ 0
    total_delta = delta_N_slab + delta_N_ads
    if abs(total_delta) > 1e-6:
        print(f"注意: 全体の電子数保存則からずれています（Δ = {total_delta:.6e}）")

    # 電子は負の電荷を持つので、スラブが電子を獲得すれば (ΔN_slab > 0) 負に帯電します
    Q_slab = -e0 * delta_N_slab  # [C]

    # 面積の単位変換：1 Å² = 1e-20 m²
    area_m2  = area_angstrom2 * 1e-20
    # 1 m² = 1e4 cm²
    area_cm2 = area_m2 * 1e4

    # 表面電荷密度 (C/m², C/cm²)
    density_C_per_m2  = Q_slab / area_m2
    density_C_per_cm2 = Q_slab / area_cm2

    # 単位変換：1 C = 1e6 μC
    density_uC_per_m2  = density_C_per_m2 * 1e6
    density_uC_per_cm2 = density_C_per_cm2 * 1e6

    # 1 m²あたりの電気素量（絶対値）
    elementary_charges_per_m2 = abs(density_C_per_m2) / e0

    # 結果の出力
    print("=== 表面帯電解析結果 ===")
    print(f"ΔN (スラブ): {delta_N_slab:.6f} 電子")
    print(f"ΔN (吸着分子): {delta_N_ads:.6f} 電子")
    print(f"スラブのネット電荷: {Q_slab:.6e} C")
    print("")
    print(f"入力面積: {area_angstrom2:.6f} Å² = {area_m2:.6e} m²")
    print("")
    print("表面電荷密度:")
    print(f"  {density_uC_per_cm2:.6e} μC/cm²")
    print(f"  {density_uC_per_m2:.6e} μC/m²")
    print("")
    print(f"1 m²あたりの電気素量: {elementary_charges_per_m2:.6e} e")

    # 結果を辞書で返す
    return {
        'delta_N_slab': delta_N_slab,
        'delta_N_ads': delta_N_ads,
        'Q_slab': Q_slab,
        'density_uC_per_cm2': density_uC_per_cm2,
        'density_uC_per_m2': density_uC_per_m2,
        'elementary_charges_per_m2': elementary_charges_per_m2,
        'area_m2': area_m2
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bader解析結果から表面帯電状態を解析します。"
    )
    parser.add_argument("--N_ads_slab", type=float, required=True,
                        help="吸着系におけるスラブの電子数")
    parser.add_argument("--N_ads_ads", type=float, required=True,
                        help="吸着系における吸着分子の電子数")
    parser.add_argument("--N_iso_slab", type=float, required=True,
                        help="孤立系におけるスラブの電子数")
    parser.add_argument("--N_iso_ads", type=float, required=True,
                        help="孤立系における吸着分子の電子数")
    parser.add_argument("--area", type=float, required=True,
                        help="表面積 [Å²]")
    
    args = parser.parse_args()
    
    calculate_surface_charge(
        N_ads_slab=args.N_ads_slab,
        N_ads_ads=args.N_ads_ads,
        N_iso_slab=args.N_iso_slab,
        N_iso_ads=args.N_iso_ads,
        area_angstrom2=args.area
    )
