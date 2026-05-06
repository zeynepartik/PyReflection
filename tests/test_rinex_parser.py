"""
RINEX parser test scripti.
- Header'dan approx_pos bilgisini yazdırır
- Toplam unique epok sayısını bulur
- İlk ve son epok zamanlarını yazdırır
- Seçilen uyduya ait ilk ve son epoktaki gözlemleri yazdırır
"""

import os
import sys

# src klasörünü import yoluna ekle
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.parsers.rinex_parser import parse_rinex_obs


# --- AYARLAR ---
RINEX_FILE = os.path.join(PROJECT_ROOT, "data", "PTLD00AUS_R_20220010000_01D_30S_MO.rnx")
TARGET_SAT = "J05"  # İncelenecek uydu (örn. G05, R10, E25, C03)


def main():
    print(f">>> Dosya okunuyor: {os.path.basename(RINEX_FILE)}")
    df, header = parse_rinex_obs(RINEX_FILE)

    # --- HEADER BİLGİSİ ---
    approx_pos = header.get("approx_pos")
    print("\n--- HEADER BİLGİSİ ---")
    print(f"Approx Position (XYZ): {approx_pos}")

    if df.empty:
        print("Uyarı: Hiç gözlem okunamadı.")
        return

    # --- EPOK İSTATİSTİKLERİ ---
    unique_epochs = df["epoch"].drop_duplicates().sort_values()
    first_epoch = unique_epochs.iloc[0]
    last_epoch = unique_epochs.iloc[-1]

    print("\n--- EPOK BİLGİSİ ---")
    print(f"Toplam unique epok sayisi : {len(unique_epochs)}")
    print(f"İlk epok                  : {first_epoch}")
    print(f"Son epok                  : {last_epoch}")

    # --- SEÇİLİ UYDU İÇİN GÖZLEMLER ---
    sat_df = df[df["satID"] == TARGET_SAT]

    print(f"\n--- {TARGET_SAT} UYDUSU GÖZLEMLERİ ---")
    if sat_df.empty:
        print(f"Uyari: {TARGET_SAT} uydusuna ait gözlem bulunamadı.")
        return

    first_obs = sat_df[sat_df["epoch"] == first_epoch]
    last_obs = sat_df[sat_df["epoch"] == last_epoch]

    print(f"\n[İlk epok @ {first_epoch}]")
    if first_obs.empty:
        print(f"  {TARGET_SAT} bu epokta gözlenmemiş.")
    else:
        print(first_obs.to_string(index=False))

    print(f"\n[Son epok @ {last_epoch}]")
    if last_obs.empty:
        print(f"  {TARGET_SAT} bu epokta gözlenmemiş.")
    else:
        print(last_obs.to_string(index=False))


if __name__ == "__main__":
    main()
