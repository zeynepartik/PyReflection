import os
import pandas as pd
from src.parsers.sp3_parser import parse_sp3
from src.parsers.rinex_parser import parse_rinex_obs
from src.geometry.coordinates import ecef_to_enu, calculate_elevation_azimuth

def main():
    # 1. DOSYA YOLLARINI BELİRLE
    data_folder = "data"
    sp3_file = ""
    rinex_file = ""

    # Klasördeki dosyaları bulalım
    for f in os.listdir(data_folder):
        if f.lower().endswith(".sp3"): sp3_file = os.path.join(data_folder, f)
        if f.lower().endswith(".rnx"): rinex_file = os.path.join(data_folder, f)

    if not sp3_file or not rinex_file:
        print("Hata: Data klasöründe hem .sp3 hem .rnx dosyası olmalı!")
        return

    # 2. VERİLERİ OKU
    print(f">>> SP3 dosyası okunuyor: {os.path.basename(sp3_file)}")
    df_sp3 = parse_sp3(sp3_file)

    print(f">>> RINEX dosyası okunuyor: {os.path.basename(rinex_file)}")
    _, rinex_header = parse_rinex_obs(rinex_file)
    
    # Anten koordinatlarını header'dan alıyoruz (X, Y, Z)
    ant_pos = rinex_header['approx_pos']
    print(f"Anten Konumu Tespit Edildi: {ant_pos}")

    # 3. GEOMETRİK HESAPLAMALARI YAP
    print(">>> Açı hesaplamaları başlatılıyor (Bu işlem biraz sürebilir)...")
    
    elevations = []
    azimuths = []

    # Her bir uydu satırı için hesaplama yapalım
    for index, row in df_sp3.iterrows():
        # Uydunun o anki koordinatları
        sat_xyz = (row['X'], row['Y'], row['Z'])
        
        # 1. Adım: ECEF -> ENU dönüşümü
        e, n, u = ecef_to_enu(sat_xyz[0], sat_xyz[1], sat_xyz[2], 
                              ant_pos[0], ant_pos[1], ant_pos[2])
        
        # 2. Adım: ENU -> Elevation, Azimuth
        el, az = calculate_elevation_azimuth(e, n, u)
        
        elevations.append(el)
        azimuths.append(az)

    # 4. SONUÇLARI TABLOYA EKLE
    df_sp3['elevation'] = elevations
    df_sp3['azimuth'] = azimuths

    # 5. EKRANA BAS VE KONTROL ET
    print("\n--- HESAPLAMA TAMAMLANDI ---")
    # Sadece anlamlı yükseklikteki (ufuk üstü) uyduları görelim
    print(df_sp3[df_sp3['elevation'] > 0].head())
    
    # Sonucu bir CSV olarak kaydetmek istersen:
    # df_sp3.to_csv("uydu_acilari.csv", index=False)
    print(f"\nToplam {len(df_sp3)} satır işlendi.")

if __name__ == "__main__":
    main()