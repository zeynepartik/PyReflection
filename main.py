import os
import pandas as pd
import numpy as np
import datetime
from src.parsers.sp3_parser import parse_sp3
from src.parsers.rinex_parser import parse_rinex_obs
from src.geometry.coordinates import ecef_to_enu, calculate_elevation_azimuth, lagrange_interpolate

def main():
    # --- 1. ADIM: DOSYA YOLLARINI BELİRLE ---
    data_folder = "data"
    sp3_file = ""
    rinex_file = ""

    for f in os.listdir(data_folder):
        if f.lower().endswith(".sp3"): sp3_file = os.path.join(data_folder, f)
        if f.lower().endswith(".rnx"): rinex_file = os.path.join(data_folder, f)

    if not sp3_file or not rinex_file:
        print("Hata: Data klasöründe hem .sp3 hem .rnx dosyası olmalı!")
        return

    # --- 2. ADIM: HAM VERİLERİ OKU ---
    print(f">>> SP3 dosyası okunuyor...")
    sp3_df = parse_sp3(sp3_file)

    print(f">>> RINEX dosyası okunuyor...")
    rnx_df, rinex_header = parse_rinex_obs(rinex_file)
    
    # Anten sabit koordinatını alıyoruz
    ant_pos = rinex_header['approx_pos']
    
    # --- 3. ADIM: UNIQUE UYDULARI BULALIM ---
    
    unique_sp3_sat_list = sp3_df['satID'].unique()
    print(f"Tespit edilen benzersiz uydu sayısı: {len(unique_sp3_sat_list)}")

    # Enterpole edilmiş yeni verileri dolduracağımız boş bir liste
    interpolated_results = []

    # --- 4. ADIM: UYDU DÖNGÜSÜ  ---
    for TARGET_SAT in unique_sp3_sat_list:
        
        # O anki seçili uydu için SP3 verilerini filtrele ve zamana göre diz
        sat_sp3_df = sp3_df[sp3_df["satID"] == TARGET_SAT].sort_values("epoch").reset_index(drop=True)
        
        # O anki seçili uydu için RINEX verilerini filtrele ve zamana göre diz
        sat_rnx_df = rnx_df[rnx_df["satID"] == TARGET_SAT].sort_values("epoch").reset_index(drop=True)
        
        # Eğer bu uydu RINEX dosyasında hiç yoksa boşuna hesaplama, sonraki uyduya geç
        if sat_rnx_df.empty:
            continue

        # RINEX içindeki benzersiz epokları (zamanları) bulalım
        sat_rnx_unique = sat_rnx_df['epoch'].unique()
        n_rnx_epoch = len(sat_rnx_unique)

        # SP3 zamanlarını matematiksel hesap kolaylığı için 'saniye' tipine çeviriyoruz (x_nodes)
        # Günün ilk saniyesini 0 kabul ederek saniyeleri sayacağız
        sp3_base_time = sat_sp3_df['epoch'].min()
        sp3_seconds = [(t - sp3_base_time).total_seconds() for t in sat_sp3_df['epoch']]

        # --- 5. ADIM: RINEX EPOK DÖNGÜSÜ  ---
        for j in range(n_rnx_epoch):
            rnx_epoch = pd.Timestamp(sat_rnx_unique[j])
            
            # Hedef zamanımızın saniye karşılığı (x_target)
            target_seconds = (rnx_epoch - sp3_base_time).total_seconds()

            #  n_low ve n_high (komşu sayıları) hesabı:
            # Hedef zamandan küçük olan ve büyük olan SP3 zamanlarını sayıyoruz
            low_indices = [i for i, s in enumerate(sp3_seconds) if s <= target_seconds]
            high_indices = [i for i, s in enumerate(sp3_seconds) if s > target_seconds]

            n_low = len(low_indices)
            n_high = len(high_indices)

            # --- 6. ADIM: İHTİMAL FİLTRESİ ---
            chosen_indices = []
            
            # (i) Ortada bir yerdeysek: Önceki 5 ve sonraki 5 komşuyu al
            if n_low >= 5 and n_high >= 5:
                chosen_indices = low_indices[-5:] + high_indices[:5]
            
            # (ii) Günün çok başındaysak: İlk 10 epoğu al
            elif n_low < 5 and n_high >= 5:
                chosen_indices = list(range(0, 10))
            
            # (iii) Günün çok sonundaysak: Son 10 epoğu al
            elif n_low >= 5 and n_high < 5:
                total_sp3 = len(sp3_seconds)
                chosen_indices = list(range(total_sp3 - 10, total_sp3))
            
            # (iv) Yeterli veri yoksa pas geç 
            else:
                continue

            # Eğer bir şekilde 10 nokta toplayamadıysak listeyi güvenliğe al
            if len(chosen_indices) < 10:
                continue

            # Lagrange için 10 adet komşu X noktamız (Zamanlar)
            x_nodes = [sp3_seconds[idx] for idx in chosen_indices]

            # --- 7. ADIM: LAGRANGE ENTERPOLASYONU  ---
            # X, Y ve Z koordinatları için ayrı ayrı Lagrange çağırıyoruz
            y_nodes_x = [sat_sp3_df.iloc[idx]['X'] for idx in chosen_indices]
            y_nodes_y = [sat_sp3_df.iloc[idx]['Y'] for idx in chosen_indices]
            y_nodes_z = [sat_sp3_df.iloc[idx]['Z'] for idx in chosen_indices]

            interp_X = lagrange_interpolate(x_nodes, y_nodes_x, target_seconds)
            interp_Y = lagrange_interpolate(x_nodes, y_nodes_y, target_seconds)
            interp_Z = lagrange_interpolate(x_nodes, y_nodes_z, target_seconds)

            # --- 8. ADIM: ELDE EDİLEN X,Y,Z İLE AÇI HESAPLAMA ---
            # Yeni bulduğumuz hassas koordinatları bizim pusula sistemine (ENU) sokuyoruz
            e, n, u = ecef_to_enu(interp_X, interp_Y, interp_Z, ant_pos[0], ant_pos[1], ant_pos[2])
            elevation, azimuth = calculate_elevation_azimuth(e, n, u)

            # Aynı saniyede bu uydunun RINEX'teki SNR değerini de yanına ekleyelim 
            rnx_rows = sat_rnx_df[sat_rnx_df['epoch'] == rnx_epoch]
            
            # Bu uydu-zaman ikilisine ait tüm SNR (S1C, S2W vb.) gözlemlerini ekliyoruz
            for _, rnx_row in rnx_rows.iterrows():
                interpolated_results.append({
                    'epoch': rnx_epoch,
                    'satID': TARGET_SAT,
                    'X': interp_X,
                    'Y': interp_Y,
                    'Z': interp_Z,
                    'elevation': elevation,
                    'azimuth': azimuth,
                    'obsType': rnx_row['obsType'],
                    'SNR': rnx_row['obsValue']
                })

    # --- 9. ADIM: YENİ VERİ SETİNİ OLUŞTUR VE KAYDET ---
    final_df = pd.DataFrame(interpolated_results)
    
    print("\n--- ENTERPOLASYON VE BİRLEŞTİRME TAMAMLANDI ---")
    print(final_df.head())
    
    # Sonucu CSV dosyası olarak kaydedelim
    final_df.to_csv("final_rnx_sp3_merged.csv", index=False)
    print(f"\nBaşarılı! Toplam {len(final_df)} adet saniyelik veri enterpole edilerek SNR ile birleştirildi.")

if __name__ == "__main__":
    main()