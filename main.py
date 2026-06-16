import os
import pandas as pd
import numpy as np
from src.processing.rinex_sp3_merger import merge_rinex_sp3
from src.analysis.filters import filter_dataset
from src.analysis.lomb_scargle import detrend_snr, calculate_lombscargle

def main():
    # --- 1. ADIM: DOSYA YOLLARINI BELİRLE ---
    data_folder = "data"
    sp3_file = ""
    rinex_file = ""

    for f in os.listdir(data_folder):
        if f.lower().endswith(".sp3"):
            sp3_file = os.path.join(data_folder, f)
        if f.lower().endswith(".rnx"):
            rinex_file = os.path.join(data_folder, f)

    if not sp3_file or not rinex_file:
        print("Hata: Data klasöründe hem .sp3 hem .rnx dosyası olmalı!")
        return

    # --- 2. ADIM: MERGE VE ARC MOTORUNU ÇALIŞTIR ---
    merge_rinex_sp3(sp3_file, rinex_file)
    
    # --- 3. ADIM: OLUŞAN PARQUET DOSYASINI HAFIZAYA OKU ---
    generated_file = os.path.join("data", "processed", "PTLD00AUS_2022001_30S.parquet")
    
    if not os.path.exists(generated_file):
        print(f"Hata: Üretilen veri dosyası bulunamadı: {generated_file}")
        return
        
    print(f"\n>>> Üretilen {generated_file} dosyası analiz ve filtreleme için yükleniyor...")
    analysis_df = pd.read_parquet(generated_file)

    # --- 4. ADIM: FAZ 5 — FİLTRELEME KATMANI ÇALIŞTIRMA ---
    print("\n>>> Faz 5 Filtreleme Katmanı devreye giriyor...")

    filtered_df = filter_dataset(
        analysis_df,                       
        elev_ranges=[(5, 25)],             
        wavelength_ranges=[(0.19, 0.24)], 
        obs_types_include=['S1C'],         
        obs_types_exclude=['S2W']          
    )

    print(f"\n>>> Filtreleme bitti. Kalan satır sayısı: {len(filtered_df)}")
    
    # Filtrelenmiş veriyi kaydet
    output_filename = "final_analysis_subset.parquet"
    filtered_df.to_parquet(output_filename, index=False)
    print(f">>> Filtrelenmiş analiz alt kümesi '{output_filename}' olarak kaydedildi.\n")

    # --- 5. ADIM: FAZ 6 — LOMB-SCARGLE TESTİ ---
    print("\n>>> Faz 6 Sinyal Analizis devreye giriyor...")
    
    
    if not filtered_df.empty:
        sample_sat = filtered_df[filtered_df['satID'] == filtered_df['satID'].iloc[0]]
        
        elevations = sample_sat['elevation'].values
        snr_values = sample_sat['obsValue'].values
        wavelength_val = sample_sat['wavelength'].iloc[0] 
        
        print(f"Test Edilen Uydu: {sample_sat['satID'].iloc[0]}, Veri Noktası Sayısı: {len(sample_sat)}")
        print(f"Sinyal Dalga Boyu (Lambda): {wavelength_val:.4f} metre")

        # 1. Trendden Arındırma
        d_snr = detrend_snr(elevations, snr_values)
        print(">>> 2. Dereceden Polinom ile SNR trendden arındırıldı.")

        # 2. Lomb-Scargle Hesaplama
        h_grid, lsp_power = calculate_lombscargle(
            elevation=elevations, 
            detrended_snr=d_snr, 
            wavelength=wavelength_val, 
            h_min=0.0, 
            h_max=20.0, 
            precision=0.01
        )
        
        print(">>> Astropy Lomb-Scargle periyodogramı matris yöntemiyle hesaplandı.")
        
        dominant_h = h_grid[np.argmax(lsp_power)]
        peak_p = np.max(lsp_power)
        print(f"\n[ANALİZ SONUCU]: Tespit Edilen Baskın Reflektör Yüksekliği: {dominant_h:.2f} metre! (Güç: {peak_p:.2f})")
    else:
        print("Uyarı: Filtreleme sonrası veri seti boş kaldığı için Faz 6 testi atlandı!")

    print("\n---  BAŞARIYLA TAMAMLANDI ---")

if __name__ == "__main__":
    main()