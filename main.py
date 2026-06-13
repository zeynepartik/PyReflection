import os
import pandas as pd
from src.processing.rinex_sp3_merger import merge_rinex_sp3
from src.analysis.filters import filter_dataset

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
    # Bu fonksiyon dosyayı otomatik olarak üretip diskteki klasöre yazıyor
    merge_rinex_sp3(sp3_file, rinex_file)
    
    # --- 3. ADIM: OLUŞAN PARQUET DOSYASINI HAFIZAYA OKU ---
    #  kodun çıktısını doğrudan adresten okuyarak hafızaya alıyoruz
    generated_file = os.path.join("data", "processed", "PTLD00AUS_2022001_30S.parquet")
    
    if not os.path.exists(generated_file):
        print(f"Hata: Üretilen veri dosyası bulunamadı: {generated_file}")
        return
        
    print(f"\n>>> Üretilen {generated_file} dosyası analiz ve filtreleme için yükleniyor...")
    analysis_df = pd.read_parquet(generated_file)

    # --- 4. ADIM: FAZ 5 — FİLTRELEME KATMANI ÇALIŞTIRMA ---
    print("\n>>> Faz 5 Filtreleme Katmanı devreye giriyor...")

    # Sadece senin istediğin filtreleri yazıyorsun, gerisi otomatik hallediliyor!
    filtered_df = filter_dataset(
        analysis_df,                       
        elev_ranges=[(5, 25)],             
        wavelength_ranges=[(0.19, 0.24)], 
        obs_types_include=['S1C'],         
        obs_types_exclude=['S2W']           
    )

    print(f"\n>>> Filtreleme bitti. Kalan satır sayısı: {len(filtered_df)}")
    
    # --- 5. ADIM: FİLTRELENMİŞ TEMİZ VERİYİ KAYDET ---
    output_filename = "final_analysis_subset.parquet"
    filtered_df.to_parquet(output_filename, index=False)
    print(f">>> Filtrelenmiş analiz alt kümesi '{output_filename}' olarak kaydedildi.\n")
    print(filtered_df.head(10))

if __name__ == "__main__":
    main()