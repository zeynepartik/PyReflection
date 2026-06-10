import os
import pandas as pd
from src.processing.rinex_sp3_merger import merge_rinex_sp3
from src.processing.arc_detector import detect_arcs_for_satellite

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

    # --- 2. ADIM: MERGE MOTORUNU ÇALIŞTIR  ---
    print(">>> Farklı zaman serilerindeki veriler enterpole ediliyor ve birleştiriliyor...")
    
     
    # Bu fonksiyon arka planda doğrudan pandas DataFrame döndürüyor.
    final_df = merge_rinex_sp3(sp3_file, rinex_file)
    
    if final_df is None:
        print("Hata: merge_rinex_sp3 fonksiyonundan veri alınamadı!")
        return

    # --- 3. ADIM: FAZ 4 - ARC TESPİTİ (MATRİS ALGORİTMASI) ---
    print("\n>>> Uyduların geçiş rotaları (Arc) matris yöntemiyle tespit ediliyor...")
    
    final_df['constellation'] = final_df['satID'].str[0]
    
    if 'SNR' in final_df.columns:
        final_df = final_df.rename(columns={'SNR': 'obsValue'})

    # Her uyduyu kendi içinde gruplayıp arc dedektörüne gönderiyoruz
    arc_processed_lists = []
    for sat_id, group in final_df.groupby('satID'):
        processed_group = detect_arcs_for_satellite(group)
        arc_processed_lists.append(processed_group)
        
    final_df = pd.concat(arc_processed_lists).reset_index(drop=True)
    
    # --- 4. ADIM: SÜTUNLARI HOCANIN ŞABLONUNA GÖRE SIRALA ---
    ordered_columns = [
        'epoch', 'constellation', 'satID', 'arcNo', 'arcType', 
        'obsType', 'obsValue', 'elevation', 'azimuth'
    ]
    final_df = final_df.reindex(columns=ordered_columns)

    # --- 5. ADIM: NİHAİ TABLOYU PARQUET OLARAK KAYDET ---
    
    output_filename = "final_rnx_sp3_merged.parquet"
    final_df.to_parquet(output_filename, index=False)
    
    print("\n--- FAZ 4 BAŞARIYLA TAMAMLANDI ---")
    print(f"Başarılı! {len(final_df)} adet saniyelik veri arcNo ve arcType ile etiketlenerek '{output_filename}' (Parquet) formatında kaydedildi.")
    print(final_df.head(10))

if __name__ == "__main__":
    main()