import pandas as pd
import numpy as np

def detect_arcs_for_satellite(sat_df):
    """
    Tek bir uyduya ait kronolojik veriyi alır.
    Hocanın istediği matris farkı (Vektörizasyon) ve 2 saatlik kesinti kuralıyla
    arcNo ve arcType sütunlarını hesaplar.
    """
    # Verinin zamana göre sıralı olduğundan emin olalım
    sat_df = sat_df.sort_values("epoch").reset_index(drop=True)
    if sat_df.empty:
        return sat_df

    n = len(sat_df)
    
    # Varsayılan değerleri atayalım
    arc_nos = np.ones(n, dtype=int)
    arc_types = np.ones(n, dtype=int) # 1: Yükselen, -1: Alçalan

    if n < 2:
        sat_df['arcNo'] = arc_nos
        sat_df['arcType'] = arc_types
        return sat_df

    # --- MATRİS YÖNTEMİYLE TÜREV (FARK) HESABI ---
    # E1'den En'e giden matris:  sat_df['elevation'].values[1:]
    # E0'dan En-1'e giden matris: sat_df['elevation'].values[:-1]
    elevations = sat_df['elevation'].values
    d_elev = elevations[1:] - elevations[:-1]

    # İşaret tespiti: Pozitifse (yükseliyor) 1, Negatifse (alçalıyor) -1
    # Sıfıra eşitse bir önceki işareti korumak için nan_to_num veya sign kullanıyoruz
    signs = np.sign(d_elev)
    # İlk noktanın işareti olmadığı için başına kopyalıyoruz ki matris boyutu n olsun
    full_signs = np.concatenate(([signs[0]], signs))
    # 0 olan yerleri (düzlükleri) bir önceki işaretle doldurmak için düzeltebiliriz
    full_signs[full_signs == 0] = 1 

    # ---  ZAMAN KESİNTİSİ KONTROLÜ (2 SAAT KURALI) ---
    epochs = sat_df['epoch'].values
    time_diffs = (epochs[1:] - epochs[:-1]) / np.timedelta64(1, 'h') # Saat cinsinden fark
    time_gaps = time_diffs >= 2.0
    full_time_gaps = np.concatenate(([False], time_gaps))

    # ---  ARC DEĞİŞİM NOKTALARININ (İŞARET DEĞİŞİMİ) TESPİTİ ---
    # İşaret bir önceki satıra göre değiştiyse (full_signs[i] != full_signs[i-1]) yeni bir Arc başlar
    sign_changes = full_signs[1:] != full_signs[:-1]
    full_sign_changes = np.concatenate(([False], sign_changes))

    # Yeni bir arc tetikleyen durumlar: YA işaret değişti YA DA 2 saatlik boşluk oldu!
    arc_triggers = full_sign_changes | full_time_gaps

    # ---  NUMARALANDIRMA VE TİP ATAMASI ---
    # Tetiklenen yerlerde arc numarasını 1 artırıyoruz (Kumulatif toplam)
    current_arc = 1
    for i in range(n):
        if arc_triggers[i]:
            current_arc += 1
        arc_nos[i] = current_arc
        # O andaki türev işareti uydunun tipini söyler (Artıyor -> 1, Azalıyor -> -1)
        arc_types[i] = 1 if full_signs[i] >= 0 else -1

    sat_df['arcNo'] = arc_nos
    sat_df['arcType'] = arc_types
    
    return sat_df