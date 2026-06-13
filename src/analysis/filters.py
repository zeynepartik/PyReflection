import pandas as pd
import numpy as np

def filter_dataset(df, 
                   elev_ranges=None, 
                   azim_ranges=None, 
                   wavelength_ranges=None,  # YENİ: Dalga boyu filtre aralığı
                   obs_types_include=None, 
                   obs_types_exclude=None, 
                   constellations=None, 
                   sat_ids=None, 
                   epoch_start=None, 
                   epoch_end=None):
    """
    Faz 5: Esnek, Sade ve Kullanıcı Dostu Filtreleme Katmanı.
    Herhangi bir parametre verilmezse o filtre otomatik olarak devredışı kalır.
    """
    # Orijinal veriyi korumak için kopyalıyoruz
    f_df = df.copy()
    if f_df.empty:
        return f_df

    # --- 1. ELEVATION FİLTRESİ (Birden çok aralık, OR mantığı) ---
    if elev_ranges is not None and len(elev_ranges) > 0:
        elev_mask = np.zeros(len(f_df), dtype=bool)
        for min_e, max_e in elev_ranges:
            elev_mask |= (f_df['elevation'] >= min_e) & (f_df['elevation'] <= max_e)
        f_df = f_df[elev_mask]

    # --- 2. AZIMUTH FİLTRESİ (Kuzey sarmalaması destekli, OR mantığı) ---
    if azim_ranges is not None and len(azim_ranges) > 0:
        azim_mask = np.zeros(len(f_df), dtype=bool)
        for min_a, max_a in azim_ranges:
            if min_a > max_a:  # Kuzey sarmalaması (Örn: 350'den 10'a)
                azim_mask |= (f_df['azimuth'] >= min_a) | (f_df['azimuth'] <= max_a)
            else:              # Normal aralık
                azim_mask |= (f_df['azimuth'] >= min_a) & (f_df['azimuth'] <= max_a)
        f_df = f_df[azim_mask]

    # --- 3. WAVELENGTH FİLTRESİ (YENİ: Min-Max Dalga Boyu Aralığı) ---
    # Örn: wavelength_ranges=[(0.19, 0.24)]
    if wavelength_ranges is not None and len(wavelength_ranges) > 0:
        wl_mask = np.zeros(len(f_df), dtype=bool)
        # Eğer ana tabloda wavelength sütunu yoksa, hata vermemesi için koruma ekliyoruz
        if 'wavelength' in f_df.columns:
            for min_wl, max_wl in wavelength_ranges:
                wl_mask |= (f_df['wavelength'] >= min_wl) & (f_df['wavelength'] <= max_wl)
            f_df = f_df[wl_mask]
        else:
            print("Uyarı: Veri tabanında 'wavelength' sütunu bulunamadığı için bu filtre atlandı.")

    # --- 4. OBS TYPE FİLTRESİ (Dahil Et / Hariç Tut) ---
    if obs_types_include is not None and len(obs_types_include) > 0:
        f_df = f_df[f_df['obsType'].isin(obs_types_include)]
        
    if obs_types_exclude is not None and len(obs_types_exclude) > 0:
        f_df = f_df[~f_df['obsType'].isin(obs_types_exclude)]

    # --- 5. CONSTELLATION FİLTRESİ (Örn: Yalnızca GPS 'G' ve Galileo 'E') ---
    if constellations is not None and len(constellations) > 0:
        f_df = f_df[f_df['constellation'].isin(constellations)]

    # --- 6. SATID FİLTRESİ (Belirli uydular listesi) ---
    if sat_ids is not None and len(sat_ids) > 0:
        f_df = f_df[f_df['satID'].isin(sat_ids)]

    # --- 7. EPOCH (ZAMAN) ARALIĞI FİLTRESİ ---
    if epoch_start is not None:
        f_df = f_df[f_df['epoch'] >= pd.to_datetime(epoch_start)]
    if epoch_end is not None:
        f_df = f_df[f_df['epoch'] <= pd.to_datetime(epoch_end)]

    return f_df.reset_index(drop=True)