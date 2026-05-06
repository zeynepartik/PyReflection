import pandas as pd
import os
import datetime

def parse_rinex_obs(file_path):
    """
    RINEX 3.x Gözlem dosyasını okur. 
    Epok satırlarındaki uydu sayısını esnek bir şekilde tespit eder.
    """
    header_info = {'obs_types': {}}
    observations = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    line_idx = 0
    # --- BÖLÜM 1: HEADER (BAŞLIK) ANALİZİ ---
    while line_idx < len(lines):
        line = lines[line_idx]
        label = line[60:].strip()
        
        if "APPROX POSITION XYZ" in label:
            header_info['approx_pos'] = [float(x) for x in line[:60].split()]
        elif "ANTENNA: DELTA H/E/N" in label:
            header_info['ant_delta'] = [float(x) for x in line[:60].split()]
        elif "SYS / # / OBS TYPES" in label:
            if line[0].strip(): 
                current_sys = line[0]
                obs_list = line[6:60].split()
                header_info['obs_types'][current_sys] = obs_list
            else:
                header_info['obs_types'][current_sys].extend(line[6:60].split())
        elif "INTERVAL" in label:
            header_info['interval'] = float(line[:10].strip())
        elif "TIME OF FIRST OBS" in label:
            p = line[:60].split()
            header_info['first_obs'] = datetime.datetime(
                int(p[0]), int(p[1]), int(p[2]), int(p[3]), int(p[4]), int(float(p[5]))
            )
        elif "END OF HEADER" in label:
            line_idx += 1
            break
        line_idx += 1

    # --- BÖLÜM 2: VERİ BLOKLARI OKUMA ---
    while line_idx < len(lines):
        line = lines[line_idx]
        
        if line.startswith('>'):
            # --- YENİ HATA DÜZELTME MANTIĞI ---
            # Satırı parçalara ayırıyoruz (split)
            parts = line[1:].split()
            
            # Zaman bilgisini al (ilk 6 parça)
            current_epoch = datetime.datetime(
                int(parts[0]), int(parts[1]), int(parts[2]),
                int(parts[3]), int(parts[4]), int(float(parts[5]))
            )
            
            # RINEX 3 standardına göre uydu sayısı genellikle parçalanmış satırın
            # sondan bir önceki (saat kayması varsa) veya sonuncu elemanıdır.
            # En garanti yol: Sondan ikinci veya üçüncü tam sayıya bakmaktır.
            # Senin dosyan için parts[7] veya parts[8] olabilir.
            # Deneyelim: Genellikle saat kayması olan dosyalarda '55' değeri 
            # listenin sondan bir önceki elemanıdır.
            try:
                # Önce standart konumu (8. indeks) dene
                num_sats = int(parts[7])
            except (ValueError, IndexError):
                # Eğer o olmazsa (kayma varsa) 7. indeksi dene
                num_sats = int(parts[6])

            # Her bir uyduyu sırayla oku
            for _ in range(num_sats):
                line_idx += 1
                if line_idx >= len(lines): break
                sat_line = lines[line_idx]
                sat_id = sat_line[:3].strip()
                sys = sat_id[0]
                
                if sys in header_info['obs_types']:
                    labels = header_info['obs_types'][sys]
                    for i, obs_label in enumerate(labels):
                        start = 3 + (i * 16)
                        end = start + 14
                        val_str = sat_line[start:end].strip()
                        
                        if val_str and obs_label.startswith('S'):
                            observations.append({
                                'epoch': current_epoch,
                                'constellation': sys,
                                'satID': sat_id,
                                'obsType': obs_label,
                                'obsValue': float(val_str)
                            })
                                
        line_idx += 1
        
    return pd.DataFrame(observations), header_info

# --- OTOMATİK ÇALIŞTIRMA ---
if __name__ == "__main__":
    data_folder = "data"
    for file_name in os.listdir(data_folder):
        if file_name.lower().endswith(".rnx"):
            print(f"\n>>> {file_name} işleniyor...")
            df_snr, header = parse_rinex_obs(os.path.join(data_folder, file_name))
            
            if not df_snr.empty:
                # Saniye ve dakika görünecek şekilde formatlıyoruz
                df_display = df_snr.copy()
                df_display['epoch'] = df_display['epoch'].dt.strftime('%Y-%m-%d %H:%M:%S')
                print(df_display.head(10))
                print(f"Bitti! Toplam {len(df_snr)} adet SNR verisi başarıyla çekildi.")