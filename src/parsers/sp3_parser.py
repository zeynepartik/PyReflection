import pandas as pd
import datetime
import os  # Bilgisayarın dosya sistemine (klasörlere) bakmak için kullanılan kütüphane

def parse_sp3(file_path):
    """
    Bu fonksiyon bir SP3 dosyasını okur, uyduların koordinatlarını (X,Y,Z)
    metre cinsinden bir tabloya (DataFrame) dönüştürür.
    """
    data = []
    
    # Dosyayı okumak için açıyoruz
    with open(file_path, 'r') as f:
        # Dosyanın her bir satırını sırayla okuyoruz
        for line in f:
            
            # Eğer satır '*' ile başlıyorsa: Yeni bir zaman (epok) dilimi bilgisidir
            if line.startswith('*'):
                parts = line.split()
                # parts[1]=Yıl, [2]=Ay, [3]=Gün, [4]=Saat, [5]=Dakika, [6]=Saniye
                # parts[6] ondalıklı bir metin ("0.00") olduğu için önce sayıya (float),
                # sonra tam sayıya (int) çevirerek datetime nesnesi oluşturuyoruz  
                current_epoch = datetime.datetime(
                    int(parts[1]), int(parts[2]), int(parts[3]),
                    int(parts[4]), int(parts[5]), int(float(parts[6]))
                )
            
            # Eğer satır 'P' ile başlıyorsa: Uydu koordinat verisidir  
            elif line.startswith('P'):
                # Satırdaki sabit yerlerden (indekslerden) bilgileri çekiyoruz:
                sat_id = line[1:4].strip()  # 1 ile 4. karakter arası uydu adı (G01 vb.)  
                constellation = sat_id[0]  # Takımyıldız (G=GPS, R=GLONASS, E=Galileo, C=BeiDou)

                # Koordinatlar kilometreden metreye çevriliyor (* 1000)  
                x = float(line[4:18]) * 1000.0  # X koordinatı  
                y = float(line[18:32]) * 1000.0 # Y koordinatı  
                z = float(line[32:46]) * 1000.0 # Z koordinatı 
                
                # Her uyduyu o anki zaman bilgisiyle beraber listeye kaydediyoruz
                data.append({
                    'epoch': current_epoch,
                    'constellation': constellation, # Takımyıldız
                    'satID': sat_id,
                    'X': x, 'Y': y, 'Z': z
                })
    
    # Elde ettiğimiz tüm listeyi bir Pandas tablosuna çevirip teslim ediyoruz  
    return pd.DataFrame(data)

# --- OTOMATİK ÇALIŞTIRMA VE TEST KISMI ---
# Bu dosya direkt çalıştırıldığında (main.py yerine burası çalıştığında) devreye girer
if __name__ == "__main__":
    # Verilerin olduğu klasörün ismini belirtiyoruz
    data_klasoru = "data"
    
    # Klasörün içindeki tüm dosyaların listesini alıyoruz
    bulunan_dosyalar = os.listdir(data_klasoru)
    
    # Her bir dosyayı tek tek kontrol ediyoruz
    for dosya_adi in bulunan_dosyalar:
        # Eğer dosya sonu '.SP3' ile bitiyorsa analiz et
        if dosya_adi.endswith(".SP3"):
            tam_yol = os.path.join(data_klasoru, dosya_adi)
            print(f"\n>>> {dosya_adi} dosyası analiz ediliyor...")
            
            # Yukarıda yazdığımız okuma fonksiyonunu çağırıyoruz
            df_sonuc = parse_sp3(tam_yol)
            
            df_yazdir = df_sonuc.copy()
            df_yazdir['epoch'] = df_yazdir['epoch'].dt.strftime('%Y-%m-%d %H:%M:%S')

            # Tablonun ilk 5 satırını ekrana basıp kontrol ediyoruz
            print(df_yazdir.head())
            print(f"Bitti! Bu dosyadan {len(df_sonuc)} adet koordinat satırı okundu.")