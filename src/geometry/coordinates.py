import numpy as np

# WGS84 Elipsoid Sabitleri (Dünya'nın geometrik modeli)
# Bu değerler GPS sisteminin temelini oluşturur.
A = 6378137.0          # Dünya'nın ekvator yarıçapı (metre)
F = 1 / 298.257223563  # Basıklık oranı
B = A * (1 - F)        # Kutuplar arası yarıçap
E2 = (A**2 - B**2) / A**2  # Birinci eksantriklik karesi

def ecef_to_geodetic(x, y, z):
    """
    ECEF (X, Y, Z) koordinatlarını Geodetik (Enlem, Boylam, Yükseklik)
    formatına dönüştürür. Bowring (1976) yöntemi kullanılmıştır.
    """
    # Boylam (Longitude) hesabı en basitidir
    lon = np.arctan2(y, x)
    
    # Enlem (Latitude) hesabı için yardımcı değişkenler
    p = np.sqrt(x**2 + y**2)
    theta = np.arctan2(z * A, p * B)
    
    # Eksantriklik karesinin (e'2) ikinci değeri
    e_prime_2 = (A**2 - B**2) / B**2
    
    # Enlem formülü
    lat = np.arctan2(z + e_prime_2 * B * np.sin(theta)**3,
                     p - E2 * A * np.cos(theta)**3)
    
    # Eğrilik yarıçapı (N)
    n = A / np.sqrt(1 - E2 * np.sin(lat)**2)
    
    # Elipsoidal Yükseklik (h)
    alt = (p / np.cos(lat)) - n
    
    # Radyan cinsinden çıkan sonuçları dereceye çeviriyoruz
    return np.degrees(lat), np.degrees(lon), alt

def ecef_to_enu(x_sat, y_sat, z_sat, x_ant, y_ant, z_ant):
    """
    Uydunun ECEF koordinatlarını, antene göre yerel ENU 
    (Doğu, Kuzey, Yukarı) koordinat sistemine çevirir.
    """
    # Önce antenin nerede olduğunu (Enlem/Boylam) bulmalıyız
    lat, lon, _ = ecef_to_geodetic(x_ant, y_ant, z_ant)
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    # Anten ile uydu arasındaki fark vektörü (Uzaydaki mesafe farkı)
    dx = x_sat - x_ant
    dy = y_sat - y_ant
    dz = z_sat - z_ant
    
    # ENU Dönüşüm Matrisi Çarpımı
    # Bu matris, uzaydaki farkı yerel "Doğu-Kuzey-Yukarı" yönlerine döndürür
    e = -np.sin(lon_rad) * dx + np.cos(lon_rad) * dy
    n = -np.sin(lat_rad) * np.cos(lon_rad) * dx - np.sin(lat_rad) * np.sin(lon_rad) * dy + np.cos(lat_rad) * dz
    u =  np.cos(lat_rad) * np.cos(lon_rad) * dx + np.cos(lat_rad) * np.sin(lon_rad) * dy + np.sin(lat_rad) * dz
    
    return e, n, u

def calculate_elevation_azimuth(e, n, u):
    """
    ENU koordinatlarından uydunun gökyüzündeki açılarını hesaplar.
    """
    # Elevation (Yükseklik Açısı): Ufuktan yukarı olan açı
    # u = yukarı yönü, e ve n ise yatay düzlem
    elevation = np.arctan2(u, np.sqrt(e**2 + n**2))
    
    # Azimuth (Semt Açısı): Kuzeyden saat yönünde olan açı
    azimuth = np.arctan2(e, n)
    
    # Radyandan dereceye çevir
    el_deg = np.degrees(elevation)
    az_deg = np.degrees(azimuth)
    
    # Azimuth'u 0-360 derece arasına sabitle
    if az_deg < 0:
        az_deg += 360
        
    return el_deg, az_deg