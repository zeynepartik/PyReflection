import numpy as np
from astropy.timeseries import LombScargle

def detrend_snr(elevation, snr_value):
    """
    SNR verisindeki düşük frekanslı trendi 2. dereceden polinom ile temizlemek için 
    Girdiler: elevation (derece), snr_value (Lineer veya dB SNR)
    Çıktı: detrended_snr
    """
    # Polinom uydururken x ekseni elevation (derece), y ekseni SNR değeridir
    poly_coeffs = np.polyfit(elevation, snr_value, deg=2)
    
    # Uydurulan polinomun o noktalardaki değerlerini hesapla (Trend)
    trend = np.polyval(poly_coeffs, elevation)
    
    # Gerçek sinyalden trendi çıkararak saf salınımı (detrended SNR) elde et
    detrended_snr = snr_value - trend
    return detrended_snr


def calculate_lombscargle(elevation, detrended_snr, wavelength, h_min=0.0, h_max=20.0, precision=0.01):
    """
    girdilerle Astropy tabanlı Lomb-Scargle Periyodogramı hesaplar.
    Frekans-Yükseklik dönüşümünü f = 2h / lambda bağıntısıyla kurar.
    
    GİRDİLER:
        - elevation: nx1 boyutunda yükseklik açısı (derece)
        - detrended_snr: nx1 boyutunda trendden arındırılmış SNR sinyali
        - wavelength: İlgili sinyalin dalga boyu (metre)
        - h_min, h_max: Aranacak yükseklik sınırları (metre)
        - precision: Yükseklik grid aralığı/hassasiyeti (metre, örn: 0.01 m)
        
    ÇIKTILAR:
        - h_grid: mx1 boyutunda arama yükseklikleri dizisi
        - lsp_power: mx1 boyutunda her yüksekliğe karşılık gelen periyodogram gücü
    """
    # Bağımsız değişkenimiz x = sin(elevation) olmalı.
    # Elevation derece cinsinden geldiği için önce radyana çeviriyoruz
    x = np.sin(np.radians(elevation))
    y = detrended_snr

    # hassasiyete (precision) göre yükseklik arama matrisini (h_grid) kurguluyoruz
    h_grid = np.arange(h_min, h_max + precision, precision)

    #  f = 2h / wavelength ilişkisini kullanarak frekans gridini türetiyoruz
    # Astropy frekans dizisi beklediği için yükseklik adımlarını doğrudan frekansa çeviriyoruz
    freq_grid = (2 * h_grid) / wavelength

    # 4. Astropy LombScargle motorunu ateşliyoruz (Verileri fit ediyoruz)
    ls = LombScargle(x, y, fit_mean=True)
    
    # Belirlediğimiz frekans gridine göre güç (power) matrisini hesaplatıyoruz
    lsp_power = ls.power(freq_grid)

    return h_grid, lsp_power