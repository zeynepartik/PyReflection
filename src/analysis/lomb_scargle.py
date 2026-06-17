import numpy as np


def detrend_snr(elevation, snr_value, poly_degree=2):
    """
    SNR verisindeki düşük frekanslı trendi polinom ile temizlemek için.
    Girdiler: elevation (derece), snr_value (Lineer veya dB SNR), poly_degree (polinom derecesi, varsayılan 2)
    Çıktı: detrended_snr
    """
    # Polinom uydururken x ekseni elevation (derece), y ekseni SNR değeridir
    poly_coeffs = np.polyfit(elevation, snr_value, deg=poly_degree)

    # Uydurulan polinomun o noktalardaki değerlerini hesapla (Trend)
    trend = np.polyval(poly_coeffs, elevation)

    # Gerçek sinyalden trendi çıkararak saf salınımı (detrended SNR) elde et
    detrended_snr = snr_value - trend
    return detrended_snr


def _get_ofac_hifac(x_inv_meters, h_max, precision):
    """
    Press et al. (1992) terminolojisiyle LSP frekans gridi için ofac/hifac hesaplar.
    Roesler & Larson (2018) GNSS-IR yaklaşımına göre.

    GİRDİLER:
        - x_inv_meters: X = sin(elev)/(lambda/2) örnekleme değişkeni (1/metre)
        - h_max: maksimum aranacak reflektör yüksekliği (metre)
        - precision: istenen yükseklik grid hassasiyeti (metre)

    ÇIKTILAR:
        - ofac: aşırı örnekleme (oversampling) faktörü
        - hifac: yüksek frekans faktörü
    """
    n = len(x_inv_meters)
    # Gözlem penceresi (span)
    window = x_inv_meters.max() - x_inv_meters.min()

    # Karakteristik tepe genişliği 1/W; bunu istenen hassasiyete bölerek ofac elde edilir
    ofac = (1.0 / window) / precision

    # Tüm örnekler eşit aralıklı olsaydı geçerli olan ortalama Nyquist frekansı
    fc = n / (2.0 * window)
    hifac = h_max / fc

    return ofac, hifac


def calculate_lombscargle(elevation, detrended_snr, wavelength, h_min=0.0, h_max=20.0, precision=0.01):
    """
    GNSS-IR için klasik (Press/Numerical Recipes 13.8) Lomb-Scargle periyodogramını
    hesaplar ve gücü spektral genliğe dönüştürür.

    Kaynak: Roesler, C. & Larson, K. M. (2018), GNSS-IR yazılım araçları (lomb.m / lombGIRAS.m).
    SNR ~ A·cos(2·pi·H·X) modeli, X = sin(elev)/(lambda/2) örnekleme değişkenidir.
    Lomb içinde t = sin(elev) kullanılır ve f -> H dönüşümü H = f·lambda/2 ile yapılır.

    GİRDİLER:
        - elevation: nx1 boyutunda yükseklik açısı (derece)
        - detrended_snr: nx1 boyutunda trendden arındırılmış SNR sinyali
        - wavelength: İlgili sinyalin dalga boyu (metre)
        - h_min, h_max: Aranacak yükseklik sınırları (metre)
        - precision: Yükseklik grid aralığı/hassasiyeti (metre, örn: 0.01 m)

    ÇIKTILAR:
        - h_grid: mx1 boyutunda arama yükseklikleri dizisi (metre)
        - spectral_amplitude: mx1 boyutunda her yüksekliğe karşılık gelen spektral genlik
          (detrended_snr ile aynı birim)
    """
    elevation = np.asarray(elevation, dtype=float)
    y = np.asarray(detrended_snr, dtype=float)

    # NaN değerleri ve karşılık gelen elevation açılarını ele (lombGIRAS davranışı)
    valid = ~np.isnan(y)
    elevation = elevation[valid]
    y = y[valid]

    # --- ofac / hifac hesabı: X = sin(elev)/(lambda/2) ---
    cf = wavelength / 2.0
    x_inv_meters = np.sin(np.radians(elevation)) / cf
    ofac, hifac = _get_ofac_hifac(x_inv_meters, h_max, precision)

    # h_min'den minimum frekans
    minf = 2.0 * h_min / wavelength

    # --- Lomb çekirdeği: bağımsız değişken t = sin(elev) ---
    t = np.sin(np.radians(elevation))
    n = len(y)
    span = t.max() - t.min()

    mu = y.mean()
    s2 = y.var(ddof=1)  # MATLAB var() varsayılanı N-1
    y_centered = y - mu

    # Frekans gridi (MATLAB ":" operatörü ile aynı): minf+df : df : hifac*N/(2*span)
    df = 1.0 / (span * ofac)
    f_start = minf + df
    f_end = hifac * n / (2.0 * span)
    n_freq = int(np.floor((f_end - f_start) / df)) + 1
    f = f_start + df * np.arange(n_freq)

    w = 2.0 * np.pi * f  # (nf,)

    # Her frekans için tau (Numerical Recipes 13.8.4)
    wt2 = 2.0 * np.outer(w, t)  # (nf, N)
    tau = np.arctan2(np.sin(wt2).sum(axis=1), np.cos(wt2).sum(axis=1)) / (2.0 * w)

    wt = np.outer(w, t)              # (nf, N)
    wtau = (w * tau)[:, np.newaxis]  # (nf, 1)
    cterm = np.cos(wt - wtau)
    sterm = np.sin(wt - wtau)

    den_c = (cterm ** 2).sum(axis=1)
    den_s = (sterm ** 2).sum(axis=1)
    num_c = cterm @ y_centered
    num_s = sterm @ y_centered

    # Normalize edilmiş güç (klasik Lomb-Scargle)
    power = (num_c ** 2 / den_c + num_s ** 2 / den_s) / (2.0 * s2)

    # Gücü spektral genliğe çevir: A = 2·sqrt(s2·P/N)
    spectral_amplitude = 2.0 * np.sqrt(s2 * power / n)

    # Frekanstan reflektör yüksekliğine: H = f·lambda/2
    h_grid = f * wavelength / 2.0

    return h_grid, spectral_amplitude
