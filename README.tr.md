# CreditLens · Açıklanabilir Kredi Riski

[Deutsch](README.md) | [English](README.en.md) | **Türkçe**

Mevcut kredi kartı müşterilerinin bir sonraki ay ödeme temerrüdü riskini analiz eden etkileşimli Python uygulaması. CreditLens; makine öğrenmesini, müşteriye özel açıklamaları, örnek maliyet simülasyonunu ve veri izlemeyi bir araya getirir.

**Geliştiren:** Cenker Sule · Yapay Zekâ ve Veri Bilimi

## Özellikler

- Lojistik regresyon, XGBoost ve sabit temerrüt oranı referansının karşılaştırılması.
- Birbirinden ayrı eğitim, kalibrasyon, doğrulama ve test veri bölümleri.
- Gösterilen kalibre edilmiş tahmini yeniden oluşturan SHAP açıklamaları.
- İnceleme ve kaçırılan temerrüt maliyetleriyle etkileşimli eşik analizi.
- Toplu CSV tahminleri, girdi doğrulama ve Popülasyon Kararlılık Endeksi (PSI) ile izleme.
- Almanca, İngilizce ve Türkçe arayüz; varsayılan dil Almancadır.
- Tekrar üretilebilir model dosyaları, otomatik testler ve Docker yapılandırması.

## Uygulama görünümü

![CreditLens Türkçe portföy özeti](docs/screenshots/overview-tr.jpeg)

*Türkçe arayüzde portföy özeti. Almanca ve İngilizce de dil menüsünden seçilebilir.*

## Ölçülen sonuçlar

Ayrı tutulan test verisi: **4.480 müşteri**, **%21,56** temerrüt oranı. XGBoost yapılandırması test başarısına göre değil, doğrulama verisindeki ortalama kesinliğe (AP) göre seçilmiştir.

| Model | ROC-AUC | Ortalama kesinlik (AP) | Brier skoru ↓ |
| --- | ---: | ---: | ---: |
| Sabit temerrüt oranı referansı | 0,500 | 0,216 | 0,1691 |
| Lojistik regresyon | 0,722 | 0,486 | 0,1441 |
| XGBoost, ham | 0,780 | 0,551 | 0,1331 |
| XGBoost, sigmoid kalibrasyonlu | 0,780 | 0,551 | 0,1332 |

Kalibrasyon bu çalıştırmada test Brier skorunu **iyileştirmemiştir**. İki sürüm de raporlanır; kalibre edilmiş sürüm önceden tanımlanmış demo akışında kullanılır. Monoton sigmoid kalibrasyonu sıralamayı ve dolayısıyla ROC-AUC ile AP'yi korur.

%95 bootstrap aralıkları: ROC-AUC **0,763–0,797**, AP **0,516–0,585**. Bunlar sabit test tahminlerindeki belirsizliği gösterir; tüm eğitim sürecinin belirsizliğini kapsamaz. Kaçırılan temerrüt başına 5, incelenen müşteri başına 1 maliyet birimi varsayımıyla doğrulamada seçilen eşik **0,22**'dir.

## Docker ile çalıştırma

Docker Desktop'ı kurup Docker motorunu başlat. Eğitimli modeller ve kaynak veri pakete dahildir. İlk imaj derlemesi internet gerektirir.

```bash
docker compose up --build
```

[http://localhost:8501](http://localhost:8501) adresini aç. Durdurmak için `Ctrl+C`, konteyneri kaldırmak için `docker compose down` kullan.

## Python ile çalıştırma

Python 3.12 önerilir. Apple Silicon üzerinde XGBoost için `brew install libomp` gerekebilir.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m streamlit run app.py
```

## Yeniden eğitim ve test

```bash
python -m credit_risk.train
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Eğitim, UCI kaynak dosyasının SHA-256 özetini kontrol eder; dosya yoksa UCI'dan indirir. `artifacts/report.json`, veri bölümü kimlikleri, tahminler, eğitim referans profilleri ve modeller yeniden oluşturulur. Doğrudan bağımlılıklar sürüme sabitlenmiştir; platformlar arasında küçük sayısal farklar olabilir.

## Proje yapısı

| Dosya / klasör | Amaç |
| --- | --- |
| `app.py` | Üç dilli Streamlit paneli |
| `src/credit_risk/data.py` | Veri edinme, doğrulama ve gruplu bölme |
| `src/credit_risk/train.py` | Eğitim, kalibrasyon, seçim ve değerlendirme |
| `src/credit_risk/model.py` | Tahmin ve toplamsal SHAP açıklamaları |
| `src/credit_risk/evaluation.py` | Metrikler, politika maliyetleri ve alt grup incelemesi |
| `src/credit_risk/monitoring.py` | PSI ve eğitim aralığı izleme |
| `artifacts/report.json` | Ölçülen sonuçların tamamı |
| `docs/model_card.md` | Model varsayımları ve sınırlamaları |
| `docs/START_HERE_TR.md` | Türkçe hızlı başlangıç rehberi |

## Beş dakikalık demo

1. Portföy özetinde model başarısını ve risk dağılımını incele.
2. Örnek müşteri seçip müşteriye özel SHAP katkılarına bak.
3. Örnek politika simülatöründe maliyetleri ve risk eşiğini değiştir.
4. Veri izleme ekranında test portföyünü yapay olarak değiştirilmiş kredi limitleriyle karşılaştır.
5. `data/batch_example.csv` dosyasını yükleyip tahminleri indir. On satırlık örnek, işlevi gösterir; güvenilir veri kayması yorumu için yeterli değildir.

## Veri ve sınırlamalar

**Yeh, I. (2009), Default of Credit Card Clients, UCI Machine Learning Repository**, [DOI: 10.24432/C55S3H](https://doi.org/10.24432/C55S3H). Tayvan'daki 30.000 müşterinin Nisan–Eylül 2005 ödeme geçmişidir. Veri lisansı: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Tutarlar **NT$** cinsindedir; EUR değildir. Kaynak ve dönüşümler: [data/README.md](data/README.md).

Cinsiyet, yaş, eğitim, medeni durum ve kimlik numarası model girdilerinden çıkarılmıştır. Cinsiyet ve yaş yalnızca betimleyici alt grup değerlendirmesinde tutulur. Bu değişkenlerin çıkarılması adilliği kanıtlamaz ve dolaylı ayrımcılık olasılığını ortadan kaldırmaz.

Veri, güncel Alman banka müşterilerini temsil etmez. Zamansal veya dış veriyle doğrulama yapılmamıştır. Bu portföy araştırması, mevcut kart sahiplerinin bir sonraki ay temerrüt riskini tahmin eder; üretimde kullanılan bir kredi karar sistemi değildir. İnceleme gerçek kayıpları otomatik olarak önlemez; maliyet simülasyonu örnek varsayımlar kullanır.

## Doğrulama durumu

**26 otomatik test geçti:** veri bölümü bütünlüğü, girdi doğrulama, SHAP toplamsallığı, maliyet hesabı, veri kayması davranışı, tahmin dosyaları ve üç dilde beş ekran test edildi.

Uygulama Mac üzerinde Docker ile çalıştırılıp Safari ekran görüntüleriyle incelendi. İlk konteynerdeki paket yolu sorunu, Dockerfile içinde düzenlenebilir kurulumla giderildi. GitHub Actions akışı testleri, Docker derlemesini ve konteyner sağlık kontrolünü tanımlar; gerçek GitHub çalışmasının sonucu Actions ekranından kontrol edilmelidir.

Görsel güncellemeler model dosyalarını, hesaplamaları veya eşikleri değiştirmez. Örneğin doğrulama simülasyonunda maliyetler 14/4 ve eşik 0,28 olduğunda inceleme oranı %25,6, kaçırılan temerrüt 430 ve müşteri başına maliyet 2,344 birimdir.

Kod: [MIT](LICENSE). Kaynak ve türetilmiş veriler: CC BY 4.0.
