# Mac'te başlangıç

1. ZIP'i aç. `credit-risk-platform` klasörünü VS Code ile aç.
2. Docker Desktop'ı başlat. Engine çalışır duruma gelsin.
3. VS Code → Terminal → New Terminal. Terminal proje klasöründe olmalı.
4. Aşağıdaki komutu çalıştır:

```bash
docker compose up --build
```

İlk çalıştırma paketleri indireceği için zaman alabilir. Ardından tarayıcıda http://localhost:8501 adresini aç.

Panel önce Almanca açılır. Sol üstten English veya Türkçe seçebilirsin.

- Portfolioübersicht: model sonuçları ve grafikler.
- Einzelfallanalyse: müşteri profili, risk tahmini ve SHAP açıklaması.
- Risikosimulation: eşik ve örnek hata maliyetleri.
- Datenmonitoring: CSV yükleme, toplu tahmin ve veri değişimi.
- Methodik & Modell: karşılaştırma, kalibrasyon ve sınırlar.

Kapatmak için terminalde Ctrl+C. Sonra `docker compose down` çalıştırabilirsin.

26 otomatik test geçti. Uygulama Mac üzerinde Docker ve Safari ile çalıştırılıp beş ekran görüntüleriyle kontrol edildi. README'ler Almanca ve İngilizce hazır. Modelin mantığını ve tarihî veri setinin sınırlarını mülakat için çalış. GitHub Actions sonucu ancak GitHub'da gerçekten çalıştıktan sonra belli olur.
