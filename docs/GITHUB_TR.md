# GitHub'a yükleme

Bu arşiv tüm projenin güncel halidir; önceki küçük arayüz güncelleme paketleri değildir.
Çalışan Mac klasörünü silme. Arşivi ayrı bir yere açarak GitHub kaynağı olarak kullanabilirsin.

1. GitHub'da yeni repository oluştur: creditlens.
2. Description: Explainable credit risk platform with XGBoost, SHAP, calibration, policy simulation and drift monitoring.
3. CV'den erişilebilmesi için Public seç. Yerel README ve LICENSE zaten var; GitHub'da ek README oluşturma.
4. VS Code terminalinde arşivden çıkardığın proje klasörüne geç.
5. GitHub kullanıcı adını kendi hesabınla değiştirerek çalıştır:

```bash
git init -b main
git add .
git commit -m "Add CreditLens explainable credit risk platform"
git remote add origin https://github.com/YOUR_USERNAME/creditlens.git
git push -u origin main
```

GitHub oturum açma isteyebilir. Kimlik bilgilerini terminal veya resmi GitHub arayüzünde gir.
Depoda zaten git geçmişi varsa git init veya remote add komutlarını tekrar çalıştırma.

Yükledikten sonra README'deki iki ekran görüntüsünü, English bağlantısını ve Actions sonucunu kontrol et.
GitHub Actions çalışması burada önceden tamamlanmış değildir; başarısını gözlemlemeden geçtiğini yazma.

Önerilen repository topics: python, machine-learning, credit-risk, xgboost, shap, explainable-ai, streamlit, docker, model-monitoring.
