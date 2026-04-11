# Prison Video Analytics MVP (Fight + Monitoring)

Bu repo, cezaevi ortamı için **erken uyarı** amaçlı, modüler bir video analiz MVP'sidir.

## Önemli Uyarı (Gerçekçilik)

- Bu sistem **kesin** “kavga oldu / intihar oldu” kararı vermez.
- Sistem “**riskli / anormal davranış tespit edildi**” şeklinde **erken uyarı** üretir.
- Gerçek ortamda **kalibrasyon** (kamera açısı, FPS, ışık, alan ölçeği, eşikler/ağırlıklar) zorunludur.

## Özellikler

- Video dosyası veya RTSP stream
- İnsan tespiti + pose (Ultralytics YOLO Pose)
- Basit CPU dostu tracking (centroid + greedy assignment)
- Feature engineering (hareket, etkileşim, agitasyon, postür, hareketsizlik, düşme, self-harm benzeri metrikler)
- Mode bazlı risk motoru:
  - `FIGHT`: koğuş kavga analizi
  - `MONITORING`: gözetimli oda (bayılma/çökme, aşırı hareketsizlik, agresyon, self-harm risk)
- Alarm + JSON event log + olay klibi kaydı (pre/post buffer, cooldown)
- OpenCV UI overlay + panel
- Debug mode: risk bileşenlerini açıklar

## Kurulum

Python 3.10+ önerilir.

```bash
cd project
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## (Opsiyonel) Fight modeli eğitimi (RWF2000 vb.)

Bu repo iki katmanlı çalışır:
- Heuristik/pipeline her zaman çalışır (pose + tracking + risk engine).
- `FIGHT` modundaki **clip sınıflandırıcı** (MODEL(FIGHT)) yalnızca `models/fight_mobilenetv3_small.pth` mevcutsa aktiftir.

Eğitim (örnek):

```bash
cd project
python train_rwf2000.py --data_root "C:\\path\\to\\RWF2000" --out "models\\fight_mobilenetv3_small.pth" --epochs 3
```

## Çalıştırma

1) `project/data/input/` içine bir video koyun (örn: `sample.mp4`)

2) `project/app/config.py` içindeki `VIDEO_SOURCE` ayarını güncelleyin:
- Dosya: `VIDEO_SOURCE = {"type": "file", "path": "data/input/sample.mp4"}`
- RTSP: `VIDEO_SOURCE = {"type": "rtsp", "url": "rtsp://..."}`

3) Çalıştırın:

```bash
python -m app.main
```

## Çıktılar

- Event JSON logları: `project/data/logs/`
- Olay klipleri: `project/data/clips/`

## Notlar / TODO

- Risk eşikleri ve ağırlıkları sahaya göre ayarlanmalı.
- Self-harm ve “fight” ayrımı için davranış modelleri ve domain kalibrasyonu gerekir.
- Tracker daha güçlü hale getirilebilir (ByteTrack/DeepSORT gibi). MVP’de basit ve CPU dostu tutuldu.
