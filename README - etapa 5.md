# 📘 README – Etapa 5: Configurarea și Antrenarea Modelului RN

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Corjuc Ioan Marian  
**Data predării:** 18.12.2025

---

## Scopul Etapei 5

Această etapă corespunde punctului **6. Configurarea și antrenarea modelului RN** din lista de 9 etape.

**Obiectiv principal:** Antrenarea efectivă a modelului YOLOv11-seg definit în Etapa 4, evaluarea performanței și integrarea în aplicația completă de aterizare sigură pentru drone.

---

## 1. Verificare Prerequisite Etapa 4

- [x] **State Machine** definit și documentat în `docs/state_machine.png`
- [x] **Contribuție 100% date originale** - imagini capturate la competiția Student AirRace 2025
- [x] **Modul 1 (Data Acquisition)** funcțional - `src/init_project.py`
- [x] **Modul 2 (RN)** cu arhitectură YOLOv11-seg definită
- [x] **Modul 3 (UI/Web Service)** funcțional - Streamlit app

---

## 2. Setul de Date pentru Antrenare

### 2.1 Distribuția Claselor în Dataset

| **Clasă** | **Număr Instanțe** | **Procent** | **Tip** |
|-----------|-------------------|-------------|---------|
| Grass_Zone | 559 | 49.6% | UNSAFE |
| Paved_Zone | 291 | 25.8% | **SAFE** |
| No_Fly_Zone | 276 | 24.5% | UNSAFE |
| **TOTAL** | **1126** | 100% | - |

### 2.2 Observații Dataset

- **Dezechilibru moderat:** Grass_Zone are aproape dublu față de celelalte clase
- **Clase UNSAFE dominante:** 74.1% din instanțe sunt zone nesigure (Grass + No_Fly)
- **Doar 1 clasă SAFE:** Paved_Zone (beton/asfalt) cu 25.8%

---

## 3. Configurația de Antrenare

### 3.1 Tabel Hiperparametri și Justificări

| **Hiperparametru** | **Valoare Aleasă** | **Justificare** |
|--------------------|-------------------|-----------------|
| **Arhitectură** | YOLOv11s-seg | Varianta "small" oferă echilibru între viteză și acuratețe pentru aplicații real-time pe dronă |
| **Epoci planificate** | 50 | Suficient pentru convergență, cu early stopping pentru prevenire overfitting |
| **Epoci efectiv rulate** | 39 | Early stopping activat (patience=5) când val_loss nu a mai scăzut |
| **Batch size** | 16 | Optim pentru 1126 instanțe: ~70 iterații/epocă, echilibru memorie/stabilitate |
| **Image size** | 640×640 | Standard YOLO pentru detecție/segmentare |
| **Optimizer** | AdamW | Weight decay integrat, convergență mai rapidă decât SGD |
| **Learning rate inițial** | 0.01 | Valoare standard pentru AdamW cu cosine annealing |
| **Learning rate final** | 0.01 | Menținut constant după warmup |
| **Cosine LR Scheduler** | Activat | Scădere graduală a LR pentru fine-tuning în epocile finale |
| **Early Stopping** | patience=5 | Oprește antrenarea dacă mAP50 nu crește în 5 epoci consecutive |

### 3.2 Augmentări Specifice Domeniului (Drone/Aerial)

| **Augmentare** | **Valoare** | **Justificare pentru Drone** |
|----------------|-------------|------------------------------|
| HSV Hue | 0.015 | Variații minime culoare - condițiile de lumină schimbătoare |
| HSV Saturation | 0.7 | Simulare saturație diferită în funcție de ora zilei |
| HSV Value | 0.4 | Variații luminozitate - umbre, nori, soare direct |
| Perspective | 0.0005 | Simulare unghi cameră variabil în zbor |
| Mosaic | 1.0 (100%) | Combinare 4 imagini - învață obiecte la scale diferite |
| MixUp | 0.1 (10%) | Suprapunere imagini pentru regularizare |
| Rotation | ±10° | Simulare instabilitate dronă la vânt |
| Horizontal Flip | 0.5 (50%) | Drona poate aborda din orice direcție |
| Scale | 0.5 | Variație altitudine - obiecte mai mari/mici |

---

## 4. Rezultatele Antrenării

### 4.1 Evoluția Loss-urilor pe 39 Epoci

| **Metrică** | **Epoch 1** | **Epoch 20** | **Epoch 39 (Final)** |
|-------------|-------------|--------------|---------------------|
| train/box_loss | 1.949 | 1.351 | 1.147 |
| train/seg_loss | 5.404 | 3.386 | 2.962 |
| train/cls_loss | 2.859 | 1.296 | 1.041 |
| val/box_loss | N/A | 1.567 | 1.526 |
| val/seg_loss | N/A | 3.293 | 2.489 |
| val/cls_loss | N/A | 1.515 | 1.198 |

**Observații:**
- Convergență solidă pe toate loss-urile
- Diferența train/val moderată → nu există overfitting sever
- Modelul a învățat bine task-ul de segmentare (seg_loss scăzut de la 5.4 → 2.9)

### 4.2 Evoluția Metricilor de Performanță

| **Metrică** | **Epoch 10** | **Epoch 20** | **Epoch 32 (Best)** | **Epoch 39 (Final)** |
|-------------|--------------|--------------|---------------------|---------------------|
| Precision (Mask) | 0.387 | 0.708 | 0.855 | 0.826 |
| Recall (Mask) | 0.369 | 0.619 | 0.697 | 0.734 |
| mAP50 (Mask) | 0.347 | 0.646 | 0.754 | 0.775 |
| mAP50-95 (Mask) | 0.126 | 0.291 | 0.422 | 0.444 |

### 4.3 Metrici Finale pe Validare (Epoch 39)

| **Metrică** | **Valoare** | **Target Nivel 1** | **Target Nivel 2** | **Status** |
|-------------|-------------|-------------------|-------------------|------------|
| **Precision (Mask)** | **82.6%** | ≥65% | ≥75% | ✅ Nivel 2 |
| **Recall (Mask)** | **73.4%** | ≥60% | ≥70% | ✅ Nivel 2 |
| **mAP50 (Mask)** | **77.5%** | ≥65% | ≥75% | ✅ Nivel 2 |
| **mAP50-95 (Mask)** | **44.4%** | - | - | - |
| **F1-score (all classes)** | **0.77** | ≥0.60 | ≥0.70 | ✅ Nivel 2 |

---

## 5. Analiza F1-Score per Clasă

Din curba F1-Confidence (la threshold optim 0.361):

| **Clasă** | **F1-score Maxim** | **Threshold Optim** | **Observații** |
|-----------|-------------------|---------------------|----------------|
| No_Fly_Zone | ~0.95 | 0.3-0.5 | **Cea mai bună** - contrast vizual clar |
| Grass_Zone | ~0.78 | 0.3-0.4 | Bună - textură distinctivă |
| Paved_Zone | ~0.65 | 0.3-0.4 | **Cea mai slabă** - confuzie cu background |
| **All Classes** | **0.77** | **0.361** | Performanță solidă |

---

## 6. Analiza Confusion Matrix

### 6.1 Matricea de Confuzie Normalizată

|  | **Grass_Zone (True)** | **No_Fly_Zone (True)** | **Paved_Zone (True)** | **Background (True)** |
|--|----------------------|----------------------|---------------------|---------------------|
| **Grass_Zone (Pred)** | **81%** | 0% | 2% | 44% |
| **No_Fly_Zone (Pred)** | 0% | **96%** | 2% | 9% |
| **Paved_Zone (Pred)** | 4% | 2% | **65%** | 47% |
| **Background (Pred)** | 15% | 2% | 32% | 0% |

### 6.2 Interpretare per Clasă

**🏆 Clasa cu cea mai bună performanță: No_Fly_Zone**
- Recall: **96%** (aproape perfectă)
- Confuzii minime cu alte clase
- **Motivație:** Contrast vizual foarte clar - zonele No_Fly conțin obstacole (steaguri, bariere) care se disting clar de teren

**⚠️ Clasa cu cea mai slabă performanță: Paved_Zone**
- Recall: **65%** 
- **32%** din Paved_Zone este clasificat greșit ca Background
- **47%** din Background este clasificat ca Paved_Zone
- **Motivație:** Zonele pavate (asfalt/beton) au textură similară cu unele zone de background, mai ales când sunt văzute de la altitudine mare

**📊 Grass_Zone**
- Recall: **81%** - bună
- **15%** clasificat ca Background
- **44%** din Background clasificat ca Grass_Zone
- **Motivație:** Iarba are textură distinctivă dar la margini se confundă cu background

---

## 7. Analiza Erorilor în Context Industrial (Drone Landing)

### 7.1 Pe ce clase greșește cel mai mult modelul?

Modelul confundă **Paved_Zone** cu **Background** în 32% din cazuri. Aceasta este o problemă critică deoarece:
- Paved_Zone este **singura clasă SAFE** pentru aterizare
- Nedetectarea zonei pavate = drona nu va ateriza când ar putea

**Cauze identificate:**
1. Zonele pavate mici (de la altitudine mare) au textură similară cu background-ul
2. Umbrele pe asfalt reduc contrastul
3. Marcajele rutiere pot fragmenta zona pavată

### 7.2 Ce caracteristici ale datelor cauzează erori?

1. **Confuzie Paved_Zone ↔ Background:**
   - Zonele mici de beton sunt greu de distins de background
   - Reflexiile solare pe asfalt creează artefacte

2. **Confuzie Grass_Zone ↔ Background:**
   - Iarba la marginea cadrului se confundă cu background
   - Vegetația uscată are textură ambiguă

3. **No_Fly_Zone funcționează excelent:**
   - Obstacolele (steaguri, bariere) au forme și culori distincte
   - Contrast ridicat cu mediul înconjurător

### 7.3 Implicații pentru Aterizarea Dronei

| **Tip Eroare** | **Frecvență** | **Consecință** | **Severitate** |
|----------------|---------------|----------------|----------------|
| Paved_Zone → Background (FN) | 32% | Drona NU aterizează când ar putea | **MEDIE** - conservator |
| Background → Paved_Zone (FP) | 47% | Drona crede că poate ateriza unde nu există zonă | **CRITICĂ** |
| Grass → Background | 15% | Zonă unsafe nedetectată | MICĂ |
| No_Fly → Orice | 4% | Obstacol nedetectat | **CRITICĂ** |

**Prioritate:** Minimizare False Positives pentru Paved_Zone (să nu credem că e safe când nu e)

### 7.4 Măsuri Corective Propuse

1. **Colectare date adiționale:**
   - Mai multe imagini cu zone pavate mici/parțiale
   - Imagini cu umbre și reflexii pe asfalt

2. **Ajustare threshold clasificare:**
   - Creștere threshold pentru Paved_Zone de la 0.361 → 0.5
   - Accept doar predicții cu confidence >50% pentru zonă sigură

3. **Post-procesare în aplicație:**
   - Verificare consistență temporală (mai multe frame-uri consecutive)
   - Zona SAFE doar dacă detectată în >3 frame-uri consecutive

4. **Augmentări suplimentare:**
   - Mai multe variații de luminozitate pentru asfalt
   - Augmentări cu umbre artificiale

---

## 8. Integrare în Aplicația UI

### 8.1 Model Încărcat

```python
# src/app/main.py - încarcă automat cel mai recent model antrenat
def find_latest_model():
    models_dir = Path("models")
    all_models = list(models_dir.rglob("best.pt"))
    latest_model = max(all_models, key=os.path.getmtime)
    return str(latest_model)

model = YOLO(find_latest_model())  # Încarcă best.pt din ultima antrenare
```

### 8.2 Logica de Siguranță

```python
# Doar Paved_Zone (clasa 2) este considerată SAFE
SAFE_CLASSES = [2]  # Paved_Zone

if cls_id in SAFE_CLASSES:
    color = [0, 255, 0]  # VERDE - Safe
    label_text = f"✅ {class_name} (SAFE)"
else:
    color = [255, 0, 0]  # ROȘU - Unsafe
    label_text = f"⛔ {class_name} (UNSAFE)"
```

### 8.3 Demonstrație Inferență Reală

Screenshot salvat în: `docs/screenshots/inference_real.png`

**Verificare funcționalitate:**
- [x] Modelul încarcă weights antrenate (nu random)
- [x] Predicțiile sunt consistente și corecte
- [x] Confidence scores au valori realiste (0.3-0.9)
- [x] Overlay vizual funcționează corect

---

## 9. Structura Fișierelor Generate

```
models/
├── drone_landing_lvl2/
│   └── weights/
│       ├── best.pt          # ← Model final (folosit în producție)
│       └── last.pt          # Model ultima epocă

results/
├── training_history.csv     # Toate 39 epocile
├── hyperparameters.yaml     # Configurația de antrenare
└── test_metrics.json        # Metrici finale

docs/
├── loss_curve.png           # Grafic loss vs epochs (results.png)
├── confusion_matrix.png     # Matricea de confuzie
└── screenshots/
    └── inference_real.png   # Demo UI cu model antrenat
```

---

## 10. Instrucțiuni de Rulare

### Antrenare Model
```bash
python src/neural_network/train_yolo.py
# Output: models/drone_landing_lvl2/weights/best.pt
```

### Evaluare pe Test Set
```bash
python src/neural_network/evaluate.py
# Output: results/test_metrics.json, docs/confusion_matrix.png
```

### Lansare UI
```bash
streamlit run src/app/main.py
# Deschide http://localhost:8501
```

---

## 11. Checklist Final Etapa 5

### Antrenare Model - Nivel 1
- [x] Model antrenat de la zero (YOLOv11s-seg)
- [x] 39 epoci rulate (early stopping activat)
- [x] Tabel hiperparametri cu justificări completat
- [x] Metrici calculate: **Precision 82.6%**, **F1 0.77**
- [x] Model salvat în `models/drone_landing_lvl2/weights/best.pt`

### Nivel 2 - Recomandat
- [x] Early Stopping implementat (patience=5)
- [x] Cosine LR Scheduler activat
- [x] Augmentări specifice domeniului drone
- [x] Grafic loss salvat în `docs/loss_curve.png`
- [x] Analiză erori în context industrial completată
- [x] Metrici Nivel 2: **Precision ≥75%** ✅, **F1 ≥0.70** ✅

### Integrare UI
- [x] Model ANTRENAT încărcat în UI
- [x] Inferență REALĂ funcțională
- [x] Screenshot demonstrativ

---

## 12. Concluzii Etapa 5

Modelul YOLOv11s-seg a fost antrenat cu succes pe dataset-ul custom de imagini aeriene, atingând performanțe care depășesc cerințele Nivel 2:

| **Metrică** | **Obținut** | **Target Nivel 2** |
|-------------|-------------|-------------------|
| Precision | 82.6% | ≥75% ✅ |
| Recall | 73.4% | ≥70% ✅ |
| mAP50 | 77.5% | ≥75% ✅ |
| F1-score | 0.77 | ≥0.70 ✅ |

**Puncte forte:**
- Detecție excelentă No_Fly_Zone (96% recall)
- Convergență stabilă fără overfitting
- Augmentări relevante pentru scenarii de zbor

**Limitări identificate:**
- Paved_Zone are recall mai scăzut (65%)
- Confuzie cu background-ul necesită atenție în Etapa 6

**Următorii pași (Etapa 6):**
- Experimente de optimizare pentru Paved_Zone
- Ajustare threshold pentru reducere False Positives
- Export ONNX pentru deployment pe hardware drone
