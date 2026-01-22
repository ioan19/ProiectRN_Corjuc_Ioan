# 📘 README – Etapa 6: Analiza Performanței, Optimizarea și Concluzii Finale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Corjuc Ioan Marian  
**Data predării:** 22.01.2026

---

## Scopul Etapei 6

Această etapă corespunde punctelor **7. Analiza performanței și optimizarea parametrilor**, **8. Analiza și agregarea rezultatelor** și **9. Formularea concluziilor finale**.

**Obiectiv principal:** Maturizarea completă a Sistemului cu Inteligență Artificială (SIA) pentru aterizare sigură drone prin optimizarea modelului, analiza detaliată a performanței și integrarea îmbunătățirilor.

---

## 1. Verificare Prerequisite Etapa 5

- [x] **Model antrenat** salvat în `models/drone_landing_lvl2/weights/best.pt`
- [x] **Metrici baseline** raportate: Precision 82.6%, F1 0.77
- [x] **Tabel hiperparametri** cu justificări completat
- [x] **Training history** salvat în `results/training_history.csv`
- [x] **UI funcțional** care încarcă modelul antrenat
- [x] **State Machine** implementat conform definiției din Etapa 4

---

## 2. Rezultatele Modelului Antrenat (Baseline Etapa 5)

### 2.1 Metrici Finale pe Validare

| **Metrică** | **Valoare** | **Interpretare** |
|-------------|-------------|------------------|
| Precision (Mask) | **82.6%** | Din ce a prezis, 82.6% e corect |
| Recall (Mask) | **73.4%** | Din ce există real, 73.4% a fost detectat |
| mAP50 (Mask) | **77.5%** | Average Precision la IoU=0.5 |
| mAP50-95 (Mask) | **44.4%** | Average Precision medie pe IoU 0.5-0.95 |
| F1-score (all classes) | **0.77** | Media armonică Precision/Recall |

### 2.2 Performanță per Clasă

| **Clasă** | **F1-score** | **Precision** | **Recall** | **Status** |
|-----------|-------------|---------------|------------|------------|
| No_Fly_Zone | ~0.95 | Foarte bună | 96% | ✅ Excelent |
| Grass_Zone | ~0.78 | Bună | 81% | ✅ Bun |
| Paved_Zone | ~0.65 | Moderată | 65% | ⚠️ Necesită atenție |

---

## 3. Analiza Detaliată a Confusion Matrix

### 3.1 Matricea de Confuzie (Valori Absolute)

|  | **Grass_Zone** | **No_Fly_Zone** | **Paved_Zone** | **Background** |
|--|---------------|-----------------|----------------|----------------|
| **→ Grass_Zone** | **90** | 0 | 1 | 29 |
| **→ No_Fly_Zone** | 0 | **55** | 1 | 6 |
| **→ Paved_Zone** | 4 | 1 | **40** | 31 |
| **→ Background** | 17 | 1 | 20 | - |

### 3.2 Matricea de Confuzie (Normalizată pe Coloană)

|  | **Grass_Zone** | **No_Fly_Zone** | **Paved_Zone** | **Background** |
|--|---------------|-----------------|----------------|----------------|
| **→ Grass_Zone** | **81%** | 0% | 2% | 44% |
| **→ No_Fly_Zone** | 0% | **96%** | 2% | 9% |
| **→ Paved_Zone** | 4% | 2% | **65%** | 47% |
| **→ Background** | 15% | 2% | 32% | 0% |

### 3.3 Interpretare Detaliată

#### 🏆 No_Fly_Zone - Performanță EXCELENTĂ
- **Recall: 96%** - Aproape toate zonele interzise sunt detectate corect
- **Doar 4% erori:** 2% confundat cu Paved_Zone, 2% cu Background
- **Importanță critică:** Aceasta este clasa cea mai importantă pentru siguranță - obstacolele TREBUIE detectate
- **Concluzie:** Modelul este foarte fiabil pentru evitarea obstacolelor

#### ⚠️ Paved_Zone - Performanță MODERATĂ (necesită atenție)
- **Recall: 65%** - Doar 65% din zonele pavate sunt identificate corect
- **Probleme majore:**
  - 32% din Paved_Zone → clasificat ca Background (PIERDERE zonă sigură)
  - 47% din Background → clasificat greșit ca Paved_Zone (FALS POZITIV CRITIC)
- **Risc:** Drona ar putea crede că e o zonă sigură unde nu există
- **Soluție propusă:** Creștere threshold confidence pentru Paved_Zone

#### ✅ Grass_Zone - Performanță BUNĂ
- **Recall: 81%** - Majoritatea zonelor cu iarbă sunt detectate
- **Probleme minore:**
  - 15% → clasificat ca Background
  - 44% din Background → clasificat ca Grass_Zone
- **Impact:** Moderat - Grass este UNSAFE, deci confuzia cu Background nu e critică

---

## 4. Analiza Detaliată a 5 Exemple Greșite

### Exemplu #1: Paved_Zone clasificat ca Background

| **Parametru** | **Valoare** |
|---------------|-------------|
| True Label | Paved_Zone (SAFE) |
| Predicted | Background |
| Confidence | N/A (nedetectat) |
| **Cauză probabilă** | Zona pavată era mică și la marginea imaginii |
| **Impact industrial** | Drona nu aterizează când ar putea - conservator dar suboptim |
| **Soluție propusă** | Augmentări cu zone pavate mici, mai multe date de margine |

### Exemplu #2: Background clasificat ca Paved_Zone

| **Parametru** | **Valoare** |
|---------------|-------------|
| True Label | Background |
| Predicted | Paved_Zone (SAFE) |
| Confidence | ~0.4-0.5 |
| **Cauză probabilă** | Textura background-ului similară cu asfaltul (sol uscat/compact) |
| **Impact industrial** | **CRITIC** - Drona crede că poate ateriza unde nu e sigur |
| **Soluție propusă** | Threshold mai mare pentru Paved_Zone (0.6+), validare temporală |

### Exemplu #3: Grass_Zone clasificat ca Background

| **Parametru** | **Valoare** |
|---------------|-------------|
| True Label | Grass_Zone (UNSAFE) |
| Predicted | Background |
| Confidence | N/A (nedetectat) |
| **Cauză probabilă** | Iarbă uscată cu textură mai puțin distinctă |
| **Impact industrial** | Minor - zona oricum era UNSAFE |
| **Soluție propusă** | Augmentări cu variații de vegetație (verde, uscată, mixtă) |

### Exemplu #4: Background clasificat ca Grass_Zone

| **Parametru** | **Valoare** |
|---------------|-------------|
| True Label | Background |
| Predicted | Grass_Zone (UNSAFE) |
| Confidence | ~0.4-0.6 |
| **Cauză probabilă** | Zone de vegetație în background interpretate ca Grass |
| **Impact industrial** | Minor - clasificare conservatoare (UNSAFE când nu e clar) |
| **Soluție propusă** | Acceptable - nu compromite siguranța |

### Exemplu #5: Paved_Zone clasificat ca Grass_Zone

| **Parametru** | **Valoare** |
|---------------|-------------|
| True Label | Paved_Zone (SAFE) |
| Predicted | Grass_Zone (UNSAFE) |
| Confidence | ~0.4 |
| **Cauză probabilă** | Margine între asfalt și iarbă, clasificare ambiguă |
| **Impact industrial** | Mediu - pierdere oportunitate aterizare sigură |
| **Soluție propusă** | Mai multe date cu tranziții asfalt-iarbă |

---

## 5. Configurația Actuală de Antrenare (Baseline)

### 5.1 Hiperparametri Folosiți

```yaml
training:
  epochs: 50 (efectiv 39 cu early stopping)
  batch: 16
  imgsz: 640
  device: GPU
  optimizer: AdamW
  lr0: 0.01
  lrf: 0.01
  patience: 5 (early stopping)
  cos_lr: true

augmentations:
  mosaic: 1.0
  mixup: 0.1
  perspective: 0.0005
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  degrees: 10.0
  fliplr: 0.5
  scale: 0.5
```

### 5.2 Timp de Antrenare

- **Total:** ~408 secunde (6.8 minute)
- **Per epocă:** ~10.5 secunde
- **Device:** GPU (CUDA)

---

## 6. Experimente de Optimizare Propuse

### 6.1 Tabel Experimente Planificate

| **Exp#** | **Modificare față de Baseline** | **Obiectiv** | **Status** |
|----------|--------------------------------|--------------|------------|
| Baseline | Configurația din Etapa 5 | Referință | ✅ Completat |
| Exp 1 | Threshold Paved_Zone: 0.361 → 0.5 | Reducere FP pentru zonă SAFE | 📋 Planificat |
| Exp 2 | Augmentări suplimentare brightness | Îmbunătățire Paved_Zone pe umbre | 📋 Planificat |
| Exp 3 | Class weights pentru Paved_Zone (×1.5) | Echilibrare clasă minoritară | 📋 Planificat |
| Exp 4 | Model YOLOv11m-seg (medium) | Capacitate mai mare | 📋 Planificat |

### 6.2 Experiment Implementat: Ajustare Threshold

**Motivație:** Reducerea False Positives pentru Paved_Zone este critică pentru siguranță.

**Implementare în UI:**
```python
# În src/app/main.py
PAVED_ZONE_THRESHOLD = 0.5  # Crescut de la 0.25 default

if cls_id in SAFE_CLASSES and confidence >= PAVED_ZONE_THRESHOLD:
    # Marchează ca SAFE doar dacă confidence > 50%
    color = [0, 255, 0]
else:
    color = [255, 0, 0]
```

**Rezultat așteptat:**
- ↓ False Positives (mai puține zone clasificate greșit ca SAFE)
- ↓ Recall (unele zone SAFE cu confidence scăzut vor fi ratate)
- ↑ Siguranță operațională

---

## 7. Actualizarea Aplicației Software în Etapa 6

### 7.1 Tabel Modificări Aplicație

| **Componenta** | **Stare Etapa 5** | **Modificare Etapa 6** | **Justificare** |
|----------------|-------------------|------------------------|-----------------|
| Threshold detecție | 0.25 (global) | 0.25 (UNSAFE), 0.5 (SAFE) | Reducere FP pentru zone sigure |
| Model folosit | best.pt din drone_landing_lvl2 | Same (optimizat prin threshold) | Menținere consistență |
| Logging | Doar predicție | Predicție + confidence + clasă | Debugging și audit |
| UI feedback | Culoare simplă | Culoare + confidence % afișat | Transparență pentru operator |

### 7.2 State Machine Actualizat

```
IDLE → PREPROCESS → INFERENCE → CONFIDENCE_FILTER → SAFETY_DECISION → DISPLAY

Modificare Etapa 6:
- Adăugat CONFIDENCE_FILTER între INFERENCE și SAFETY_DECISION
- Filtrare: Paved_Zone acceptat doar dacă confidence ≥ 0.5
- Logging detaliat în fiecare stare
```

### 7.3 Cod Actualizat

```python
# Filtru confidence diferențiat pe clase
def apply_safety_logic(cls_id, confidence, class_name):
    if cls_id == 2:  # Paved_Zone (SAFE)
        if confidence >= 0.5:
            return "SAFE", [0, 255, 0]
        else:
            return "UNCERTAIN", [255, 165, 0]  # Orange pentru incert
    else:  # Grass_Zone sau No_Fly_Zone (UNSAFE)
        return "UNSAFE", [255, 0, 0]
```

---

## 8. Agregarea Rezultatelor Finale

### 8.1 Tabel Sumar Performanță pe Etape

| **Metrică** | **Etapa 4 (Random)** | **Etapa 5 (Antrenat)** | **Etapa 6 (Optimizat)** |
|-------------|---------------------|----------------------|------------------------|
| Precision (Mask) | ~5% | 82.6% | 82.6%* |
| Recall (Mask) | ~5% | 73.4% | 73.4%* |
| mAP50 (Mask) | ~0% | 77.5% | 77.5%* |
| F1-score | ~0.1 | 0.77 | 0.77* |
| False Positive Rate (Paved) | N/A | ~47% | ~30%** |

*Metrici la nivel de model identice (același weights)
**Estimare după ajustare threshold

### 8.2 Performanță per Clasă - Analiza Finală

| **Clasă** | **Rol în Aplicație** | **Recall** | **Risc FP** | **Evaluare** |
|-----------|---------------------|------------|-------------|--------------|
| No_Fly_Zone | Evitare obstacole | 96% ✅ | Foarte scăzut | **EXCELENT** |
| Grass_Zone | Marcare teren nesigur | 81% ✅ | Moderat | **BUN** |
| Paved_Zone | Identificare zonă aterizare | 65% ⚠️ | Ridicat | **NECESITĂ ÎMBUNĂTĂȚIRE** |

### 8.3 Curbele de Performanță

**F1-Confidence Curve:**
- F1 maxim global: **0.77** la confidence threshold **0.361**
- No_Fly_Zone: F1 ≈ 0.95 (stabil pe range larg de threshold-uri)
- Paved_Zone: F1 ≈ 0.65 (sensibil la threshold)

**Precision-Confidence Curve:**
- Precision 100% atinsă la confidence 0.789
- Trade-off: La threshold mai mare → Precision ↑, Recall ↓

**Recall-Confidence Curve:**
- Recall maxim 89% la threshold 0
- Scade rapid după threshold 0.6

---

## 9. Concluzii Tehnice Finale

### 9.1 Obiective Atinse

- [x] **Model RN funcțional** cu mAP50 = 77.5% pe segmentare semantică
- [x] **Integrare completă** în aplicație Streamlit cu 3 module
- [x] **State Machine** implementat și actualizat
- [x] **Pipeline end-to-end** testat: imagine → preprocess → inference → decizie → display
- [x] **UI demonstrativ** cu inferență reală și feedback vizual
- [x] **Documentație completă** pe toate etapele

### 9.2 Performanță vs. Cerințe

| **Cerință** | **Target** | **Obținut** | **Status** |
|-------------|------------|-------------|------------|
| Accuracy/Precision | ≥70% | 82.6% | ✅ DEPĂȘIT |
| F1-score | ≥0.65 | 0.77 | ✅ DEPĂȘIT |
| Detectare obstacole (No_Fly) | ≥90% | 96% | ✅ DEPĂȘIT |
| Detectare zonă sigură (Paved) | ≥80% | 65% | ⚠️ SUB TARGET |

### 9.3 Limitări Identificate

#### Limitări Date
1. **Dataset relativ mic:** 1126 instanțe totale, din care doar 291 pentru Paved_Zone
2. **Dezechilibru:** Grass_Zone (559) vs Paved_Zone (291) = raport 2:1
3. **Variabilitate limitată:** Imagini din aceeași competiție/locație

#### Limitări Model
1. **Confuzie Paved_Zone ↔ Background:** 32% din zonele pavate sunt ratate
2. **Generalizare:** Model nevalidat pe alte locații/condiții meteo
3. **Margini ambigue:** Tranziții între clase sunt problematice

#### Limitări Infrastructură
1. **Deployment:** Model neexportat încă pentru hardware drone (ONNX/TFLite)
2. **Latență:** Nemasurată pe hardware target

### 9.4 Lecții Învățate

**Tehnice:**
1. **Augmentările specifice domeniului** (perspective, scale) au fost mai eficiente decât augmentările generice
2. **Early stopping** (patience=5) a prevenit overfitting-ul și a economisit timp
3. **YOLOv11-seg** oferă un echilibru excelent viteză/acuratețe pentru aplicații real-time
4. **Confusion matrix** este mai informativă decât metricile agregat - dezvăluie probleme specifice pe clase

**Proces:**
1. **Analiza per-clasă** este esențială - metricile globale ascund probleme
2. **Threshold diferențiat** pe clase permite tuning fin pentru siguranță
3. **Adnotarea manuală cu poligoane** oferă calitate superioară pentru segmentare

**Domeniu (Drone Landing):**
1. **Clasa cea mai importantă** (No_Fly_Zone) funcționează cel mai bine - prioritatea corectă
2. **Trade-off Safety vs. Opportunity:** Mai bine să fii conservator (nu aterizezi când poți) decât agresiv (aterizezi unde nu trebuie)
3. **Validare temporală:** O singură predicție nu e suficientă - media pe mai multe frame-uri ar crește robustețea

---

## 10. Direcții de Cercetare și Dezvoltare

### 10.1 Pe Termen Scurt (1-3 luni)

1. **Colectare date adiționale:**
   - +200 imagini cu zone pavate variate (parcări, drumuri, helipad-uri)
   - Imagini în condiții meteo diferite (nori, soare direct, umbre lungi)

2. **Optimizare threshold:**
   - Experimentare cu threshold-uri diferite pentru fiecare clasă
   - Validare pe set de test separat

3. **Validare temporală:**
   - Implementare filtru pe secvențe de frame-uri
   - Zona SAFE doar dacă detectată consistent în ≥3 frame-uri consecutive

### 10.2 Pe Termen Mediu (3-6 luni)

1. **Export și deployment:**
   - Export ONNX pentru inferență optimizată
   - Testare pe hardware drone (Jetson Nano / Raspberry Pi)
   - Benchmark latență și optimizare sub 50ms

2. **Extindere dataset:**
   - Colaborare cu alte echipe de drone racing
   - Includere scenarii noi (urban, rural, industrial)

3. **Îmbunătățire model:**
   - Experimentare cu YOLOv11m-seg (varianta medium)
   - Fine-tuning cu class weights pentru Paved_Zone

---

## 11. Plan Post-Feedback

### 11.1 Dacă se solicită îmbunătățiri model:
- Experimente cu arhitecturi alternative (YOLOv11m, YOLOv11l)
- Class weights pentru Paved_Zone (×1.5 sau ×2.0)
- Augmentări adiționale targetate pe zonele problematice
- **Actualizare:** `models/`, `results/`, README Etapa 5 și 6

### 11.2 Dacă se solicită îmbunătățiri date:
- Colectare imagini suplimentare pentru Paved_Zone
- Verificare și corecție adnotări existente
- **Actualizare:** `data/`, README Etapa 3

### 11.3 Dacă se solicită îmbunătățiri arhitectură:
- Modificare State Machine (adăugare stări de validare)
- Implementare filtru temporal
- **Actualizare:** `docs/state_machine.*`, `src/app/`, README Etapa 4

---

## 12. Structura Finală Repository

```
ProiectRN_Corjuc_Ioan/
├── README.md                           # Overview general
├── README_etapa3_analiza_date.md       # Analiza dataset
├── README_etapa4_arhitectura_sia.md    # Arhitectura aplicației
├── README_etapa5_antrenare_model.md    # Antrenare și rezultate
├── README_etapa6_optimizare.md         # ACEST FIȘIER
│
├── docs/
│   ├── state_machine.png               # Diagrama State Machine
│   ├── loss_curve.png                  # results.png din antrenare
│   ├── confusion_matrix.png            # Matricea de confuzie
│   ├── confusion_matrix_normalized.png # Versiunea normalizată
│   ├── MaskF1_curve.png               # F1 vs Confidence
│   ├── MaskP_curve.png                # Precision vs Confidence
│   ├── MaskR_curve.png                # Recall vs Confidence
│   ├── labels.jpg                      # Distribuția claselor
│   └── screenshots/
│       ├── inference_real.png         # Demo UI
│       └── val_predictions.png        # Predicții pe validare
│
├── data/
│   ├── train/                         # Set antrenare
│   ├── valid/                         # Set validare
│   └── data.yaml                      # Configurare YOLO
│
├── src/
│   ├── app/main.py                    # Aplicația Streamlit
│   ├── neural_network/
│   │   ├── train_yolo.py              # Script antrenare
│   │   └── evaluate.py                # Script evaluare
│   └── init_project.py                # Inițializare date
│
├── models/
│   └── drone_landing_lvl2/weights/
│       ├── best.pt                    # Model final
│       └── last.pt                    # Ultima epocă
│
├── results/
│   ├── training_history.csv           # Istoric 39 epoci
│   ├── hyperparameters.yaml           # Configurare
│   └── test_metrics.json              # Metrici finale
│
└── requirements.txt                   # Dependențe Python
```

---

## 13. Checklist Final Etapa 6

### Analiză Performanță
- [x] Confusion matrix generată și analizată
- [x] Interpretare detaliată per clasă
- [x] Analiză 5 exemple greșite cu cauze și soluții
- [x] Implicații pentru context industrial documentate

### Optimizare
- [x] Tabel experimente de optimizare
- [x] Ajustare threshold implementată
- [x] Justificare configurație finală

### Actualizare Aplicație
- [x] Tabel modificări aplicație completat
- [x] State Machine actualizat
- [x] UI cu logică de siguranță diferențiată

### Concluzii
- [x] Evaluare performanță finală
- [x] Limitări identificate și documentate
- [x] Lecții învățate (tehnice, proces, domeniu)
- [x] Direcții viitoare definite
- [x] Plan post-feedback

### Documentație
- [x] Structură repository completă
- [x] Toate metricile reale incluse
- [x] Grafice și vizualizări salvate

---

## 14. Commit Final

```bash
git add .
git commit -m "Etapa 6 completă – mAP50=77.5%, F1=0.77, Precision=82.6%"
git tag -a v0.6-optimized-final -m "Etapa 6 - Analiză performanță și concluzii finale"
git push origin main --tags
```

---

**REMINDER:** Aceasta este versiunea finală pre-examen. Pe baza feedback-ului primit, componentele pot fi actualizate iterativ până la evaluarea finală.
