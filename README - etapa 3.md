# 📘 README – Etapa 3: Analiza și Pregătirea Setului de Date pentru Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Corjuc Ioan Marian  
**Data:** [20.11.2025] 

---

## Introducere

Acest document descrie activitățile realizate în **Etapa 3**, în care se analizează și se preprocesează setul de date necesar proiectului „Rețele Neuronale". Scopul etapei este pregătirea corectă a datelor pentru instruirea modelului RN, respectând bunele practici privind calitatea, consistența și reproductibilitatea datelor.

---

##  1. Structura Repository-ului Github (versiunea Etapei 3)

```
ProiectRN_Corjuc_Ioan/ 
├── README.md
├── ReadMe - etapa 3
├── docs/
├── data/
│   ├── raw/               # date brute
│   ├── processed/         # date curățate și transformate
│   ├── train/             # set de instruire
│   ├── valid/             # set de validare
│   └── test/              # set de testare
├── src/
│   ├── preprocessing/     # funcții pentru preprocesare
│   ├── data_acquisition/  # generare / achiziție date (dacă există)
│   └── neural_network/    # implementarea RN (în etapa următoare)
├── config/                # fișiere de configurare
└── requirements.txt       # dependențe Python (dacă aplicabil)
```
---

##  2. Descrierea Setului de Date

### 2.1 Sursa datelor

* **Origine:** Contribuție proprie 100%. Imaginile au fost capturate folosind camera unei drone de competitie in timpul concursului Student AirRace 2025 în condiții reale de exterior.
* **Modul de achiziție:** ☑ Senzori reali (Captură foto/video)
* **Platformă adnotare**: Roboflow
* **Perioada / condițiile colectării:** Octombrie 2025

### 2.2 Caracteristicile dataset-ului

* **Număr total de observații:** 100 de imagini
* **Număr de caracteristici (features):** 5
* **Tipuri de date:** ☑  Imagini
* **Format fișiere:** ☑ JPG , ☑ TXT (coordonate normalizate ale poligoanelor)

### 2.3 Descrierea fiecărei caracteristici

### 2.3 Descrierea Claselor (Features)

| **ID Clasă** | **Nume Clasă** | **Tip Zonă** | **Descriere** | **Acțiune Dronă** |
|:---:|:---|:---|:---|:---|
| **0** | **Flags** | Unsafe (Obstacol) | Obiecte dinamice, steaguri, markere verticale | Ocolire / Abort |
| **1** | **Grass** | Unsafe | Iarbă înaltă, teren denivelat sau umed | Ocolire |
| **2** | **Landing Zone** | **SAFE** | Suprafață plană, curată (ex: pad, asfalt) | **ATERIZARE PERMISĂ** |
| **3** | **Sky** | N/A | Linia orizontului, cer, spațiu gol | Navigație / Ignorare |
| **4** | **Tree** | Unsafe (Obstacol) | Vegetație înaltă, coroane copaci, crengi | Ocolire |

**Fișier recomandat:**  `data/README.md`

---

##  3. Analiza Exploratorie a Datelor (EDA) – Sintetic

### 3.1 Statistici descriptive aplicate

* **Distribuția instanțelor per clasă**
Analiza dezechilibrului (Class Imbalance): De exemplu, clasa Sky și Grass apar în majoritatea imaginilor, în timp ce Landing Zone apare doar în cadrele specifice apropierii de sol.
* **Dimensiunea obiectelor (Area Coverage)**
S-a analizat aria relativă ocupată de fiecare clasă. Tree și Sky ocupă porțiuni mari, în timp ce Landing Zone poate fi mică la altitudini mari.


### 3.2 Analiza calității datelor

* **Verificarea adnotărilor** 
Inspectare vizuală în Roboflow pentru a asigura că poligoanele urmăresc conturul exact (tight fit) pentru a maximiza precizia segmentării.
* **Variația condițiilor:**
Dataset-ul conține imagini cu lumină variabilă (soare, nori) pentru a asigura generalizarea modelului

### 3.3 Probleme identificate

* [Ocluzii] Unele zone de aterizare sunt parțial acoperite de umbrele copacilor
* [Similitudini_vizuale] Confuzie posibilă între Grass (unsafe) și anumite texturi de Tree văzute de sus. S-a rezolvat prin adnotare granulară

---

##  4. Preprocesarea Datelor

### 4.1 Curățarea datelor

* **Curățarea și Transformarea**
  * Auto-Orientation : Eliminarea meta-datelor EXIF de rotație.
  * Resizing : Toate imaginile sunt redimensionate la 640x640 pixeli (standard YOLOv8) folosind interpolare biliniară. Aceasta permite antrenarea eficientă pe GPU.

### 4.2 Transformarea caracteristicilor

* **Normalizare** : Valorile pixelilor [0, 255] sunt convertite în floating point [0.0, 1.0].
* **Encoding pentru variabile categoriale**
* **Ajustarea dezechilibrului de clasă**

## 4.3 Augmentarea Datelor (Data Augmentation)
Pentru a mări artificial dataset-ul și a preveni overfitting-ul (dat fiind numărul limitat de imagini originale), s-au aplicat transformări în timpul generării dataset-ului:

* **Flip**: Orizontal (simulare abordare din direcții opuse).
* **Crop**: Random crop (0-20%) pentru a simula drona fiind mai aproape de subiect.
* **Rotation**: ±15 grade (simulare instabilitate dronă la vânt).
* **Mosaic**: O tehnică specifică YOLO care combină 4 imagini într-una singură, obligând modelul să detecteze obiecte la scări diferite și în contexte complexe.

### 4.4 Structurarea seturilor de date

**Împărțire recomandată:**
Setul de date a fost împărțit respectând principiul stratificării:
* **70% – Train:** 70 imagini
* **20% – Validation:** 20 imagini
* **10% – Test:** 10 imagini

**Principii respectate:**
* Stratificare pentru clasificare
* Fără scurgere de informație (data leakage)
* Statistici calculate DOAR pe train și aplicate pe celelalte seturi

### 4.4 Salvarea rezultatelor preprocesării

* Date preprocesate în `data/processed/`
* Seturi train/val/test în foldere dedicate
* Parametrii de preprocesare în `config/preprocessing_config.*` (opțional)

---

##  5. Fișiere Generate în Această Etapă

* `data/raw/` – date brute
* `data/processed/` – date curățate & transformate
* `data/train/`, `data/validation/`, `data/test/` – seturi finale
* `src/preprocessing/` – codul de preprocesare
* `data/README.md` – descrierea dataset-ului

---

##  6. Stare Etapă (de completat de student)

- [ ] Structură repository configurată
- [ ] Dataset analizat (EDA realizată)
- [ ] Date preprocesate
- [ ] Seturi train/val/test generate
- [ ] Documentație actualizată în README + `data/README.md`

---
