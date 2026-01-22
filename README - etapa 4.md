# 📘 README – Etapa 4: Arhitectura Completă a Aplicației SIA - Drone Safe Landing

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Corjuc Ioan Marian 
**Data:** [03.12.2025]
**Etapa:** 4 - Dezvoltarea arhitecturii aplicației software bazată pe RN

---

## 1. Scopul Proiectului și Soluția SIA

Acest proiect propune un **Sistem cu Inteligență Artificială (SIA)** pentru asistarea dronelor autonome în procesul de aterizare. Sistemul analizează imagini video în timp real pentru a identifica zonele sigure (fără obstacole, teren plan) și zonele periculoase.

### Tabelul: Nevoie Reală → Soluție Tehnică

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul meu** | **Modul software responsabil** |
|---------------------------|------------------------------|--------------------------------|
| Evitarea avarierii dronelor la aterizare | Detectare automată obstacole (copaci, steaguri) și validare zonă sigură (iarbă/platformă) | Modul RN (YOLO) + Logică Decizie |
| Aterizare autonomă în timp real | Procesare video cu latență minimă (<50ms) și feedback vizual imediat | Web Service (Streamlit) |

---

## 2. Contribuția Originală la Setul de Date (40%)

Pentru a asigura o performanță ridicată și specifică domeniului, am dezvoltat o metodologie proprie de generare a datelor, care asigură **100% conținut original** în dataset-ul final.

**Tipul contribuției:** [X] Date achiziționate cu senzori proprii & Etichetare manuală.
**Descriere:** Dataset-ul conține 100% imagini originale capturate în scenarii reale de zbor. Am adnotat manual peste 3000 de instanțe de poligoane pentru cele 5 clase, asigurând o calitate superioară dataset-urilor generice publice.

**Descriere detaliată:**
Acesta funcționează astfel:
1.  Am capturat imagini originale folosind o dronă și camera în scenarii reale de zbor (competitie de racing)
2.  Am adnotat manual mii de instanțe folosind **poligoane (Instance Segmentation)** pentru cele 5 clase critice: `Flags`, `Grass`, `Landing Zone`, `Sky`, `Tree`. Această abordare asigură o delimitare mult mai precisă a formelor neregulate față de bounding-box-urile clasice.
3.  Dezvoltarea scriptului `src/init_project.py` care scanează folderele brute (raw), consolidează imaginile și generează automat structura necesară antrenării (train/valid) și fișierul de configurare `data.yaml`.

**Locația datelor:** `data/raw/` (consolidate apoi în `data/train` și `data/valid`).
---

## 3. Justificarea Arhitecturii (State Machine)

Diagrama fluxului de date (disponibilă în `docs/state_machine.png`) urmărește un model de procesare "Single-Stage" optimizat pentru viteză.

**Stări Principale:**
1.  **IDLE:** Sistemul așteaptă input (imagine/video).
2.  **PREPROCESS:** Redimensionare imagine la 640x640px și normalizare pixelilor.
3.  **INFERENCE (YOLOv11):** Rulare model de segmentare semantică.
4.  **SAFETY LOGIC:** Filtrare rezultate. Doar clasa `Landing Zone` este considerată **SAFE**. Orice altceva (Copaci, Iarbă neclară, Cer) este marcat **UNSAFE**.
5.  **DISPLAY:** Afișare overlay colorat (Verde/Roșu) peste imaginea originală.

**Justificare:** Am ales arhitectura **YOLOv11-seg** deoarece oferă un echilibru ideal între viteză (necesară pentru decizii în timp real pe dronă) și capacitatea de a înțelege contextul global al imaginii (spre deosebire de clasificarea pe patch-uri mici).

---

## 4. Descrierea Modulelor Implementate

Proiectul este structurat modular pentru a respecta principiile de inginerie software ("Separation of Concerns").

**Modul**  **Locație**  **Descriere Funcțională (Etapa 4)** 

**1. Data Acquisition / Management**  `src/init_project.py` Script funcțional care scanează folderele brute, consolidează imaginile, generează automat split-ul de validare (20%) și creează fișierul de configurare `data.yaml` cu căi absolute. 
**2. Neural Network Module** `src/train_yolo.py`  Script care definește și încarcă arhitectura **YOLOv11n-seg**. În acest stadiu, modelul este instanțiat (`yolo11n-seg.pt`) dar neantrenat pe datele custom, fiind pregătit pentru Etapa 5. 
**3. Web Service / UI** `src/app_yolo.py`  Aplicație Streamlit funcțională care permite încărcarea unei imagini, rulează inferența (momentan cu modelul generic) și demonstrează pipeline-ul complet până la afișarea rezultatului vizual.    

---

## 5. Instrucțiuni de Instalare și Rulare

Pentru a testa arhitectura completă (End-to-End), urmați pașii de mai jos:

### Pasul 1: Instalare Dependențe
Asigurați-vă că aveți Python 3.8+ instalat.

`pip install -r requirements.txt`

### Pasul 2: Inițializare Date

`python src/init_project.py`

### Pasul 3: Testare UI (Demo)

`streamlit run src/app_yolo.py`