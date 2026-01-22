import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
from pathlib import Path

# --- CONFIGURARE ---
st.set_page_config(page_title="Drone Precision Landing", layout="wide", page_icon="🚁")

# AICI DEFINIM CE ESTE SIGUR PE NOUL DATASET (3 Clase):
# 0: Grass_Zone (Acum UNSAFE)
# 1: No_Fly_Zone (UNSAFE)
# 2: Paved_Zone (SAFE - Doar Asfalt/Beton)
SAFE_CLASSES = [2] 

st.title("🚁 Sistem de Aterizare (Strict Mode)")
st.markdown("""
**Legendă Detecție:**
* 🟩 **SAFE:** Zonă Pavată (Paved_Zone)
* 🟥 **UNSAFE:** Iarbă (Grass) & Zone Interzise (No_Fly_Zone)
""")

# --- FUNCȚIE DE CĂUTARE AUTOMATĂ A CELUI MAI NOU MODEL ---
def find_latest_model():
    models_dir = Path("models")
    if not models_dir.exists():
        return None
    
    # Caută recursiv best.pt
    all_models = list(models_dir.rglob("best.pt"))
    if not all_models:
        return None
    
    # Îl returnează pe cel mai recent modificat
    latest_model = max(all_models, key=os.path.getmtime)
    return str(latest_model)

@st.cache_resource
def load_model():
    best_model_path = find_latest_model()
    
    if best_model_path:
        print(f"✅ Model încărcat: {best_model_path}")
        return YOLO(best_model_path)
    else:
        st.error("⚠️ Nu s-a găsit model antrenat! Rulează 'python src/train_yolo.py'")
        # Fallback
        return YOLO("yolo11n-seg.pt")

# Încărcare model
model = load_model()

# Afișare status model în sidebar
if model:
    st.sidebar.success(f"Model Activ: YOLOv11 Custom (3 Clase)")
    if hasattr(model, 'names'):
        st.sidebar.info(f"Clase Detectate: {model.names}")

def process_image(image_pil, conf_threshold):
    img_np = np.array(image_pil)
    
    # Inferență YOLO
    results = model(img_np, conf=conf_threshold)
    result = results[0]
    
    # Copie pentru desenat (Overlay)
    overlay = img_np.copy()
    
    # Dacă avem detecții
    if result.masks is not None:
        for mask, box in zip(result.masks.data, result.boxes):
            cls_id = int(box.cls)
            # Verificăm numele clasei
            if hasattr(model, 'names'):
                class_name = model.names[cls_id]
            else:
                class_name = str(cls_id)
            
            # Scalare mască la rezoluția imaginii
            mask_np = mask.cpu().numpy()
            mask_resized = cv2.resize(mask_np, (img_np.shape[1], img_np.shape[0]))
            binary_mask = mask_resized > 0.5
            
            # --- LOGICA DE COLORARE ---
            if cls_id in SAFE_CLASSES:
                # VERDE - Safe (Doar Paved)
                color = [0, 255, 0] 
                label_text = f"✅ {class_name} (SAFE)"
                
                # Chenar Verde
                x1, y1, x2, y2 = box.xyxy[0]
                cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
            else:
                # ROȘU - Unsafe (Grass + No_Fly)
                color = [255, 0, 0]
                label_text = f"⛔ {class_name} (UNSAFE)"
            
            # Aplicăm culoarea pe mască (overlay)
            overlay[binary_mask] = color
            
            # Punem eticheta text
            x1, y1, x2, y2 = box.xyxy[0]
            text_y = int(y1) - 10 if int(y1) - 10 > 10 else int(y1) + 20
            
            cv2.putText(overlay, label_text, (int(x1), text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Amestecăm imaginile (Transparență 40% pentru overlay)
    final_img = cv2.addWeighted(img_np, 0.6, overlay, 0.4, 0)
    return final_img

# --- INTERFAȚA ---
col_conf, col_upload = st.columns([1, 2])
with col_conf:
    st.subheader("Configurare")
    conf = st.slider("Sensibilitate Detecție", 0.1, 1.0, 0.25)
    
    st.write("---")
    st.write("**Legendă Siguranță:**")
    st.success("✅ SAFE: Paved_Zone (Beton/Asfalt)")
    st.error("⛔ UNSAFE: Grass_Zone (Iarbă) + No_Fly_Zone")

with col_upload:
    uploaded = st.file_uploader("Încarcă imagine dronă", type=["jpg", "png", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    
    c1, c2 = st.columns(2)
    c1.image(image, caption="Original", use_container_width=True)
    
    if st.button("Analizează Teren"):
        with st.spinner("Procesare imagine..."):
            res = process_image(image, conf)
            c2.image(res, caption="Rezultat Analiză", use_container_width=True)