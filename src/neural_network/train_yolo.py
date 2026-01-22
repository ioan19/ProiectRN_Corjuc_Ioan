from ultralytics import YOLO
import os
import torch
import yaml
from pathlib import Path

def train():
    # --- CONFIGURARE ---
    EPOCHS = 50
    BATCH = 16
    IMGSZ = 640
    NAME = 'drone_landing_lvl2'
    PROJECT = 'models'
    
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f" Device: {device}")

    model = YOLO('yolo26s-seg.pt') 

    results = model.train(
        data=os.path.abspath('data/data.yaml'), 
        epochs=EPOCHS, 
        imgsz=IMGSZ, 
        batch=BATCH,
        device=device,
        name=NAME, 
        project=PROJECT,
        
        # Nivel 2 Params
        patience=5,  
        cos_lr=True,  
        lr0=0.01,
        lrf=0.01,
        optimizer='AdamW',

        # Augmentări Nivel 2
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        perspective=0.0005, mosaic=1.0, mixup=0.1, degrees=10.0, fliplr=0.5, scale=0.5,
        
        plots=True
    )
    
    # Salvăm hyperparameters.yaml AICI (pentru că aici știm config-ul de antrenare)
    # Dar metricile le lăsăm pentru evaluate.py
    hyperparams = {
        "training": {
            "epochs": EPOCHS, "batch": BATCH, "imgsz": IMGSZ, "device": device
        },
        "augmentations": {
            "mosaic": 1.0, "mixup": 0.1, "perspective": 0.0005, "hsv_v": 0.4
        }
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/hyperparameters.yaml', 'w') as f:
        yaml.dump(hyperparams, f, sort_keys=False)

    print(f"✅ Antrenare Completă! Model salvat în models/{NAME}")

if __name__ == '__main__':
    train()