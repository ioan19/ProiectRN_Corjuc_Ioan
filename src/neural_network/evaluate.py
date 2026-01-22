from ultralytics import YOLO
import json
import os
import shutil
import pandas as pd
from pathlib import Path

def evaluate():
    # --- CONFIGURARE ---
    # Caută automat cel mai nou model
    MODELS_DIR = Path('models')
    RESULTS_DIR = Path('results')
    DOCS_DIR = Path('docs')
    DATA_YAML = 'data/data.yaml'
    
    RESULTS_DIR.mkdir(exist_ok=True)
    DOCS_DIR.mkdir(exist_ok=True)

    # 1. Găsește modelul
    all_models = list(MODELS_DIR.rglob("best.pt"))
    if not all_models:
        print("❌ EROARE: Nu am găsit modelul antrenat!")
        return
    best_model_path = max(all_models, key=os.path.getmtime)
    print(f"🚀 Evaluăm modelul: {best_model_path}")
    
    # 2. Copiază training_history.csv (din folderul de antrenare în results/)
    # YOLO salvează un results.csv în folderul run-ului
    train_run_dir = best_model_path.parent.parent # models/drone_landing_lvl2
    yolo_csv = train_run_dir / "results.csv"
    
    if yolo_csv.exists():
        shutil.copy(yolo_csv, RESULTS_DIR / "training_history.csv")
        print("✅ Istoric antrenare copiat în results/training_history.csv")
        
        # Copiem și Loss Curve pentru documentație
        yolo_png = train_run_dir / "results.png"
        if yolo_png.exists():
            shutil.copy(yolo_png, DOCS_DIR / "loss_curve.png")
            print("✅ Loss Curve copiată în docs/loss_curve.png")
    else:
        print("⚠️ Nu am găsit results.csv original.")

    # 3. Rulează Evaluarea pe TEST SET
    print("📊 Rulăm inferența pe setul de TEST...")
    model = YOLO(best_model_path)
    metrics = model.val(
        data=DATA_YAML,
        split='test',       # Critic: Evaluăm pe TEST, nu VALID
        imgsz=640,
        batch=8,
        device=0,
        plots=True,
        project='results',
        name='eval_run'
    )

    # 4. Generează test_metrics.json
    final_metrics = {
        "test_metrics": {
            "mask": {
                "map_50": round(metrics.seg.map50, 4),
                "map_50_95": round(metrics.seg.map, 4),
                "precision": round(metrics.seg.mp, 4),
                "recall": round(metrics.seg.mr, 4),
                # Calculăm F1 manual
                "f1_score": round(2 * (metrics.seg.mp * metrics.seg.mr) / (metrics.seg.mp + metrics.seg.mr + 1e-16), 4)
            },
            "box": {
                "map_50": round(metrics.box.map50, 4),
                "map_50_95": round(metrics.box.map, 4)
            }
        },
        "model_used": str(best_model_path)
    }

    with open(RESULTS_DIR / 'test_metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=2)
    print("✅ Metrici finale salvate în results/test_metrics.json")

    # 5. Copiază Matricea de Confuzie
    eval_dir = RESULTS_DIR / 'eval_run'
    conf_matrix = list(eval_dir.glob("confusion_matrix*.png"))
    if conf_matrix:
        shutil.copy(conf_matrix[0], DOCS_DIR / "confusion_matrix.png")
        print("✅ Matricea de confuzie salvată în docs/")

    print("\n🏁 Proces de evaluare complet!")

if __name__ == "__main__":
    evaluate()