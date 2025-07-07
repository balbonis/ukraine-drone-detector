#TrainYoloV8_Raybird3

from ultralytics import YOLO
import torch

def main():
    # ✅ Check if CUDA is available
    if torch.cuda.is_available():
        device = 0  # use first GPU
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("⚠ CUDA not available — training on CPU")

    # ✅ Load YOLOv8s fast, medium accuracy, 4GB VRAM , Best balance for real-time RTX3050/3060
    model = YOLO('yolov8s.pt')

    # ✅ Train settings
    model.train(
        data='C:/Users/drizz/Downloads/Dada/MyFlaskProject/drone_ukraine.yaml',  # your dataset YAML
        # FOR 1K Images
        #epochs=75, #sweetspot between 50-100 for 1000 images
        #imgsz=640, #for small size
        #batch=16,  #default
        #patience=20, #stopsif it reaches 20 instances w/o improvement
        #device=device, #GPU/CUDA 
        #augment=True,
        
        # FOR PROTOTYPING
        epochs=50, #for fast processing 5-10 epochs only; max 50 to brute force
        imgsz=640, #for small size
        batch=1,  #default
        patience=20, #stopsif it reaches 20 instances w/o improvement
        device=device, #GPU/CUDA 
        augment=True,

        project='ukraine_drone_detection',
        name='ukraine_drone_yolov8_model',
        exist_ok=True  # overwrite existing run
    )

    # ✅ Validate after training
    metrics = model.val()
    print("Validation metrics:", metrics)

if __name__ == "__main__":
    main()