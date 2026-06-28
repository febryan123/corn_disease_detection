import os
import numpy as np
from PIL import Image
import io

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    from torchvision.models import densenet121
except ImportError:
    torch = None

# Fallback default kelas (akan ditimpa otomatis kalau file checkpoint
# punya metadata class_names di dalamnya)
CLASSES = [
    "Bercak_Daun",
    "Hawar_Daun",
    "Karat_Daun",
    "Sehat"
]

IMAGE_SIZE      = 224
IMAGENET_MEAN   = [0.485, 0.456, 0.406]
IMAGENET_STD    = [0.229, 0.224, 0.225]
DROPOUT_RATE    = 0.5

DEVICE = torch.device("cuda" if torch and torch.cuda.is_available() else "cpu") if torch else None

# Transform HARUS sama dengan eval_tf di notebook training
# (bukan resize langsung ke 224x224 + bagi 255 seperti versi Keras lama)
inference_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
]) if torch else None


def build_densenet121(num_classes, dropout=0.5):
    """Arsitektur sama persis dengan saat training (lihat Cell 11 notebook)."""
    model = densenet121(weights=None)  # bobot di-load manual, bukan dari ImageNet
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(p=dropout * 0.5),
        nn.Linear(512, num_classes),
    )
    return model


def load_model(weights_path="model_final.pt"):
    """
    Load model PyTorch hasil training DenseNet121.
    Mendukung dua jenis file dari Cell 17 notebook:
      - '..._full_checkpoint.pt'  (berisi metadata + class_names + state_dict)
      - '..._weights.pth'         (hanya state_dict)
    """
    if torch is None:
        print("PyTorch tidak terinstal. Install dengan: pip install torch torchvision")
        return None

    if not os.path.exists(weights_path):
        print(f"Warning: {weights_path} tidak ditemukan. Letakkan file model di sini.")
        return None

    state = torch.load(weights_path, map_location=DEVICE, weights_only=False)

    if isinstance(state, dict) and "model_state_dict" in state:
        class_names = state.get("class_names", CLASSES)
        model_state = state["model_state_dict"]
    else:
        class_names = CLASSES
        model_state = state

    model = build_densenet121(num_classes=len(class_names), dropout=DROPOUT_RATE)
    model.load_state_dict(model_state)
    model.to(DEVICE)
    model.eval()

    print(f"Loaded model from {weights_path}  ({len(class_names)} kelas: {class_names})")
    return {"model": model, "class_names": class_names}


def predict_image(image_bytes, model_bundle):
    """
    Proses gambar dan kembalikan kelas prediksi beserta confidence-nya.
    """
    if model_bundle is None or model_bundle.get("model") is None:
        return {
            "success": False,
            "error": "Model tidak termuat. Pastikan file model (.pt/.pth) ada."
        }

    model       = model_bundle["model"]
    class_names = model_bundle["class_names"]

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        x = inference_transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(x)
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

        predicted_idx     = int(np.argmax(probabilities))
        predicted_class   = class_names[predicted_idx]
        confidence_score  = float(probabilities[predicted_idx]) * 100

        return {
            "success": True,
            "prediction": predicted_class,
            "confidence": f"{confidence_score:.2f}%",
            "class_index": predicted_idx,
            "all_probabilities": {
                class_names[i]: f"{float(probabilities[i])*100:.2f}%"
                for i in range(len(class_names))
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }