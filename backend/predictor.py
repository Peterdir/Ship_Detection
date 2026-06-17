import os
import cv2
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from torchvision.models.detection import fasterrcnn_resnet50_fpn
# pyrefly: ignore [missing-import]
from torchvision.transforms import functional as F
from PIL import Image

class ShipPredictor:
    def __init__(self, model_path="model/faster_rcnn_best.pth", device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = self._load_model()
        
    def _load_model(self):
        # The user requested num_classes=2 (0: background, 1: ship)
        model = fasterrcnn_resnet50_fpn(num_classes=2)
        if os.path.exists(self.model_path):
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            print(f"Model loaded from {self.model_path}")
        else:
            print(f"Warning: Model file not found at {self.model_path}. Please place the model file there.")
        
        model.to(self.device)
        model.eval()
        return model

    def predict(self, image_path, output_path, confidence_threshold=0.5):
        # Read image using PIL and convert to tensor
        img_pil = Image.open(image_path).convert("RGB")
        img_tensor = F.to_tensor(img_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            predictions = self.model(img_tensor)

        # Get results
        boxes = predictions[0]["boxes"].cpu().numpy()
        scores = predictions[0]["scores"].cpu().numpy()
        labels = predictions[0]["labels"].cpu().numpy()

        # Filter out based on confidence threshold and label == 1 (ship)
        filtered_indices = [
            i for i, (score, label) in enumerate(zip(scores, labels))
            if score > confidence_threshold and label == 1
        ]

        final_boxes = boxes[filtered_indices]
        final_scores = scores[filtered_indices]
        
        ship_count = len(final_boxes)
        has_ship = ship_count > 0
        average_confidence = float(final_scores.mean()) if has_ship else 0.0

        ships = []
        for score, bbox in zip(final_scores, final_boxes):
            ships.append({
                "confidence": float(score),
                "bbox": [float(x) for x in bbox]
            })

        # Draw bounding boxes
        img_cv2 = cv2.imread(image_path)
        
        for score, bbox in zip(final_scores, final_boxes):
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f"Ship: {score:.2f}"
            cv2.putText(img_cv2, label_text, (x1, max(y1 - 10, 0)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save result image
        cv2.imwrite(output_path, img_cv2)

        return {
            "has_ship": has_ship,
            "ship_count": ship_count,
            "average_confidence": round(average_confidence, 4),
            "ships": ships
        }
