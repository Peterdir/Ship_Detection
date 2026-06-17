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
        # Cấu hình num_classes=2 (0: background, 1: ship)
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
        # Đọc ảnh bằng PIL và chuyển sang tensor
        img_pil = Image.open(image_path).convert("RGB")
        img_tensor = F.to_tensor(img_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            predictions = self.model(img_tensor)

        # Lấy kết quả thô từ model
        boxes = predictions[0]["boxes"].cpu().numpy()
        scores = predictions[0]["scores"].cpu().numpy()
        labels = predictions[0]["labels"].cpu().numpy()

        # BƯỚC 1: Lọc thô dựa trên điểm tự tin và NHÃN SỐ 1
        keep_indices = []
        for i in range(len(scores)):
            if scores[i] > confidence_threshold and labels[i] == 1:
                keep_indices.append(i)

        filtered_boxes = boxes[keep_indices]
        filtered_scores = scores[keep_indices]

        final_boxes = []
        final_scores = []

        # BƯỚC 2: Lọc tinh bằng NMS (Xóa các box trùng đè lên nhau)
        if len(filtered_boxes) > 0:
            nms_boxes = []
            for box in filtered_boxes:
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                nms_boxes.append([float(x1), float(y1), float(w), float(h)])

            # nms_threshold=0.3: Ép các box đè nhau > 30% phải gộp làm 1
            indices_after_nms = cv2.dnn.NMSBoxes(
                bboxes=nms_boxes, 
                scores=[float(s) for s in filtered_scores], 
                score_threshold=confidence_threshold, 
                nms_threshold=0.3 
            )

            if len(indices_after_nms) > 0:
                for idx in indices_after_nms.flatten():
                    final_boxes.append(filtered_boxes[idx])
                    final_scores.append(filtered_scores[idx])

        # Tổng hợp kết quả trả về
        ship_count = len(final_boxes)
        has_ship = ship_count > 0
        average_confidence = float(sum(final_scores) / ship_count) if has_ship else 0.0

        ships = []
        for score, bbox in zip(final_scores, final_boxes):
            ships.append({
                "confidence": float(score),
                "bbox": [float(x) for x in bbox]
            })

        # Vẽ box lên ảnh bằng OpenCV
        img_cv2 = cv2.imread(image_path)
        for score, bbox in zip(final_scores, final_boxes):
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f"Ship: {score:.2f}"
            cv2.putText(img_cv2, label_text, (x1, max(y1 - 10, 0)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Lưu ảnh kết quả
        cv2.imwrite(output_path, img_cv2)

        return {
            "has_ship": has_ship,
            "ship_count": ship_count,
            "average_confidence": round(average_confidence, 4),
            "ships": ships
        }