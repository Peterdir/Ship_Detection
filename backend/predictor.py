import os
import cv2
import numpy as np
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

    def split_image(self, image, tile_size=640, overlap_ratio=0.2):
        """
        Chia ảnh thành các tile với kích thước và tỷ lệ overlap cho trước.
        """
        h, w = image.shape[:2]
        stride = max(1, int(tile_size * (1 - overlap_ratio)))
        
        tiles = []
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y1 = y
                x1 = x
                y2 = y1 + tile_size
                x2 = x1 + tile_size
                
                # Điều chỉnh nếu vượt quá kích thước ảnh
                if y2 > h:
                    y1 = max(0, h - tile_size)
                    y2 = h
                if x2 > w:
                    x1 = max(0, w - tile_size)
                    x2 = w
                    
                tile = image[y1:y2, x1:x2]
                tiles.append({
                    'image': tile,
                    'x_offset': x1,
                    'y_offset': y1,
                    'orig_width': x2 - x1,
                    'orig_height': y2 - y1
                })
                
                if x2 == w:
                    break
            if y2 == h:
                break
                
        # Lọc các tile trùng lặp
        unique_tiles = []
        seen = set()
        for t in tiles:
            coords = (t['x_offset'], t['y_offset'], t['orig_width'], t['orig_height'])
            if coords not in seen:
                seen.add(coords)
                unique_tiles.append(t)
                
        return unique_tiles

    def is_edge_box(self, box, tile_width, tile_height, margin_ratio=0.03):
        """
        Kiểm tra box có nằm sát viền tile không.
        Box sát viền + confidence thấp → khả năng cao là tàu bị cắt → nên loại bỏ.
        
        Args:
            box: [x1, y1, x2, y2] tọa độ trong tile
            tile_width, tile_height: kích thước tile
            margin_ratio: tỷ lệ margin so với kích thước tile (mặc định 3%)
            
        Returns:
            True nếu box nằm sát viền tile
        """
        x1, y1, x2, y2 = box
        margin_x = tile_width * margin_ratio
        margin_y = tile_height * margin_ratio
        
        # Kiểm tra nếu bất kỳ cạnh nào của box nằm trong margin so với viền tile
        touches_left = x1 < margin_x
        touches_top = y1 < margin_y
        touches_right = x2 > (tile_width - margin_x)
        touches_bottom = y2 > (tile_height - margin_y)
        
        return touches_left or touches_top or touches_right or touches_bottom

    def detect_tile(self, tile_image, confidence_threshold=0.5, edge_filter=True):
        """
        Chạy inference trên một tile ảnh.
        Có tùy chọn lọc box sát viền (edge-aware filtering).
        
        Args:
            tile_image: ảnh tile (BGR)
            confidence_threshold: ngưỡng confidence tối thiểu
            edge_filter: nếu True, loại bỏ box sát viền có confidence thấp
        """
        # Chuyển BGR (OpenCV) sang RGB (PIL)
        img_rgb = cv2.cvtColor(tile_image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tensor = F.to_tensor(img_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            predictions = self.model(img_tensor)

        boxes = predictions[0]["boxes"].cpu().numpy()
        scores = predictions[0]["scores"].cpu().numpy()
        labels = predictions[0]["labels"].cpu().numpy()

        th, tw = tile_image.shape[:2]
        
        # Ngưỡng confidence cao hơn cho box sát viền
        edge_confidence_threshold = 0.80

        filtered_boxes = []
        filtered_scores = []
        for i in range(len(scores)):
            if scores[i] > confidence_threshold and labels[i] == 1:
                # Edge-aware filtering: box sát viền + confidence thấp → loại bỏ
                if edge_filter and self.is_edge_box(boxes[i], tw, th):
                    if scores[i] < edge_confidence_threshold:
                        # Box sát viền với confidence thấp → khả năng cao là tàu bị cắt
                        # Tàu này sẽ được phát hiện nguyên vẹn bởi tile/scale khác
                        continue
                filtered_boxes.append(boxes[i])
                filtered_scores.append(scores[i])

        return filtered_boxes, filtered_scores

    def merge_boxes(self, tile_results):
        """
        Gộp và chuyển bounding boxes về hệ tọa độ của ảnh gốc.
        """
        all_boxes = []
        all_scores = []
        for res in tile_results:
            x_off = res['x_offset']
            y_off = res['y_offset']
            scale_x = res.get('scale_x', 1.0)
            scale_y = res.get('scale_y', 1.0)
            
            for box, score in zip(res['boxes'], res['scores']):
                x1, y1, x2, y2 = box
                
                # Scale back if the tile was resized
                x1 /= scale_x
                x2 /= scale_x
                y1 /= scale_y
                y2 /= scale_y
                
                # Add offsets
                orig_x1 = x1 + x_off
                orig_y1 = y1 + y_off
                orig_x2 = x2 + x_off
                orig_y2 = y2 + y_off
                
                all_boxes.append([orig_x1, orig_y1, orig_x2, orig_y2])
                all_scores.append(score)
                
        return all_boxes, all_scores

    def global_nms(self, boxes, scores, confidence_threshold=0.5, nms_threshold=0.3, iom_threshold=0.4):
        """
        Thực hiện Non-Maximum Suppression (NMS) toàn cục sử dụng cả IoU, IoM 
        và bộ lọc Hallucination để khắc phục hiệu ứng phụ của Tiling.
        """
        if len(boxes) == 0:
            return [], []
            
        # Lọc các box dưới ngưỡng confidence và Phạt điểm theo Kích thước
        filtered_indices = []
        for i, (s, box) in enumerate(zip(scores, boxes)):
            if s < confidence_threshold:
                continue
                
            area = (box[2] - box[0]) * (box[3] - box[1])
            # Phạt điểm theo Kích thước (Area-based Confidence Penalty): 
            # Nếu Box khổng lồ (vượt quá 30000 pixel vuông), bắt buộc phải có độ tự tin > 0.85
            if area > 30000 and s < 0.85:
                continue
                
            filtered_indices.append(i)
            
        valid_boxes = [boxes[i] for i in filtered_indices]
        valid_scores = [scores[i] for i in filtered_indices]
        
        if len(valid_boxes) == 0:
            return [], []
            
        # --- BỘ LỌC HALLUCINATION (TILING ARTIFACTS) ---
        # Khi một góc của tàu lọt vào rìa của tile, model có thể nhận diện sai
        # và vẽ ra một box khổng lồ (phần lớn là nước/bọt sóng). 
        hallucination_indices = set()
        for i in range(len(valid_boxes)):
            for j in range(len(valid_boxes)):
                if i == j:
                    continue
                boxA = valid_boxes[i]
                boxB = valid_boxes[j]
                
                areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
                areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
                
                # Nếu box A cực lớn so với box B (gấp hơn 3 lần)
                if areaA > areaB * 3.0:
                    # Kỹ thuật Giãn nở Box (Bounding Box Expansion): 
                    # Phóng to Box B thêm 15% để bắt các trường hợp Box ảo giác nằm kề sát mà không giao nhau
                    wB = boxB[2] - boxB[0]
                    hB = boxB[3] - boxB[1]
                    dw = wB * 0.15
                    dh = hB * 0.15
                    
                    exp_boxB = [
                        boxB[0] - dw,
                        boxB[1] - dh,
                        boxB[2] + dw,
                        boxB[3] + dh
                    ]
                    
                    inter_x1 = max(boxA[0], exp_boxB[0])
                    inter_y1 = max(boxA[1], exp_boxB[1])
                    inter_x2 = min(boxA[2], exp_boxB[2])
                    inter_y2 = min(boxA[3], exp_boxB[3])
                    
                    inter_w = max(0, inter_x2 - inter_x1)
                    inter_h = max(0, inter_y2 - inter_y1)
                    inter_area = inter_w * inter_h
                    
                    # Nếu chúng giao nhau đáng kể (ít nhất 10% của box nhỏ)
                    if inter_area > 0.1 * areaB:
                        # Phân biệt dựa trên độ tự tin của Box nhỏ (B)
                        # TH1: Tàu nhỏ là tàu nguyên vẹn có điểm rất cao (> 0.85). 
                        # Box lớn (A) chắc chắn là ảo giác (hallucination bọt sóng) -> Xóa A.
                        if valid_scores[j] > 0.85:
                            hallucination_indices.add(i)
                        # TH2: Tàu lớn là thật. Box nhỏ (B) chỉ là mảnh vỡ (mũi tàu) do cắt tile nên điểm thấp.
                        # -> Xóa B.
                        else:
                            hallucination_indices.add(j)
                        
        # Lọc bỏ các box ảo giác
        new_boxes = []
        new_scores = []
        for i in range(len(valid_boxes)):
            if i not in hallucination_indices:
                new_boxes.append(valid_boxes[i])
                new_scores.append(valid_scores[i])
                
        valid_boxes = new_boxes
        valid_scores = new_scores
        
        if len(valid_boxes) == 0:
            return [], []
            
        # Sắp xếp các box theo điểm số giảm dần
        sorted_indices = sorted(range(len(valid_scores)), key=lambda i: valid_scores[i], reverse=True)
        
        keep_indices = []
        for i in sorted_indices:
            box1 = valid_boxes[i]
            x1_1, y1_1, x2_1, y2_1 = box1
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            
            keep = True
            for j in keep_indices:
                box2 = valid_boxes[j]
                x1_2, y1_2, x2_2, y2_2 = box2
                area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
                
                # Tính toán phần giao nhau (Intersection)
                inter_x1 = max(x1_1, x1_2)
                inter_y1 = max(y1_1, y1_2)
                inter_x2 = min(x2_1, x2_2)
                inter_y2 = min(y2_1, y2_2)
                
                inter_w = max(0, inter_x2 - inter_x1)
                inter_h = max(0, inter_y2 - inter_y1)
                inter_area = inter_w * inter_h
                
                # Intersection over Minimum (IoM) và Intersection over Union (IoU)
                min_area = min(area1, area2)
                
                if min_area > 0:
                    iom = inter_area / min_area
                    iou = inter_area / (area1 + area2 - inter_area)
                else:
                    iom = 0
                    iou = 0
                    
                # Loại bỏ box nếu phần đè lấn nhau lớn hơn ngưỡng (IoU hoặc IoM)
                if iom > iom_threshold or iou > nms_threshold:
                    keep = False
                    break
                    
            if keep:
                keep_indices.append(i)
                
        final_boxes = [valid_boxes[i] for i in keep_indices]
        final_scores = [valid_scores[i] for i in keep_indices]
                
        return final_boxes, final_scores

    def validate_box_content(self, image, bbox, area_threshold=10000, density_threshold=0.05):
        """
        Kiểm tra xem Box khổng lồ có bị rỗng (chứa toàn nước) hay không bằng thuật toán Otsu.
        """
        x1, y1, x2, y2 = map(int, bbox)
        area = (x2 - x1) * (y2 - y1)
        
        # Nếu box nhỏ, bỏ qua kiểm tra
        if area < area_threshold:
            return True
            
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Lấy vùng ROI
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return True
            
        # Chuyển sang ảnh xám
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Phân ngưỡng Otsu để tách vật thể khỏi nền
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Tính toán mật độ pixel trắng
        white_pixels = cv2.countNonZero(thresh)
        total_pixels = roi.shape[0] * roi.shape[1]
        
        density = white_pixels / float(total_pixels)
        
        if density < density_threshold:
            print(f"Hallucination detected, box is mostly water (Density: {density:.3f})")
            return False
            
        return True

    def normalize_image(self, image, max_size=2048, min_size=320):
        """
        Chuẩn hóa kích thước ảnh đầu vào:
        - Ảnh quá lớn (>max_size): scale xuống giữ nguyên tỷ lệ → giảm số tile, tăng tốc
        - Ảnh quá nhỏ (<min_size): scale lên giữ nguyên tỷ lệ → tàu to hơn, dễ detect
        """
        h, w = image.shape[:2]
        max_dim = max(h, w)
        scale = 1.0
        
        if max_dim > max_size:
            scale = max_size / max_dim
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"  Image normalized: {w}x{h} -> {new_w}x{new_h} (scale down {scale:.2f})")
        elif max_dim < min_size:
            scale = min_size / max_dim
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            print(f"  Image normalized: {w}x{h} -> {new_w}x{new_h} (scale up {scale:.2f})")
            
        return image, scale

    def _detect_at_scale(self, image, tile_size, overlap_ratio, confidence_threshold, resize_to=None):
        """
        Chạy detection ở một mức tile_size cụ thể.
        """
        tiles = self.split_image(image, tile_size=tile_size, overlap_ratio=overlap_ratio)
        tile_results = []
        
        for t in tiles:
            tile_img = t['image']
            scale_x, scale_y = 1.0, 1.0
            
            # Resize tile nếu được yêu cầu (phóng to tile nhỏ để model nhìn rõ hơn)
            if resize_to is not None:
                th, tw = tile_img.shape[:2]
                if tw > 0 and th > 0:
                    tile_img = cv2.resize(tile_img, resize_to)
                    scale_x = resize_to[0] / tw
                    scale_y = resize_to[1] / th
                
            boxes, scores = self.detect_tile(tile_img, confidence_threshold, edge_filter=True)
            
            tile_results.append({
                'boxes': boxes,
                'scores': scores,
                'x_offset': t['x_offset'],
                'y_offset': t['y_offset'],
                'orig_width': t['orig_width'],
                'orig_height': t['orig_height'],
                'scale_x': scale_x,
                'scale_y': scale_y
            })
            
        return tile_results

    def multi_scale_detect(self, image, confidence_threshold=0.5):
        """
        Pipeline detect đa tỷ lệ (Multi-Scale Tiling):
        - Chuẩn hóa kích thước ảnh đầu vào
        - Chạy detection ở nhiều tile size khác nhau
        - Tàu nhỏ sẽ được phát hiện ở tile size nhỏ (phóng to → dễ nhận)
        - Tàu bị cắt ở scale A sẽ nguyên vẹn ở scale B nhờ overlap + multi-scale
        """
        # Chuẩn hóa kích thước ảnh đầu vào
        orig_h, orig_w = image.shape[:2]
        image, norm_scale = self.normalize_image(image)
        h, w = image.shape[:2]
        all_tile_results = []
        
        # Xác định các scale configs dựa trên kích thước ảnh
        if w <= 640 and h <= 640:
            # Ảnh nhỏ/vừa (≤640x640)
            scale_configs = [
                # Scale 1: Full image — detect tàu lớn
                {'tile_size': max(w, h), 'overlap': 0.0, 'resize_to': None},
                # Scale 2: Tile 320x320 phóng to → 640x640 — detect tàu nhỏ
                {'tile_size': 320, 'overlap': 0.25, 'resize_to': (640, 640)},
                # Scale 3: Tile 224x224 phóng to → 640x640 — detect tàu rất nhỏ
                {'tile_size': 224, 'overlap': 0.30, 'resize_to': (640, 640)},
            ]
        else:
            # Ảnh lớn (>640x640)
            scale_configs = [
                # Scale 1: Tile 640x640 — detect tàu lớn
                {'tile_size': 640, 'overlap': 0.25, 'resize_to': None},
                # Scale 2: Tile 448x448 phóng to → 640x640 — detect tàu trung bình
                {'tile_size': 448, 'overlap': 0.25, 'resize_to': (640, 640)},
                # Scale 3: Tile 320x320 phóng to → 640x640 — detect tàu nhỏ
                {'tile_size': 320, 'overlap': 0.30, 'resize_to': (640, 640)},
            ]
        
        # Chạy detection ở từng scale
        for i, cfg in enumerate(scale_configs):
            ts = cfg['tile_size']
            # Nếu tile_size >= cả ảnh, detect toàn bộ ảnh (không cần tiling)
            if ts >= w and ts >= h:
                resize = cfg['resize_to']
                tile_img = image
                scale_x, scale_y = 1.0, 1.0
                
                if resize is not None:
                    tile_img = cv2.resize(image, resize)
                    scale_x = resize[0] / w
                    scale_y = resize[1] / h
                
                boxes, scores = self.detect_tile(tile_img, confidence_threshold, edge_filter=False)
                all_tile_results.append({
                    'boxes': boxes,
                    'scores': scores,
                    'x_offset': 0,
                    'y_offset': 0,
                    'orig_width': w,
                    'orig_height': h,
                    'scale_x': scale_x,
                    'scale_y': scale_y
                })
                print(f"  Scale {i+1}: Full image ({w}x{h}), found {len(boxes)} raw detections")
            else:
                results = self._detect_at_scale(
                    image, 
                    tile_size=ts, 
                    overlap_ratio=cfg['overlap'], 
                    confidence_threshold=confidence_threshold,
                    resize_to=cfg['resize_to']
                )
                total_dets = sum(len(r['boxes']) for r in results)
                print(f"  Scale {i+1}: Tile {ts}x{ts}, {len(results)} tiles, found {total_dets} raw detections")
                all_tile_results.extend(results)
        
        # Gộp boxes từ tất cả các scale về tọa độ ảnh gốc
        merged_boxes, merged_scores = self.merge_boxes(all_tile_results)
        print(f"  Total merged detections (before NMS): {len(merged_boxes)}")
        
        # Global NMS để loại bỏ trùng lặp giữa các scale
        nms_boxes, nms_scores = self.global_nms(
            merged_boxes, merged_scores, confidence_threshold, nms_threshold=0.3
        )
        print(f"  After NMS: {len(nms_boxes)} detections")
        
        # Content Validation — lọc box rỗng (chứa toàn nước)
        final_boxes = []
        final_scores = []
        for box, score in zip(nms_boxes, nms_scores):
            if self.validate_box_content(image, box):
                final_boxes.append(box)
                final_scores.append(score)
        
        # Scale boxes về tọa độ ảnh gốc (nếu ảnh đã bị normalize)
        if norm_scale != 1.0:
            scaled_boxes = []
            for box in final_boxes:
                x1, y1, x2, y2 = box
                scaled_boxes.append([
                    x1 / norm_scale,
                    y1 / norm_scale,
                    x2 / norm_scale,
                    y2 / norm_scale
                ])
            final_boxes = scaled_boxes
            print(f"  Boxes scaled back to original ({orig_w}x{orig_h}), factor: {1/norm_scale:.2f}")
        
        print(f"  Final detections: {len(final_boxes)} ships")
        return final_boxes, final_scores

    def detect_image(self, image, confidence_threshold=0.5):
        """
        Pipeline detect ảnh (wrapper cho multi_scale_detect).
        Giữ lại method cũ để tương thích ngược.
        """
        return self.multi_scale_detect(image, confidence_threshold)

    def predict(self, image_path, output_path, confidence_threshold=0.5):
        # Đọc ảnh bằng OpenCV
        img_cv2 = cv2.imread(image_path)
        
        # Chạy pipeline inference
        final_boxes, final_scores = self.detect_image(img_cv2, confidence_threshold)

        # Tổng hợp kết quả
        ship_count = len(final_boxes)
        has_ship = ship_count > 0
        average_confidence = float(sum(final_scores) / ship_count) if has_ship else 0.0

        ships = []
        for score, bbox in zip(final_scores, final_boxes):
            ships.append({
                "confidence": float(score),
                "bbox": [float(x) for x in bbox]
            })

        # Vẽ boxes lên ảnh kết quả
        for score, bbox in zip(final_scores, final_boxes):
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f"Ship: {score:.2f}"
            cv2.putText(img_cv2, label_text, (x1, max(y1 - 10, 0)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Lưu ảnh
        cv2.imwrite(output_path, img_cv2)

        return {
            "has_ship": has_ship,
            "ship_count": ship_count,
            "average_confidence": round(average_confidence, 4),
            "ships": ships
        }