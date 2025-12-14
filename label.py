label

import cv2
import numpy as np
from ultralytics import SAM
import os
from tkinter import simpledialog, filedialog
import tkinter as tk
import shutil
import random

class DatasetSplitter:
    """資料集分割工具，將資料按比例分配到train和val"""
    
    def __init__(self, train_ratio=0.8):
        self.train_ratio = train_ratio
        self.val_ratio = 1.0 - train_ratio
        
    def select_source_folder(self):
        """選擇源資料夾"""
        root = tk.Tk()
        root.withdraw()
        
        folder_path = filedialog.askdirectory(
            title="選擇包含圖片的資料夾"
        )
        
        root.destroy()
        return folder_path
    
    def select_output_folder(self):
        """選擇輸出資料夾"""
        root = tk.Tk()
        root.withdraw()
        
        folder_path = filedialog.askdirectory(
            title="選擇輸出資料夾（將創建train/val結構）"
        )
        
        root.destroy()
        return folder_path
    
    def get_image_label_pairs(self, source_folder):
        """獲取圖片和標註文件配對"""
        images_folder = source_folder
        labels_folder = os.path.join(source_folder, "labels")
        
        # 檢查labels資料夾是否存在
        if not os.path.exists(labels_folder):
            print(f"❌ 找不到labels資料夾: {labels_folder}")
            return []
        
        # 獲取所有圖片文件
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [f for f in os.listdir(images_folder) 
                      if f.lower().endswith(image_extensions) and 
                      os.path.isfile(os.path.join(images_folder, f))]
        
        # 配對圖片和標註
        pairs = []
        for img_file in image_files:
            base_name = os.path.splitext(img_file)[0]
            label_file = f"{base_name}.txt"
            label_path = os.path.join(labels_folder, label_file)
            
            # 只添加有對應標註的圖片
            if os.path.exists(label_path):
                pairs.append({
                    'image': img_file,
                    'label': label_file,
                    'image_path': os.path.join(images_folder, img_file),
                    'label_path': label_path
                })
            else:
                print(f"⚠️  跳過（無標註）: {img_file}")
        
        return pairs
    
    def create_output_structure(self, output_folder):
        """創建輸出資料夾結構"""
        # 創建資料夾
        folders = [
            os.path.join(output_folder, "images", "train"),
            os.path.join(output_folder, "images", "val"),
            os.path.join(output_folder, "labels", "train"),
            os.path.join(output_folder, "labels", "val")
        ]
        
        for folder in folders:
            os.makedirs(folder, exist_ok=True)
            print(f"✓ 創建資料夾: {folder}")
        
        return folders
    
    def split_and_copy(self, source_folder, output_folder, train_ratio=None):
        """分割並複製資料"""
        if train_ratio is not None:
            self.train_ratio = train_ratio
            self.val_ratio = 1.0 - train_ratio
        
        print(f"\n{'='*60}")
        print(f"資料集分割工具")
        print(f"{'='*60}")
        print(f"源資料夾: {source_folder}")
        print(f"輸出資料夾: {output_folder}")
        print(f"分割比例: Train {self.train_ratio*100:.0f}% / Val {self.val_ratio*100:.0f}%")
        print(f"{'='*60}\n")
        
        # 獲取圖片標註配對
        pairs = self.get_image_label_pairs(source_folder)
        
        if not pairs:
            print("❌ 未找到有效的圖片-標註配對")
            return
        
        print(f"✓ 找到 {len(pairs)} 組有效配對\n")
        
        # 隨機打亂
        random.shuffle(pairs)
        
        # 計算分割點
        train_count = int(len(pairs) * self.train_ratio)
        train_pairs = pairs[:train_count]
        val_pairs = pairs[train_count:]
        
        print(f"分配結果:")
        print(f"  • Train: {len(train_pairs)} 張")
        print(f"  • Val: {len(val_pairs)} 張\n")
        
        # 創建輸出結構
        self.create_output_structure(output_folder)
        
        # 複製train資料
        print("複製 Train 資料...")
        for i, pair in enumerate(train_pairs, 1):
            # 複製圖片
            dst_img = os.path.join(output_folder, "images", "train", pair['image'])
            shutil.copy2(pair['image_path'], dst_img)
            
            # 複製標註
            dst_label = os.path.join(output_folder, "labels", "train", pair['label'])
            shutil.copy2(pair['label_path'], dst_label)
            
            if i % 10 == 0 or i == len(train_pairs):
                print(f"  進度: {i}/{len(train_pairs)}")
        
        # 複製val資料
        print("\n複製 Val 資料...")
        for i, pair in enumerate(val_pairs, 1):
            # 複製圖片
            dst_img = os.path.join(output_folder, "images", "val", pair['image'])
            shutil.copy2(pair['image_path'], dst_img)
            
            # 複製標註
            dst_label = os.path.join(output_folder, "labels", "val", pair['label'])
            shutil.copy2(pair['label_path'], dst_label)
            
            if i % 10 == 0 or i == len(val_pairs):
                print(f"  進度: {i}/{len(val_pairs)}")
        
        print(f"\n{'='*60}")
        print("✓ 資料分割完成！")
        print(f"{'='*60}")
        print(f"輸出位置: {output_folder}")
        print(f"  • images/train: {len(train_pairs)} 張")
        print(f"  • images/val: {len(val_pairs)} 張")
        print(f"  • labels/train: {len(train_pairs)} 個")
        print(f"  • labels/val: {len(val_pairs)} 個")
        print(f"{'='*60}\n")
    
    def interactive_split(self):
        """互動式分割流程"""
        print("\n" + "="*60)
        print("資料集分割工具")
        print("="*60)
        
        # 選擇源資料夾
        print("\n步驟 1: 選擇包含圖片和labels資料夾的源資料夾")
        source_folder = self.select_source_folder()
        
        if not source_folder:
            print("❌ 未選擇源資料夾，取消操作")
            return
        
        # 選擇輸出資料夾
        print("\n步驟 2: 選擇輸出資料夾")
        output_folder = self.select_output_folder()
        
        if not output_folder:
            print("❌ 未選擇輸出資料夾，取消操作")
            return
        
        # 詢問是否使用自訂比例
        root = tk.Tk()
        root.withdraw()
        
        custom_ratio = simpledialog.askstring(
            "分割比例",
            "請輸入Train比例 (0-100)，預設80:",
            initialvalue="80",
            parent=root
        )
        
        root.destroy()
        
        if custom_ratio:
            try:
                train_ratio = float(custom_ratio) / 100
                train_ratio = max(0.1, min(0.9, train_ratio))  # 限制在10%-90%
            except:
                train_ratio = 0.8
        else:
            train_ratio = 0.8
        
        # 執行分割
        self.split_and_copy(source_folder, output_folder, train_ratio)


class YOLOLabelTool:
    def __init__(self, model_path="sam2_b.pt", display_size=(1280, 720)):
        """初始化SAM2模型和標註工具"""
        self.model = SAM(model_path)
        self.classes = []
        self.current_image = None
        self.current_image_path = None
        self.display_image = None
        self.display_size = display_size
        self.point = None
        
        # 原始圖片尺寸
        self.original_height = 0
        self.original_width = 0
        
        # 顯示圖片尺寸和縮放參數
        self.display_height = 0
        self.display_width = 0
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        # 類別選擇相關
        self.selected_class = None
        self.class_window_active = False
        
        # 當前標註
        self.current_labels = []
        
        # Bounding box 調整相關
        self.adjusting_bbox = False
        self.bbox_edge = None
        self.temp_bbox = None
        
    def calculate_resize_params(self, original_width, original_height):
        """計算縮放參數（保持長寬比）"""
        target_width, target_height = self.display_size
        
        scale_w = target_width / original_width
        scale_h = target_height / original_height
        scale = min(scale_w, scale_h)
        
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        offset_x = (target_width - new_width) // 2
        offset_y = (target_height - new_height) // 2
        
        return scale, new_width, new_height, offset_x, offset_y
    
    def resize_image_for_display(self, image):
        """縮放圖片用於顯示（保持長寬比，加黑邊）"""
        h, w = image.shape[:2]
        scale, new_w, new_h, offset_x, offset_y = self.calculate_resize_params(w, h)
        
        self.scale = scale
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.display_width = new_w
        self.display_height = new_h
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        display = np.zeros((self.display_size[1], self.display_size[0], 3), dtype=np.uint8)
        display[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized
        
        return display
    
    def display_to_original_coords(self, display_x, display_y):
        """將顯示座標轉換為原始圖片座標"""
        x = display_x - self.offset_x
        y = display_y - self.offset_y
        
        if x < 0 or y < 0 or x >= self.display_width or y >= self.display_height:
            return None
        
        original_x = int(x / self.scale)
        original_y = int(y / self.scale)
        
        original_x = max(0, min(original_x, self.original_width - 1))
        original_y = max(0, min(original_y, self.original_height - 1))
        
        return [original_x, original_y]
    
    def original_to_display_coords(self, original_x, original_y):
        """將原始圖片座標轉換為顯示座標"""
        display_x = int(original_x * self.scale) + self.offset_x
        display_y = int(original_y * self.scale) + self.offset_y
        return display_x, display_y
        
    def mouse_callback(self, event, x, y, flags, param):
        """滑鼠點擊事件處理"""
        if event == cv2.EVENT_LBUTTONDOWN:
            original_coords = self.display_to_original_coords(x, y)
            
            if original_coords is not None:
                self.point = original_coords
                self.update_display_with_labels()
                
                cv2.circle(self.display_image, (x, y), 5, (0, 255, 0), -1)
                
                info_text = f"Click: ({original_coords[0]},{original_coords[1]}) - Press 's' to label"
                cv2.putText(self.display_image, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.imshow("Image", self.display_image)
                print(f"點擊座標 - 原始: ({original_coords[0]},{original_coords[1]})")
    
    def bbox_adjust_mouse_callback(self, event, x, y, flags, param):
        """Bounding box調整視窗的滑鼠回調"""
        if self.temp_bbox is None:
            return
        
        original_coords = self.display_to_original_coords(x, y)
        if original_coords is None:
            return
        
        x_orig, y_orig = original_coords
        x1, y1, x2, y2 = self.temp_bbox
        
        edge_threshold = 10
        
        if event == cv2.EVENT_MOUSEMOVE:
            if self.adjusting_bbox and self.bbox_edge:
                if self.bbox_edge == 'left':
                    self.temp_bbox[0] = min(x_orig, x2 - 10)
                elif self.bbox_edge == 'right':
                    self.temp_bbox[2] = max(x_orig, x1 + 10)
                elif self.bbox_edge == 'top':
                    self.temp_bbox[1] = min(y_orig, y2 - 10)
                elif self.bbox_edge == 'bottom':
                    self.temp_bbox[3] = max(y_orig, y1 + 10)
                
                self.temp_bbox[0] = max(0, self.temp_bbox[0])
                self.temp_bbox[1] = max(0, self.temp_bbox[1])
                self.temp_bbox[2] = min(self.original_width, self.temp_bbox[2])
                self.temp_bbox[3] = min(self.original_height, self.temp_bbox[3])
                
            else:
                near_left = abs(x_orig - x1) < edge_threshold and y1 <= y_orig <= y2
                near_right = abs(x_orig - x2) < edge_threshold and y1 <= y_orig <= y2
                near_top = abs(y_orig - y1) < edge_threshold and x1 <= x_orig <= x2
                near_bottom = abs(y_orig - y2) < edge_threshold and x1 <= x_orig <= x2
                
                if near_left:
                    self.bbox_edge = 'left'
                elif near_right:
                    self.bbox_edge = 'right'
                elif near_top:
                    self.bbox_edge = 'top'
                elif near_bottom:
                    self.bbox_edge = 'bottom'
                else:
                    self.bbox_edge = None
        
        elif event == cv2.EVENT_LBUTTONDOWN:
            if self.bbox_edge:
                self.adjusting_bbox = True
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.adjusting_bbox = False
    
    def load_existing_labels(self, image_path):
        """載入現有的標註文件"""
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        label_dir = os.path.join(os.path.dirname(image_path), "labels")
        label_path = os.path.join(label_dir, f"{base_name}.txt")
        
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        center_x = float(parts[1])
                        center_y = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        labels.append({
                            'class_id': class_id,
                            'center_x': center_x,
                            'center_y': center_y,
                            'width': width,
                            'height': height
                        })
            print(f"✓ 載入 {len(labels)} 個現有標註")
        
        return labels
    
    def draw_label_on_image(self, image, label, color=(0, 255, 255), thickness=2):
        """在圖片上繪製單個標註框"""
        center_x = label['center_x'] * self.original_width
        center_y = label['center_y'] * self.original_height
        width = label['width'] * self.original_width
        height = label['height'] * self.original_height
        
        x1 = int(center_x - width / 2)
        y1 = int(center_y - height / 2)
        x2 = int(center_x + width / 2)
        y2 = int(center_y + height / 2)
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        class_id = label['class_id']
        if class_id < len(self.classes):
            class_name = self.classes[class_id]
            label_text = f"{class_name}"
            
            label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (x1, y1 - label_size[1] - 8), 
                         (x1 + label_size[0] + 8, y1), color, -1)
            cv2.putText(image, label_text, (x1 + 4, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return image
    
    def draw_adjustable_bbox(self, image, bbox, color=(0, 255, 0), highlight_edge=None):
        """繪製可調整的bounding box"""
        x1, y1, x2, y2 = bbox
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        
        left_color = (255, 255, 0) if highlight_edge == 'left' else (255, 0, 0)
        cv2.line(image, (x1, y1), (x1, y2), left_color, 8 if highlight_edge == 'left' else 5)
        
        right_color = (255, 255, 0) if highlight_edge == 'right' else (255, 0, 0)
        cv2.line(image, (x2, y1), (x2, y2), right_color, 8 if highlight_edge == 'right' else 5)
        
        top_color = (255, 255, 0) if highlight_edge == 'top' else (255, 0, 0)
        cv2.line(image, (x1, y1), (x2, y1), top_color, 8 if highlight_edge == 'top' else 5)
        
        bottom_color = (255, 255, 0) if highlight_edge == 'bottom' else (255, 0, 0)
        cv2.line(image, (x1, y2), (x2, y2), bottom_color, 8 if highlight_edge == 'bottom' else 5)
        
        corner_size = 10
        cv2.circle(image, (x1, y1), corner_size, (0, 255, 255), -1)
        cv2.circle(image, (x2, y1), corner_size, (0, 255, 255), -1)
        cv2.circle(image, (x1, y2), corner_size, (0, 255, 255), -1)
        cv2.circle(image, (x2, y2), corner_size, (0, 255, 255), -1)
        
        return image
    
    def update_display_with_labels(self):
        """更新顯示圖像，包含所有現有標註"""
        display_img = self.current_image.copy()
        
        for label in self.current_labels:
            display_img = self.draw_label_on_image(display_img, label, color=(0, 255, 255), thickness=2)
        
        self.display_image = self.resize_image_for_display(display_img)
            
    def load_classes(self, classes_file="classes.txt"):
        """載入類別列表"""
        if os.path.exists(classes_file):
            with open(classes_file, 'r', encoding='utf-8') as f:
                self.classes = [line.strip() for line in f.readlines() if line.strip()]
            print(f"已載入類別: {self.classes}")
        else:
            print("未找到classes.txt，將在標註時新增類別")
            
    def save_classes(self, classes_file="classes.txt"):
        """儲存類別列表"""
        with open(classes_file, 'w', encoding='utf-8') as f:
            for cls in self.classes:
                f.write(f"{cls}\n")
    
    def class_selection_mouse_callback(self, event, x, y, flags, param):
        """類別選擇視窗的滑鼠回調"""
        if event == cv2.EVENT_LBUTTONDOWN and self.class_window_active:
            button_height = 40
            button_width = 400
            start_y = 50
            spacing = 10
            start_x = 50
            
            for idx in range(len(self.classes) + 1):
                button_y = start_y + idx * (button_height + spacing)
                
                if (start_x <= x <= start_x + button_width and 
                    button_y <= y <= button_y + button_height):
                    
                    if idx < len(self.classes):
                        self.selected_class = (idx, self.classes[idx])
                        print(f"選擇類別: {self.classes[idx]}")
                    else:
                        self.selected_class = self.add_new_class_dialog()
                    
                    self.class_window_active = False
                    break
    
    def add_new_class_dialog(self):
        """彈出對話框新增新類別"""
        root = tk.Tk()
        root.withdraw()
        
        new_class = simpledialog.askstring(
            "新增類別", 
            "請輸入新類別名稱:",
            parent=root
        )
        
        root.destroy()
        
        if new_class and new_class.strip():
            new_class = new_class.strip()
            self.classes.append(new_class)
            self.save_classes()
            print(f"新增類別: {new_class}")
            return (len(self.classes) - 1, new_class)
        
        return None
    
    def show_class_selection_window(self):
        """顯示類別選擇視窗（GUI界面）"""
        self.selected_class = None
        self.class_window_active = True
        
        window_name = "Select Class"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.class_selection_mouse_callback)
        
        button_height = 40
        button_width = 400
        start_y = 50
        spacing = 10
        start_x = 50
        
        num_buttons = len(self.classes) + 1
        window_height = start_y + num_buttons * (button_height + spacing) + 50
        window_width = button_width + 2 * start_x
        
        while self.class_window_active:
            canvas = np.ones((window_height, window_width, 3), dtype=np.uint8) * 255
            
            cv2.putText(canvas, "Select Object Class", (start_x, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            for idx, cls in enumerate(self.classes):
                button_y = start_y + idx * (button_height + spacing)
                
                cv2.rectangle(canvas, 
                            (start_x, button_y),
                            (start_x + button_width, button_y + button_height),
                            (100, 200, 100), -1)
                
                cv2.rectangle(canvas,
                            (start_x, button_y),
                            (start_x + button_width, button_y + button_height),
                            (0, 100, 0), 2)
                
                text = f"{idx}: {cls}"
                cv2.putText(canvas, text,
                           (start_x + 10, button_y + 27),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            new_class_y = start_y + len(self.classes) * (button_height + spacing)
            cv2.rectangle(canvas,
                        (start_x, new_class_y),
                        (start_x + button_width, new_class_y + button_height),
                        (100, 100, 200), -1)
            cv2.rectangle(canvas,
                        (start_x, new_class_y),
                        (start_x + button_width, new_class_y + button_height),
                        (0, 0, 100), 2)
            cv2.putText(canvas, "+ Add New Class",
                       (start_x + 10, new_class_y + 27),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            tip_y = new_class_y + button_height + 30
            cv2.putText(canvas, "Click to select or press ESC to cancel",
                       (start_x, tip_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            
            cv2.imshow(window_name, canvas)
            
            key = cv2.waitKey(50) & 0xFF
            if key == 27:
                self.class_window_active = False
                self.selected_class = None
                break
        
        cv2.destroyWindow(window_name)
        return self.selected_class
    
    def segment_with_sam2(self, image_path, point):
        """使用SAM2進行分割"""
        print(f"SAM2分割中...")
        
        results = self.model(image_path, points=[point], labels=[1])
        
        if len(results) > 0 and results[0].masks is not None:
            mask = results[0].masks.data[0].cpu().numpy()
            return mask
        return None
    
    def mask_to_bbox(self, mask):
        """將mask轉換為bounding box座標"""
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        contours, _ = cv2.findContours(
            mask_uint8, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
        
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        
        return [x, y, x + w, y + h]
    
    def bbox_to_yolo_format(self, bbox, class_id):
        """將bbox轉換為YOLO格式"""
        x1, y1, x2, y2 = bbox
        
        center_x = ((x1 + x2) / 2) / self.original_width
        center_y = ((y1 + y2) / 2) / self.original_height
        width = (x2 - x1) / self.original_width
        height = (y2 - y1) / self.original_height
        
        label_dict = {
            'class_id': class_id,
            'center_x': center_x,
            'center_y': center_y,
            'width': width,
            'height': height
        }
        
        return label_dict
    
    def save_yolo_labels(self, image_path, labels):
        """儲存所有YOLO標註到文件"""
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        label_dir = os.path.join(os.path.dirname(image_path), "labels")
        
        os.makedirs(label_dir, exist_ok=True)
        
        label_path = os.path.join(label_dir, f"{base_name}.txt")
        
        with open(label_path, 'w', encoding='utf-8') as f:
            for label in labels:
                line = f"{label['class_id']} {label['center_x']:.6f} {label['center_y']:.6f} {label['width']:.6f} {label['height']:.6f}"
                f.write(line + '\n')
        
        print(f"✓ 標註已儲存: {label_path}")
    
    def show_result_window(self, bbox, class_id, class_name):
        """顯示標註結果視窗，允許調整bounding box"""
        window_name = "Result - Adjust BBox"
        cv2.namedWindow(window_name)
        
        self.temp_bbox = list(bbox)
        self.adjusting_bbox = False
        self.bbox_edge = None
        
        cv2.setMouseCallback(window_name, self.bbox_adjust_mouse_callback)
        
        print(f"✓ SAM2分割完成: {class_name}")
        print("  → 拖動邊緣調整框的大小")
        print("  → 按 'c' 接受並儲存")
        print("  → 按 'x' 取消（不儲存）")
        
        while True:
            result_img = self.current_image.copy()
            result_img = self.draw_adjustable_bbox(result_img, self.temp_bbox, 
                                                   color=(0, 255, 0), 
                                                   highlight_edge=self.bbox_edge)
            
            overlay = result_img.copy()
            cv2.rectangle(overlay, (10, 10), (600, 130), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, result_img, 0.3, 0, result_img)
            
            cv2.putText(result_img, f"Class: {class_name}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(result_img, "Drag edges to adjust bbox", (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(result_img, "Press 'C' - Accept & Save", (20, 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(result_img, "Press 'X' - Cancel", (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            w = self.temp_bbox[2] - self.temp_bbox[0]
            h = self.temp_bbox[3] - self.temp_bbox[1]
            size_text = f"Size: {w}x{h}"
            cv2.putText(result_img, size_text, 
                       (self.temp_bbox[0], self.temp_bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            display_result = self.resize_image_for_display(result_img)
            cv2.imshow(window_name, display_result)
            
            key = cv2.waitKey(10) & 0xFF
            if key == ord('c'):
                cv2.destroyWindow(window_name)
                label_dict = self.bbox_to_yolo_format(self.temp_bbox, class_id)
                return 'accept', label_dict
            elif key == ord('x') or key == 27:
                cv2.destroyWindow(window_name)
                return 'cancel', None
    
    def process_image(self, image_path):
        """處理單張圖像的標註流程"""
        self.current_image_path = image_path
        self.current_image = cv2.imread(image_path)
        
        if self.current_image is None:
            print(f"❌ 無法載入圖像: {image_path}")
            return 'error'
        
        self.original_height, self.original_width = self.current_image.shape[:2]
        
        self.current_labels = self.load_existing_labels(image_path)
        self.update_display_with_labels()
        
        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", self.mouse_callback)
        
        print(f"\n{'='*60}")
        print(f"圖像: {os.path.basename(image_path)}")
        print(f"尺寸: {self.original_width}x{self.original_height}")
        print(f"現有標註: {len(self.current_labels)} 個")
        print(f"{'='*60}")
        print("操作:")
        print("  • 點擊物體 → 按 's' 標註")
        print("  • 按 'r' 重置點")
        print("  • 按 'a' 上一張")
        print("  • 按 'd' 下一張")
        print("  • 按 'q' 退出")
        print(f"{'='*60}")
        
        while True:
            cv2.imshow("Image", self.display_image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                cv2.destroyAllWindows()
                return 'quit'
            elif key == ord('d'):
                cv2.destroyAllWindows()
                return 'next'
            elif key == ord('a'):
                cv2.destroyAllWindows()
                return 'prev'
            elif key == ord('r'):
                self.point = None
                self.update_display_with_labels()
                print("✓ 已重置")
            elif key == ord('s') and self.point is not None:
                result = self.show_class_selection_window()
                
                if result is not None:
                    class_id, class_name = result
                    print(f"→ 類別: {class_name}")
                    
                    mask = self.segment_with_sam2(image_path, self.point)
                    
                    if mask is not None:
                        bbox = self.mask_to_bbox(mask)
                        
                        if bbox:
                            user_choice, label_dict = self.show_result_window(bbox, class_id, class_name)
                            
                            if user_choice == 'accept' and label_dict:
                                self.current_labels.append(label_dict)
                                self.save_yolo_labels(image_path, self.current_labels)
                                print("✓ 已接受並儲存")
                                self.update_display_with_labels()
                            elif user_choice == 'cancel':
                                print("✗ 已取消，不儲存")
                        else:
                            print("❌ 轉換失敗")
                    else:
                        print("❌ 分割失敗")
                else:
                    print("✗ 取消標註")
                
                self.point = None
                self.update_display_with_labels()
        
        cv2.destroyAllWindows()
        return 'next'


def show_main_menu():
    """顯示主選單"""
    print("\n" + "="*60)
    print("YOLO 標註工具集")
    print("="*60)
    print("請選擇功能:")
    print("  1. 標註工具 (SAM2 輔助標註)")
    print("  2. 資料集分割 (Train/Val 分割)")
    print("  3. 退出")
    print("="*60)
    
    choice = input("請輸入選項 (1-3): ").strip()
    return choice


def main():
    while True:
        choice = show_main_menu()
        
        if choice == '1':
            # 標註工具
            print("\n啟動標註工具...")
            tool = YOLOLabelTool(
                model_path="sam2_b.pt",
                display_size=(1280, 720)
            )
            
            tool.load_classes("classes.txt")
            
            # 選擇圖片資料夾
            root = tk.Tk()
            root.withdraw()
            image_folder = filedialog.askdirectory(title="選擇包含圖片的資料夾")
            root.destroy()
            
            if not image_folder:
                print("❌ 未選擇資料夾")
                continue
            
            image_files = [f for f in sorted(os.listdir(image_folder)) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            if not image_files:
                print("❌ 資料夾中沒有圖片")
                continue
            
            current_idx = 0
            
            while 0 <= current_idx < len(image_files):
                filename = image_files[current_idx]
                print(f"\n[{current_idx+1}/{len(image_files)}]")
                image_path = os.path.join(image_folder, filename)
                
                result = tool.process_image(image_path)
                
                if result == 'quit':
                    print("\n✓ 使用者退出標註")
                    break
                elif result == 'next':
                    current_idx += 1
                elif result == 'prev':
                    current_idx -= 1
                    if current_idx < 0:
                        current_idx = 0
                        print("已經是第一張圖片")
                elif result == 'error':
                    current_idx += 1
            
            print("\n✓ 標註完成!")
            
        elif choice == '2':
            # 資料集分割
            print("\n啟動資料集分割工具...")
            splitter = DatasetSplitter(train_ratio=0.8)
            splitter.interactive_split()
            
        elif choice == '3':
            print("\n👋 再見！")
            break
        
        else:
            print("\n❌ 無效選項，請重新選擇")


if __name__ == "__main__":
    main()
