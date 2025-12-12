import cv2
import numpy as np
from ultralytics import SAM
import os
from tkinter import simpledialog
import tkinter as tk

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
        
    def calculate_resize_params(self, original_width, original_height):
        """計算縮放參數（保持長寬比）"""
        target_width, target_height = self.display_size
        
        # 計算縮放比例
        scale_w = target_width / original_width
        scale_h = target_height / original_height
        scale = min(scale_w, scale_h)
        
        # 計算縮放後的實際尺寸
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # 計算居中偏移量
        offset_x = (target_width - new_width) // 2
        offset_y = (target_height - new_height) // 2
        
        return scale, new_width, new_height, offset_x, offset_y
    
    def resize_image_for_display(self, image):
        """縮放圖片用於顯示（保持長寬比，加黑邊）"""
        h, w = image.shape[:2]
        scale, new_w, new_h, offset_x, offset_y = self.calculate_resize_params(w, h)
        
        # 儲存縮放參數
        self.scale = scale
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.display_width = new_w
        self.display_height = new_h
        
        # 縮放圖片
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 創建黑色背景
        display = np.zeros((self.display_size[1], self.display_size[0], 3), dtype=np.uint8)
        
        # 將縮放後的圖片放到中央
        display[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized
        
        return display
    
    def display_to_original_coords(self, display_x, display_y):
        """將顯示座標轉換為原始圖片座標"""
        # 減去偏移量
        x = display_x - self.offset_x
        y = display_y - self.offset_y
        
        # 檢查是否在有效區域內
        if x < 0 or y < 0 or x >= self.display_width or y >= self.display_height:
            return None
        
        # 縮放回原始尺寸
        original_x = int(x / self.scale)
        original_y = int(y / self.scale)
        
        # 確保不超出原始圖片範圍
        original_x = max(0, min(original_x, self.original_width - 1))
        original_y = max(0, min(original_y, self.original_height - 1))
        
        return [original_x, original_y]
        
    def mouse_callback(self, event, x, y, flags, param):
        """滑鼠點擊事件處理"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 轉換為原始圖片座標
            original_coords = self.display_to_original_coords(x, y)
            
            if original_coords is not None:
                self.point = original_coords
                
                # 重新繪製圖像（包含現有標註和新點）
                self.update_display_with_labels()
                
                # 在顯示圖像上標記點擊位置
                cv2.circle(self.display_image, (x, y), 5, (0, 255, 0), -1)
                
                # 顯示座標資訊
                info_text = f"Click: ({original_coords[0]},{original_coords[1]}) - Press 's' to label"
                cv2.putText(self.display_image, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.imshow("Image", self.display_image)
                print(f"點擊座標 - 原始: ({original_coords[0]},{original_coords[1]})")
    
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
                    if len(parts) == 5:  # class_id center_x center_y width height
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
        # 將YOLO格式轉換回像素坐標
        center_x = label['center_x'] * self.original_width
        center_y = label['center_y'] * self.original_height
        width = label['width'] * self.original_width
        height = label['height'] * self.original_height
        
        x1 = int(center_x - width / 2)
        y1 = int(center_y - height / 2)
        x2 = int(center_x + width / 2)
        y2 = int(center_y + height / 2)
        
        # 繪製矩形
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # 繪製類別標籤
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
    
    def update_display_with_labels(self):
        """更新顯示圖像，包含所有現有標註"""
        # 複製原始圖像
        display_img = self.current_image.copy()
        
        # 繪製所有現有標註
        for label in self.current_labels:
            display_img = self.draw_label_on_image(display_img, label, color=(0, 255, 255), thickness=2)
        
        # 縮放到顯示尺寸
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
            
            # 檢查點擊在哪個按鈕上
            for idx in range(len(self.classes) + 1):
                button_y = start_y + idx * (button_height + spacing)
                
                if (start_x <= x <= start_x + button_width and 
                    button_y <= y <= button_y + button_height):
                    
                    if idx < len(self.classes):
                        # 選擇現有類別
                        self.selected_class = (idx, self.classes[idx])
                        print(f"選擇類別: {self.classes[idx]}")
                    else:
                        # 新增新類別
                        self.selected_class = self.add_new_class_dialog()
                    
                    self.class_window_active = False
                    break
    
    def add_new_class_dialog(self):
        """彈出對話框新增新類別"""
        root = tk.Tk()
        root.withdraw()  # 隱藏主視窗
        
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
        
        # 創建視窗
        window_name = "Select Class"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.class_selection_mouse_callback)
        
        # 計算視窗大小
        button_height = 40
        button_width = 400
        start_y = 50
        spacing = 10
        start_x = 50
        
        num_buttons = len(self.classes) + 1  # 現有類別 + 新增按鈕
        window_height = start_y + num_buttons * (button_height + spacing) + 50
        window_width = button_width + 2 * start_x
        
        while self.class_window_active:
            # 創建白色背景
            canvas = np.ones((window_height, window_width, 3), dtype=np.uint8) * 255
            
            # 繪製標題
            cv2.putText(canvas, "Select Object Class", (start_x, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            # 繪製現有類別按鈕
            for idx, cls in enumerate(self.classes):
                button_y = start_y + idx * (button_height + spacing)
                
                # 繪製按鈕背景
                cv2.rectangle(canvas, 
                            (start_x, button_y),
                            (start_x + button_width, button_y + button_height),
                            (100, 200, 100), -1)
                
                # 繪製按鈕邊框
                cv2.rectangle(canvas,
                            (start_x, button_y),
                            (start_x + button_width, button_y + button_height),
                            (0, 100, 0), 2)
                
                # 繪製類別名稱和編號
                text = f"{idx}: {cls}"
                cv2.putText(canvas, text,
                           (start_x + 10, button_y + 27),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # 繪製"新增類別"按鈕
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
            
            # 顯示提示
            tip_y = new_class_y + button_height + 30
            cv2.putText(canvas, "Click to select or press ESC to cancel",
                       (start_x, tip_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            
            cv2.imshow(window_name, canvas)
            
            # 等待按鍵
            key = cv2.waitKey(50) & 0xFF
            if key == 27:  # ESC鍵
                self.class_window_active = False
                self.selected_class = None
                break
        
        cv2.destroyWindow(window_name)
        return self.selected_class
    
    def segment_with_sam2(self, image_path, point):
        """使用SAM2進行分割（使用原始圖片和原始座標）"""
        print(f"SAM2分割中...")
        
        # 執行SAM2推理（使用原始圖片）
        results = self.model(image_path, points=[point], labels=[1])
        
        # 取得分割結果
        if len(results) > 0 and results[0].masks is not None:
            mask = results[0].masks.data[0].cpu().numpy()
            return mask
        return None
    
    def mask_to_yolo_bbox(self, mask, class_id):
        """將mask轉換為YOLO格式的bounding box（基於原始圖片尺寸）"""
        # 將mask轉換為uint8
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # 尋找輪廓
        contours, _ = cv2.findContours(
            mask_uint8, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
        
        # 取得最大的輪廓
        contour = max(contours, key=cv2.contourArea)
        
        # 計算bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # 計算中心點和正規化座標
        center_x = (x + w / 2) / self.original_width
        center_y = (y + h / 2) / self.original_height
        norm_w = w / self.original_width
        norm_h = h / self.original_height
        
        # YOLO格式: class_id center_x center_y width height
        yolo_line = f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}"
        
        # 創建label字典
        label_dict = {
            'class_id': class_id,
            'center_x': center_x,
            'center_y': center_y,
            'width': norm_w,
            'height': norm_h
        }
        
        return yolo_line, label_dict
    
    def save_yolo_labels(self, image_path, labels):
        """儲存所有YOLO標註到文件"""
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        label_dir = os.path.join(os.path.dirname(image_path), "labels")
        
        # 創建labels資料夾（如果不存在）
        os.makedirs(label_dir, exist_ok=True)
        
        label_path = os.path.join(label_dir, f"{base_name}.txt")
        
        # 寫入所有標註
        with open(label_path, 'w', encoding='utf-8') as f:
            for label in labels:
                line = f"{label['class_id']} {label['center_x']:.6f} {label['center_y']:.6f} {label['width']:.6f} {label['height']:.6f}"
                f.write(line + '\n')
        
        print(f"✓ 標註已儲存: {label_path}")
    
    def show_result_window(self, result_image, class_name):
        """顯示標註結果視窗"""
        window_name = "Result"
        cv2.namedWindow(window_name)
        cv2.imshow(window_name, result_image)
        
        print(f"✓ 標註完成: {class_name}")
        print("按 'c' 繼續標註")
        
        while True:
            key = cv2.waitKey(50) & 0xFF
            if key == ord('c'):
                cv2.destroyWindow(window_name)
                return True
            elif key == 27:  # ESC
                cv2.destroyWindow(window_name)
                return False
    
    def process_image(self, image_path):
        """處理單張圖像的標註流程"""
        self.current_image_path = image_path
        self.current_image = cv2.imread(image_path)
        
        if self.current_image is None:
            print(f"❌ 無法載入圖像: {image_path}")
            return 'error'
        
        # 儲存原始尺寸
        self.original_height, self.original_width = self.current_image.shape[:2]
        
        # 載入現有標註
        self.current_labels = self.load_existing_labels(image_path)
        
        # 更新顯示（包含現有標註）
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
            elif key == ord('d'):  # 下一張
                cv2.destroyAllWindows()
                return 'next'
            elif key == ord('a'):  # 上一張
                cv2.destroyAllWindows()
                return 'prev'
            elif key == ord('r'):
                self.point = None
                self.update_display_with_labels()
                print("✓ 已重置")
            elif key == ord('s') and self.point is not None:
                # 顯示類別選擇視窗
                result = self.show_class_selection_window()
                
                if result is not None:
                    class_id, class_name = result
                    print(f"→ 類別: {class_name}")
                    
                    # 使用SAM2分割
                    mask = self.segment_with_sam2(image_path, self.point)
                    
                    if mask is not None:
                        # 轉換為YOLO bounding box格式
                        bbox_result = self.mask_to_yolo_bbox(mask, class_id)
                        
                        if bbox_result:
                            yolo_line, label_dict = bbox_result
                            
                            # 添加到當前標註列表
                            self.current_labels.append(label_dict)
                            
                            # 儲存所有標註
                            self.save_yolo_labels(image_path, self.current_labels)
                            
                            # 繪製結果
                            result_img = self.current_image.copy()
                            result_img = self.draw_label_on_image(result_img, label_dict, color=(0, 255, 0), thickness=3)
                            display_result = self.resize_image_for_display(result_img)
                            
                            # 顯示結果視窗，按'c'繼續
                            if self.show_result_window(display_result, class_name):
                                # 更新主視窗顯示
                                self.update_display_with_labels()
                            
                        else:
                            print("❌ 轉換失敗")
                    else:
                        print("❌ 分割失敗")
                else:
                    print("✗ 取消標註")
                
                # 重置點
                self.point = None
                self.update_display_with_labels()
        
        cv2.destroyAllWindows()
        return 'next'

def main():
    print("="*60)
    print("YOLO 快速標註工具 (SAM2)")
    print("="*60)
    
    # 初始化工具
    tool = YOLOLabelTool(
        model_path="sam2_b.pt",
        display_size=(1280, 720)
    )
    
    # 載入現有類別
    tool.load_classes("classes.txt")
    
    # 批次處理資料夾
    image_folder = "40_1"
    image_files = [f for f in sorted(os.listdir(image_folder)) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print("❌ 資料夾中沒有圖片")
        return
    
    current_idx = 0
    
    while 0 <= current_idx < len(image_files):
        filename = image_files[current_idx]
        print(f"\n[{current_idx+1}/{len(image_files)}]")
        image_path = os.path.join(image_folder, filename)
        
        result = tool.process_image(image_path)
        
        if result == 'quit':
            print("\n✓ 使用者退出")
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

if __name__ == "__main__":
    main()
