import cv2
import numpy as np
from ultralytics import SAM, YOLO
import os
from tkinter import simpledialog, filedialog, messagebox
import tkinter as tk
import shutil
import random


# ─────────────────────────────────────────────────────────────
# DatasetSplitter（不變）
# ─────────────────────────────────────────────────────────────
class DatasetSplitter:
    def __init__(self, train_ratio=0.8, val_ratio=0.2, test_ratio=0.0):
        self.train_ratio = train_ratio
        self.val_ratio   = val_ratio
        self.test_ratio  = test_ratio

    def select_source_folder(self):
        root = tk.Tk(); root.withdraw()
        p = filedialog.askdirectory(title="選擇包含圖片的資料夾")
        root.destroy(); return p

    def select_output_folder(self):
        root = tk.Tk(); root.withdraw()
        p = filedialog.askdirectory(title="選擇輸出資料夾（將創建 train/val/test 結構）")
        root.destroy(); return p

    def get_image_label_pairs(self, source_folder):
        labels_folder = os.path.join(source_folder, "labels")
        if not os.path.exists(labels_folder):
            print(f"❌ 找不到 labels 資料夾: {labels_folder}"); return []
        exts = ('.jpg', '.jpeg', '.png', '.bmp')
        pairs = []
        for f in os.listdir(source_folder):
            if f.lower().endswith(exts) and os.path.isfile(os.path.join(source_folder, f)):
                base = os.path.splitext(f)[0]
                lp = os.path.join(labels_folder, f"{base}.txt")
                if os.path.exists(lp):
                    pairs.append({'image': f, 'label': f"{base}.txt",
                                  'image_path': os.path.join(source_folder, f),
                                  'label_path': lp})
                else:
                    print(f"⚠️  跳過（無標註）: {f}")
        return pairs

    def create_output_structure(self, output_folder, include_test=True):
        folders = [os.path.join(output_folder, "images", "train"),
                   os.path.join(output_folder, "images", "val"),
                   os.path.join(output_folder, "labels", "train"),
                   os.path.join(output_folder, "labels", "val")]
        if include_test:
            folders += [os.path.join(output_folder, "images", "test"),
                        os.path.join(output_folder, "labels", "test")]
        for f in folders:
            os.makedirs(f, exist_ok=True); print(f"✓ 創建: {f}")

    def split_and_copy(self, source_folder, output_folder,
                       train_ratio=None, val_ratio=None, test_ratio=None):
        if train_ratio is not None: self.train_ratio = train_ratio
        if val_ratio   is not None: self.val_ratio   = val_ratio
        if test_ratio  is not None: self.test_ratio  = test_ratio
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 0.01:
            self.train_ratio /= total; self.val_ratio /= total; self.test_ratio /= total
        pairs = self.get_image_label_pairs(source_folder)
        if not pairs: print("❌ 未找到有效的圖片-標註配對"); return
        random.shuffle(pairs)
        tc = int(len(pairs) * self.train_ratio)
        vc = int(len(pairs) * self.val_ratio)
        splits = [("train", pairs[:tc]), ("val", pairs[tc:tc+vc]), ("test", pairs[tc+vc:])]
        self.create_output_structure(output_folder, include_test=bool(pairs[tc+vc:]))
        for name, sp in splits:
            if not sp: continue
            print(f"\n複製 {name.capitalize()} ...")
            for i, p in enumerate(sp, 1):
                shutil.copy2(p['image_path'], os.path.join(output_folder,"images",name,p['image']))
                shutil.copy2(p['label_path'], os.path.join(output_folder,"labels",name,p['label']))
                if i % 10 == 0 or i == len(sp): print(f"  {i}/{len(sp)}")
        print(f"\n✓ 分割完成！\n")

    def interactive_split(self):
        source = self.select_source_folder()
        if not source: print("❌ 取消"); return
        output = self.select_output_folder()
        if not output: print("❌ 取消"); return
        root = tk.Tk(); root.title("設定分割比例"); root.geometry("400x250")
        root.update_idletasks()
        root.geometry(f'400x250+{root.winfo_screenwidth()//2-200}+{root.winfo_screenheight()//2-125}')
        tk.Label(root, text="請輸入分割比例 (%)", font=("Arial",14,"bold")).pack(pady=10)
        entries = {}
        for lbl, default in [("Train:","80"),("Val:","20"),("Test:","0")]:
            f = tk.Frame(root); f.pack(pady=5)
            tk.Label(f, text=lbl, width=10).pack(side=tk.LEFT)
            e = tk.Entry(f, width=10); e.insert(0,default); e.pack(side=tk.LEFT)
            tk.Label(f, text="%").pack(side=tk.LEFT); entries[lbl] = e
        tk.Label(root, text="總和應為 100%", fg="gray").pack(pady=5)
        result = {'confirmed': False}
        def on_confirm():
            try:
                t,v,ts = float(entries["Train:"].get()),float(entries["Val:"].get()),float(entries["Test:"].get())
                if abs(t+v+ts-100)>0.1: messagebox.showwarning("警告",f"總和 {t+v+ts}%，將自動正規化")
                result.update({'train':t/100,'val':v/100,'test':ts/100,'confirmed':True}); root.destroy()
            except ValueError: messagebox.showerror("錯誤","請輸入有效的數字")
        bf = tk.Frame(root); bf.pack(pady=15)
        tk.Button(bf,text="確認",command=on_confirm,width=10,bg="#4CAF50",fg="white").pack(side=tk.LEFT,padx=5)
        tk.Button(bf,text="取消",command=root.destroy,width=10,bg="#f44336",fg="white").pack(side=tk.LEFT,padx=5)
        root.mainloop()
        if not result['confirmed']: return
        self.split_and_copy(source, output, result['train'], result['val'], result['test'])


# ─────────────────────────────────────────────────────────────
# YOLOAutoLabeler  ── 補標缺漏 class + 逐張審閱
# ─────────────────────────────────────────────────────────────
class YOLOAutoLabeler:
    """
    核心邏輯
    ─────────
    1. 讀每張圖的 .txt，取得「已有的 class set」
    2. missing_classes = 所有 class - 已有 class
    3. 若 missing_classes 非空 → 執行 YOLO 推論，只保留缺漏 class 的框
    4. 有新偵測 → 加入審閱佇列
    5. 審閱視窗：現有標註（灰色🔒）+ 新偵測（彩色，可調整/刪除）
    6. 使用者按 D 儲存，最終 .txt = 現有 + 核准的新框
    """

    COLORS = [
        (50,205,50),(30,144,255),(0,140,255),(0,206,209),
        (238,130,238),(127,255,0),(255,140,0),(64,224,208),
        (220,20,60),(255,215,0),
    ]
    LOCK_COLOR  = (150, 150, 150)   # 現有標註：灰色
    LOCK_TEXT   = (220, 220, 220)

    def __init__(self):
        self.model           = None
        self.conf_threshold  = 0.25
        self.class_names     = {}    # {id: name}

        self.display_size    = (1280, 720)
        self.original_width  = 0
        self.original_height = 0
        self.scale           = 1.0
        self.offset_x        = 0
        self.offset_y        = 0
        self.disp_w          = 0
        self.disp_h          = 0
        self.current_image   = None

        # 審閱狀態
        # box dict 欄位: class_id, cx, cy, w, h, locked(bool)
        self.review_boxes        = []
        self.review_selected_idx = -1
        self.review_dragging     = False
        self.review_drag_edge    = None
        self.review_hover_edge   = None

    # ── GUI ──────────────────────────────────────────────────

    def _pick_weights(self):
        root = tk.Tk(); root.withdraw()
        p = filedialog.askopenfilename(title="選擇 YOLO 權重檔 (.pt)",
                                       filetypes=[("PyTorch weights","*.pt"),("All","*.*")])
        root.destroy(); return p

    def _pick_folder(self):
        root = tk.Tk(); root.withdraw()
        p = filedialog.askdirectory(title="選擇要補標的圖片資料夾")
        root.destroy(); return p

    def _ask_confidence(self):
        """簡單對話框：只問信心度門檻"""
        root = tk.Tk(); root.title("補標設定"); root.geometry("380x170")
        root.update_idletasks()
        root.geometry(f'380x170+{root.winfo_screenwidth()//2-190}+{root.winfo_screenheight()//2-85}')

        tk.Label(root, text="YOLO 補標設定", font=("Arial",14,"bold")).pack(pady=12)

        f = tk.Frame(root); f.pack(pady=8)
        tk.Label(f, text="信心度門檻 (0~1):", font=("Arial",11), width=20, anchor="w").pack(side=tk.LEFT)
        conf_var = tk.StringVar(value="0.25")
        tk.Entry(f, textvariable=conf_var, width=8, font=("Arial",11)).pack(side=tk.LEFT)

        tk.Label(root,
                 text="※ 僅對「缺漏 class」的圖片執行推論，現有標註絕不覆蓋",
                 fg="#555", font=("Arial",9)).pack(pady=4)

        result = {'conf': None}
        def ok():
            try:
                v = float(conf_var.get())
                if not (0 < v <= 1): raise ValueError
                result['conf'] = v; root.destroy()
            except ValueError:
                messagebox.showerror("錯誤","請輸入 0~1 之間的數值（例如 0.25）")

        bf = tk.Frame(root); bf.pack(pady=10)
        tk.Button(bf, text="開始分析", command=ok,
                  width=12, bg="#4CAF50", fg="white", font=("Arial",10,"bold")).pack(side=tk.LEFT, padx=5)
        tk.Button(bf, text="取消", command=root.destroy,
                  width=12, bg="#f44336", fg="white").pack(side=tk.LEFT, padx=5)
        root.mainloop()
        return result['conf']

    # ── 座標工具 ─────────────────────────────────────────────

    def _setup_display(self, image):
        h, w = image.shape[:2]
        self.original_width, self.original_height = w, h
        tw, th = self.display_size
        s = min(tw/w, th/h)
        self.scale  = s
        self.disp_w = int(w*s); self.disp_h = int(h*s)
        self.offset_x = (tw - self.disp_w)//2
        self.offset_y = (th - self.disp_h)//2

    def _to_display(self, image):
        r = cv2.resize(image, (self.disp_w, self.disp_h), interpolation=cv2.INTER_LINEAR)
        c = np.zeros((self.display_size[1], self.display_size[0], 3), dtype=np.uint8)
        c[self.offset_y:self.offset_y+self.disp_h, self.offset_x:self.offset_x+self.disp_w] = r
        return c

    def _d2o(self, dx, dy):
        """display → original coords; None if outside image"""
        x, y = dx - self.offset_x, dy - self.offset_y
        if x < 0 or y < 0 or x >= self.disp_w or y >= self.disp_h:
            return None
        return (max(0, min(int(x/self.scale), self.original_width-1)),
                max(0, min(int(y/self.scale), self.original_height-1)))

    # ── box 工具 ─────────────────────────────────────────────

    def _bp(self, box):
        """box → (x1,y1,x2,y2) pixel"""
        cx = box['cx']*self.original_width;  cy = box['cy']*self.original_height
        w  = box['w'] *self.original_width;  h  = box['h'] *self.original_height
        return int(cx-w/2), int(cy-h/2), int(cx+w/2), int(cy+h/2)

    def _pn(self, x1, y1, x2, y2):
        """pixel → normalized (cx,cy,w,h)"""
        return ((x1+x2)/2/self.original_width,  (y1+y2)/2/self.original_height,
                (x2-x1)/self.original_width,     (y2-y1)/self.original_height)

    def _edge_near(self, ox, oy, box, T=14):
        x1,y1,x2,y2 = self._bp(box)
        if abs(ox-x1)<T and y1-T<=oy<=y2+T: return 'left'
        if abs(ox-x2)<T and y1-T<=oy<=y2+T: return 'right'
        if abs(oy-y1)<T and x1-T<=ox<=x2+T: return 'top'
        if abs(oy-y2)<T and x1-T<=ox<=x2+T: return 'bottom'
        return None

    def _drag(self, box, edge, ox, oy, MIN=10):
        x1,y1,x2,y2 = self._bp(box)
        if   edge=='left':   x1 = max(0,                   min(ox, x2-MIN))
        elif edge=='right':  x2 = min(self.original_width,  max(ox, x1+MIN))
        elif edge=='top':    y1 = max(0,                   min(oy, y2-MIN))
        elif edge=='bottom': y2 = min(self.original_height, max(oy, y1+MIN))
        box['cx'], box['cy'], box['w'], box['h'] = self._pn(x1,y1,x2,y2)

    # ── 滑鼠 ─────────────────────────────────────────────────

    def _mouse(self, event, dx, dy, flags, param):
        orig = self._d2o(dx, dy)

        if event == cv2.EVENT_MOUSEMOVE:
            if orig is None: return
            ox, oy = orig
            if self.review_dragging and self.review_drag_edge and self.review_selected_idx >= 0:
                b = self.review_boxes[self.review_selected_idx]
                if not b.get('locked'):
                    self._drag(b, self.review_drag_edge, ox, oy)
            elif self.review_selected_idx >= 0:
                b = self.review_boxes[self.review_selected_idx]
                self.review_hover_edge = None if b.get('locked') else self._edge_near(ox, oy, b)

        elif event == cv2.EVENT_LBUTTONDOWN:
            if orig is None: self.review_selected_idx = -1; return
            ox, oy = orig
            # 嘗試拖曳已選取框的邊緣
            if self.review_selected_idx >= 0:
                b = self.review_boxes[self.review_selected_idx]
                if not b.get('locked'):
                    e = self._edge_near(ox, oy, b)
                    if e:
                        self.review_dragging   = True
                        self.review_drag_edge  = e
                        return
            # 選取框（從最上層往下找）
            for i in range(len(self.review_boxes)-1, -1, -1):
                b = self.review_boxes[i]
                x1,y1,x2,y2 = self._bp(b)
                if x1<=ox<=x2 and y1<=oy<=y2:
                    self.review_selected_idx = i
                    self.review_hover_edge   = None
                    if b.get('locked'):
                        print(f"  ℹ️  [{i}] {self.class_names.get(b['class_id'],b['class_id'])} — 現有標註（🔒不可編輯）")
                    return
            self.review_selected_idx = -1

        elif event == cv2.EVENT_LBUTTONUP:
            self.review_dragging  = False
            self.review_drag_edge = None

    # ── 繪製審閱畫面 ─────────────────────────────────────────

    def _draw_review(self, item, image_idx, total):
        img = self.current_image.copy()

        for i, box in enumerate(self.review_boxes):
            x1,y1,x2,y2 = self._bp(box)
            locked   = box.get('locked', False)
            selected = (i == self.review_selected_idx)
            color    = self.LOCK_COLOR if locked else self.COLORS[box['class_id'] % len(self.COLORS)]

            # 框
            cv2.rectangle(img, (x1,y1), (x2,y2), color, 3 if selected else 2)

            # 標籤
            cname = self.class_names.get(box['class_id'], f"cls{box['class_id']}")
            prefix = "🔒" if locked else f"[{i}]"
            label  = f"{prefix} {cname}"
            (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(img, (x1, y1-th-6), (x1+tw+6, y1), color, -1)
            txt_color = (30,30,30) if locked else (0,0,0)
            cv2.putText(img, label, (x1+3, y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, txt_color, 2)

            # 選取框的邊緣把手（只有非鎖定框才有）
            if selected and not locked:
                h = self.review_hover_edge
                for edge, pt1, pt2 in [('left',(x1,y1),(x1,y2)),('right',(x2,y1),(x2,y2)),
                                        ('top',(x1,y1),(x2,y1)),('bottom',(x1,y2),(x2,y2))]:
                    ec = (255,255,0) if h==edge else (255,80,0)
                    ew = 7          if h==edge else 4
                    cv2.line(img, pt1, pt2, ec, ew)
                for pt in [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]:
                    cv2.circle(img, pt, 8, (0,255,255), -1)

        display = self._to_display(img)
        dh, dw  = display.shape[:2]

        # ── 上方資訊列 ──
        cv2.rectangle(display, (0,0), (dw,58), (20,20,20), -1)

        fname    = os.path.basename(item['image_path'])
        new_cnt  = sum(1 for b in self.review_boxes if not b.get('locked'))
        lock_cnt = sum(1 for b in self.review_boxes if b.get('locked'))

        # 缺漏 class 名稱列表
        missing_names = [self.class_names.get(c, f"cls{c}") for c in sorted(item['missing_classes'])]
        missing_str   = ", ".join(missing_names)

        line1 = f"[{image_idx+1}/{total}]  {fname}"
        line2 = f"缺漏 class: {missing_str}   |   🔒現有: {lock_cnt}   新偵測: {new_cnt}"
        if self.review_selected_idx >= 0:
            sel = self.review_boxes[self.review_selected_idx]
            sn  = self.class_names.get(sel['class_id'], f"cls{sel['class_id']}")
            tag = "🔒鎖定" if sel.get('locked') else "✏️ 選取中"
            line2 += f"   |   {tag}: {sn}"

        cv2.putText(display, line1, (10,22), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255,255,255), 2)
        cv2.putText(display, line2, (10,46), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180,220,255), 1)

        # ── 下方操作提示列 ──
        cv2.rectangle(display, (0, dh-36), (dw, dh), (20,20,20), -1)
        ctrl = ("  點擊=選框  |  拖曳邊緣=調整大小  |  Del/BackSpace=刪除選取框  |"
                "  D / Enter=儲存並下一張  |  A=儲存並上一張  |  X=跳過不儲存  |  Q=退出")
        cv2.putText(display, ctrl, (8, dh-12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180,180,180), 1)

        return display

    # ── 資料讀寫 ─────────────────────────────────────────────

    def _load_existing(self, image_path):
        """讀取現有 .txt，回傳 locked box list"""
        base  = os.path.splitext(os.path.basename(image_path))[0]
        lpath = os.path.join(os.path.dirname(image_path), "labels", f"{base}.txt")
        boxes = []
        if os.path.exists(lpath):
            with open(lpath) as f:
                for line in f:
                    p = line.strip().split()
                    if len(p) == 5:
                        boxes.append({'class_id': int(p[0]),
                                      'cx': float(p[1]), 'cy': float(p[2]),
                                      'w':  float(p[3]), 'h':  float(p[4]),
                                      'locked': True})
        return boxes

    def _infer(self, image_path):
        """YOLO 推論，回傳所有偵測（unlocked）"""
        results = self.model(image_path, conf=self.conf_threshold, verbose=False)
        boxes = []
        if results and results[0].boxes is not None:
            for b in results[0].boxes:
                cx,cy,w,h = b.xywhn[0].tolist()
                boxes.append({'class_id': int(b.cls[0]),
                               'cx':cx, 'cy':cy, 'w':w, 'h':h,
                               'locked': False})
        return boxes

    def _save(self, image_path, boxes):
        """將 box list 寫入 .txt（自動過濾 locked 欄位）"""
        base      = os.path.splitext(os.path.basename(image_path))[0]
        label_dir = os.path.join(os.path.dirname(image_path), "labels")
        os.makedirs(label_dir, exist_ok=True)
        lpath = os.path.join(label_dir, f"{base}.txt")
        with open(lpath, 'w') as f:
            for b in boxes:
                f.write(f"{b['class_id']} {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}\n")
        return lpath

    # ── 核心：分析所有圖片，建立審閱佇列 ────────────────────

    def _build_review_queue(self, files, folder):
        """
        對每張圖：
          1. 讀現有 .txt，確認已有哪些 class
          2. missing_classes = 全部 class - 已有 class
          3. 若有缺漏 → YOLO 推論，只保留缺漏 class 的框
          4. 有新偵測 → 加入審閱佇列
        回傳 queue 及統計資訊
        """
        all_ids = set(self.class_names.keys())
        queue   = []
        stat_complete       = 0  # 所有 class 都已有
        stat_missing_no_det = 0  # 有缺漏但 YOLO 沒偵測到
        stat_need_review    = 0  # 有新偵測，需要審閱

        print(f"\n分析 {len(files)} 張圖片（類別: "
              f"{[self.class_names[i] for i in sorted(all_ids)]}）\n")

        for i, fname in enumerate(files, 1):
            ip = os.path.join(folder, fname)

            # 讀現有標註
            existing = self._load_existing(ip)
            present  = {b['class_id'] for b in existing}
            missing  = all_ids - present

            if not missing:
                stat_complete += 1
            else:
                # 有缺漏 class → 推論
                all_new = self._infer(ip)
                # 只保留缺漏 class 的框
                new_filtered = [b for b in all_new if b['class_id'] in missing]

                if new_filtered:
                    queue.append({
                        'image_path':     ip,
                        'existing_boxes': existing,        # locked=True
                        'new_boxes':      new_filtered,    # locked=False
                        'missing_classes': missing,
                    })
                    stat_need_review += 1
                else:
                    stat_missing_no_det += 1

            # 進度
            if i % 10 == 0 or i == len(files):
                print(f"  進度: {i:>4}/{len(files)}  "
                      f"（待審閱: {stat_need_review}  缺漏但無偵測: {stat_missing_no_det}  "
                      f"已齊全: {stat_complete}）")

        print(f"\n{'='*60}")
        print(f"分析完成")
        print(f"  ✅ 所有 class 已齊全（跳過）    : {stat_complete} 張")
        print(f"  🔍 有缺漏、有新偵測（待審閱）   : {stat_need_review} 張")
        print(f"  ⚠️  有缺漏、但 YOLO 無偵測       : {stat_missing_no_det} 張")
        print(f"{'='*60}")

        return queue

    # ── 單張審閱 ─────────────────────────────────────────────

    def _review_one(self, item, image_idx, total):
        """
        顯示審閱視窗。
        回傳 (action, boxes_to_save | None)
          action: 'save_next' | 'save_prev' | 'skip_next' | 'quit'
        """
        self.current_image = cv2.imread(item['image_path'])
        if self.current_image is None:
            print(f"❌ 無法載入: {item['image_path']}")
            return 'save_next', None

        self._setup_display(self.current_image)

        # 組合：現有（灰/鎖定） + 新偵測（彩色/可編輯）
        self.review_boxes        = item['existing_boxes'][:] + item['new_boxes'][:]
        self.review_selected_idx = -1
        self.review_dragging     = False
        self.review_drag_edge    = None
        self.review_hover_edge   = None

        # 視窗名稱必須用純 ASCII，Qt on Linux 對非 ASCII 名稱的 setMouseCallback 會報 NULL pointer
        wname = "AutoLabel_Review"
        cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(wname, self.display_size[0], self.display_size[1])
        cv2.setMouseCallback(wname, self._mouse)
        cv2.setWindowTitle(wname, f"Review [{image_idx+1}/{total}] - {os.path.basename(item['image_path'])}") 

        missing_names = [self.class_names.get(c, f"cls{c}") for c in sorted(item['missing_classes'])]
        print(f"\n  [{image_idx+1}/{total}] {os.path.basename(item['image_path'])}")
        print(f"    缺漏 class : {', '.join(missing_names)}")
        print(f"    新偵測框   : {len(item['new_boxes'])}  |  現有鎖定框: {len(item['existing_boxes'])}")

        def _collect_save_boxes():
            """現有鎖定框（原封不動）+ 使用者核准的新框"""
            existing_lines = item['existing_boxes']   # 永遠保留
            new_approved   = [b for b in self.review_boxes if not b.get('locked')]
            return existing_lines + new_approved

        while True:
            frame = self._draw_review(item, image_idx, total)
            cv2.imshow(wname, frame)
            key = cv2.waitKey(10) & 0xFF

            if key in (ord('d'), 13):       # D / Enter → 儲存並下一張
                cv2.destroyWindow(wname)
                return 'save_next', _collect_save_boxes()

            elif key == ord('a'):           # A → 儲存並上一張
                cv2.destroyWindow(wname)
                return 'save_prev', _collect_save_boxes()

            elif key == ord('x'):           # X → 跳過，不動 .txt
                cv2.destroyWindow(wname)
                print("    ⏭  跳過（.txt 不更動）")
                return 'skip_next', None

            elif key == ord('q'):           # Q → 退出
                cv2.destroyWindow(wname)
                return 'quit', None

            elif key in (8, 46, 127):       # Backspace / . / Delete → 刪除選取框
                if 0 <= self.review_selected_idx < len(self.review_boxes):
                    b = self.review_boxes[self.review_selected_idx]
                    if b.get('locked'):
                        print("    ⚠️  現有標註（🔒）無法刪除")
                    else:
                        cname = self.class_names.get(b['class_id'], f"cls{b['class_id']}")
                        del self.review_boxes[self.review_selected_idx]
                        self.review_selected_idx = -1
                        self.review_hover_edge   = None
                        remaining = sum(1 for rb in self.review_boxes if not rb.get('locked'))
                        print(f"    🗑  刪除 [{cname}]（剩 {remaining} 個新框）")
                else:
                    print("    ⚠️  請先點擊選取一個框")

    # ── 主入口 ───────────────────────────────────────────────

    def run(self):
        print("\n" + "="*60)
        print("YOLO 補標工具 — 自動偵測缺漏 Class")
        print("="*60)

        # 1. 選擇權重
        print("\n步驟 1：選擇 YOLO 權重檔")
        weights = self._pick_weights()
        if not weights: print("❌ 取消"); return
        print(f"✓ 載入: {os.path.basename(weights)}")
        self.model       = YOLO(weights)
        self.class_names = self.model.names
        print(f"  模型類別: { {k:v for k,v in self.class_names.items()} }")

        # 2. 選擇資料夾
        print("\n步驟 2：選擇圖片資料夾")
        folder = self._pick_folder()
        if not folder: print("❌ 取消"); return
        print(f"✓ 資料夾: {folder}")

        # 3. 設定信心度
        conf = self._ask_confidence()
        if conf is None: print("❌ 取消"); return
        self.conf_threshold = conf
        print(f"✓ 信心度門檻: {conf}")

        # 4. 收集圖片
        exts  = ('.jpg','.jpeg','.png','.bmp')
        files = sorted([f for f in os.listdir(folder)
                        if f.lower().endswith(exts) and
                        os.path.isfile(os.path.join(folder, f))])
        if not files: print("❌ 資料夾中沒有圖片"); return
        print(f"✓ 找到 {len(files)} 張圖片")

        # 5. 建立審閱佇列（分析 + 推論）
        queue = self._build_review_queue(files, folder)

        if not queue:
            print("\n✅ 所有圖片的標註都已齊全，無需補標！")
            return

        print(f"\n共 {len(queue)} 張圖片需要複判。")
        print("操作說明：")
        print("  • 點擊框 = 選取（彩色框可調整/刪除，灰色🔒鎖定）")
        print("  • 拖曳邊緣 = 調整大小")
        print("  • Del / Backspace = 刪除選取的新偵測框")
        print("  • D 或 Enter = 儲存並下一張")
        print("  • A = 儲存並上一張")
        print("  • X = 跳過（.txt 不更動）")
        print("  • Q = 退出")
        input("\n按 Enter 開始審閱...\n")

        # 6. 逐張審閱
        idx = 0; saved = 0; skipped = 0

        while 0 <= idx < len(queue):
            item = queue[idx]
            action, final_boxes = self._review_one(item, idx, len(queue))

            if action in ('save_next', 'save_prev', 'quit'):
                if final_boxes is not None:
                    lp = self._save(item['image_path'], final_boxes)
                    saved += 1
                    new_added = sum(1 for b in final_boxes if not b.get('locked'))
                    print(f"    ✓ 儲存（新增 {new_added} 框，共 {len(final_boxes)} 框）"
                          f" → {os.path.basename(lp)}")
                if   action == 'save_next': idx += 1
                elif action == 'save_prev': idx = max(0, idx-1)
                elif action == 'quit':      break

            elif action == 'skip_next':
                skipped += 1; idx += 1

        print(f"\n{'='*60}")
        print(f"✓ 補標完成")
        print(f"  已儲存  : {saved} 張（.txt 已更新）")
        print(f"  已跳過  : {skipped} 張（.txt 未動）")
        print(f"  未審閱  : {len(queue)-saved-skipped} 張（提早退出）")
        print(f"  標註位置: {os.path.join(folder,'labels')}")
        print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────
# YOLOLabelTool（SAM2 手動標註，不變）
# ─────────────────────────────────────────────────────────────
class YOLOLabelTool:
    def __init__(self, model_path="sam2_b.pt", display_size=(1280, 720)):
        self.model = SAM(model_path)
        self.classes = []; self.current_image = None; self.current_image_path = None
        self.display_image = None; self.display_size = display_size; self.point = None
        self.original_height = 0; self.original_width = 0
        self.display_height  = 0; self.display_width  = 0
        self.scale = 1.0; self.offset_x = 0; self.offset_y = 0
        self.selected_class = None; self.class_window_active = False
        self.class_window_position = None; self.current_labels = []
        self.adjusting_bbox = False; self.bbox_edge = None; self.temp_bbox = None

    def calculate_resize_params(self, w, h):
        tw,th = self.display_size; s = min(tw/w, th/h)
        nw,nh = int(w*s),int(h*s)
        return s, nw, nh, (tw-nw)//2, (th-nh)//2

    def resize_image_for_display(self, image):
        h,w = image.shape[:2]
        s,nw,nh,ox,oy = self.calculate_resize_params(w,h)
        self.scale,self.offset_x,self.offset_y = s,ox,oy
        self.display_width,self.display_height = nw,nh
        resized = cv2.resize(image,(nw,nh),interpolation=cv2.INTER_LINEAR)
        d = np.zeros((self.display_size[1],self.display_size[0],3),dtype=np.uint8)
        d[oy:oy+nh,ox:ox+nw] = resized; return d

    def display_to_original_coords(self,dx,dy):
        x,y = dx-self.offset_x, dy-self.offset_y
        if x<0 or y<0 or x>=self.display_width or y>=self.display_height: return None
        return [max(0,min(int(x/self.scale),self.original_width-1)),
                max(0,min(int(y/self.scale),self.original_height-1))]

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            orig = self.display_to_original_coords(x,y)
            if orig:
                self.point = orig; self.update_display_with_labels()
                cv2.circle(self.display_image,(x,y),5,(0,255,0),-1)
                cv2.putText(self.display_image,f"({orig[0]},{orig[1]}) - Press 's'",(10,30),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
                cv2.imshow("Image",self.display_image)

    def bbox_adjust_mouse_callback(self,event,x,y,flags,param):
        if self.temp_bbox is None: return
        orig = self.display_to_original_coords(x,y)
        if orig is None: return
        ox,oy=orig; x1,y1,x2,y2=self.temp_bbox; T=10
        if event==cv2.EVENT_MOUSEMOVE:
            if self.adjusting_bbox and self.bbox_edge:
                if   self.bbox_edge=='left':   self.temp_bbox[0]=max(0,min(ox,x2-10))
                elif self.bbox_edge=='right':  self.temp_bbox[2]=min(self.original_width,max(ox,x1+10))
                elif self.bbox_edge=='top':    self.temp_bbox[1]=max(0,min(oy,y2-10))
                elif self.bbox_edge=='bottom': self.temp_bbox[3]=min(self.original_height,max(oy,y1+10))
            else:
                if   abs(ox-x1)<T and y1<=oy<=y2: self.bbox_edge='left'
                elif abs(ox-x2)<T and y1<=oy<=y2: self.bbox_edge='right'
                elif abs(oy-y1)<T and x1<=ox<=x2: self.bbox_edge='top'
                elif abs(oy-y2)<T and x1<=ox<=x2: self.bbox_edge='bottom'
                else: self.bbox_edge=None
        elif event==cv2.EVENT_LBUTTONDOWN:
            if self.bbox_edge: self.adjusting_bbox=True
        elif event==cv2.EVENT_LBUTTONUP: self.adjusting_bbox=False

    def load_existing_labels(self,image_path):
        base=os.path.splitext(os.path.basename(image_path))[0]
        lp=os.path.join(os.path.dirname(image_path),"labels",f"{base}.txt")
        labels=[]
        if os.path.exists(lp):
            with open(lp) as f:
                for line in f:
                    p=line.strip().split()
                    if len(p)==5:
                        labels.append({'class_id':int(p[0]),'center_x':float(p[1]),
                                       'center_y':float(p[2]),'width':float(p[3]),'height':float(p[4])})
            print(f"✓ 載入 {len(labels)} 個現有標註")
        return labels

    def draw_label_on_image(self,image,label,color=(0,255,255),thickness=2):
        cx=label['center_x']*self.original_width; cy=label['center_y']*self.original_height
        w=label['width']*self.original_width;      h=label['height']*self.original_height
        x1,y1,x2,y2=int(cx-w/2),int(cy-h/2),int(cx+w/2),int(cy+h/2)
        cv2.rectangle(image,(x1,y1),(x2,y2),color,thickness)
        if label['class_id']<len(self.classes):
            name=self.classes[label['class_id']]
            sz,_=cv2.getTextSize(name,cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
            cv2.rectangle(image,(x1,y1-sz[1]-8),(x1+sz[0]+8,y1),color,-1)
            cv2.putText(image,name,(x1+4,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)
        return image

    def draw_adjustable_bbox(self,image,bbox,color=(0,255,0),highlight_edge=None):
        x1,y1,x2,y2=bbox; cv2.rectangle(image,(x1,y1),(x2,y2),color,3)
        for edge,pt1,pt2 in [('left',(x1,y1),(x1,y2)),('right',(x2,y1),(x2,y2)),
                              ('top',(x1,y1),(x2,y1)),('bottom',(x1,y2),(x2,y2))]:
            c=(255,255,0) if highlight_edge==edge else (255,0,0)
            w=8 if highlight_edge==edge else 5
            cv2.line(image,pt1,pt2,c,w)
        for pt in [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]:
            cv2.circle(image,pt,10,(0,255,255),-1)
        return image

    def update_display_with_labels(self):
        d=self.current_image.copy()
        for lbl in self.current_labels: d=self.draw_label_on_image(d,lbl,(0,255,255),2)
        self.display_image=self.resize_image_for_display(d)

    def load_classes(self,f="classes.txt"):
        if os.path.exists(f):
            with open(f,'r',encoding='utf-8') as fp:
                self.classes=[l.strip() for l in fp if l.strip()]
            print(f"已載入類別: {self.classes}")
        else: print("未找到 classes.txt，將在標註時新增類別")

    def save_classes(self,f="classes.txt"):
        with open(f,'w',encoding='utf-8') as fp:
            [fp.write(f"{c}\n") for c in self.classes]

    def class_selection_mouse_callback(self,event,x,y,flags,param):
        if event==cv2.EVENT_LBUTTONDOWN and self.class_window_active:
            bh,bw,sy,sp,sx=40,400,50,10,50
            for idx in range(len(self.classes)+1):
                by=sy+idx*(bh+sp)
                if sx<=x<=sx+bw and by<=y<=by+bh:
                    self.selected_class=(idx,self.classes[idx]) if idx<len(self.classes) else self.add_new_class_dialog()
                    self.class_window_active=False; break

    def add_new_class_dialog(self):
        root=tk.Tk(); root.withdraw()
        nc=simpledialog.askstring("新增類別","請輸入新類別名稱:",parent=root)
        root.destroy()
        if nc and nc.strip():
            nc=nc.strip(); self.classes.append(nc); self.save_classes()
            return (len(self.classes)-1,nc)
        return None

    def show_class_selection_window(self):
        self.selected_class=None; self.class_window_active=True
        wn="Select Class"; cv2.namedWindow(wn)
        cv2.setMouseCallback(wn,self.class_selection_mouse_callback)
        if self.class_window_position: cv2.moveWindow(wn,*self.class_window_position)
        bh,bw,sy,sp,sx=40,400,50,10,50
        nb=len(self.classes)+1; wh=sy+nb*(bh+sp)+50; ww=bw+2*sx
        while self.class_window_active:
            canvas=np.ones((wh,ww,3),dtype=np.uint8)*255
            cv2.putText(canvas,"Select Object Class",(sx,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2)
            for idx,cls in enumerate(self.classes):
                by=sy+idx*(bh+sp)
                cv2.rectangle(canvas,(sx,by),(sx+bw,by+bh),(100,200,100),-1)
                cv2.rectangle(canvas,(sx,by),(sx+bw,by+bh),(0,100,0),2)
                cv2.putText(canvas,f"{idx}: {cls}",(sx+10,by+27),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)
            ny=sy+len(self.classes)*(bh+sp)
            cv2.rectangle(canvas,(sx,ny),(sx+bw,ny+bh),(100,100,200),-1)
            cv2.rectangle(canvas,(sx,ny),(sx+bw,ny+bh),(0,0,100),2)
            cv2.putText(canvas,"+ Add New Class",(sx+10,ny+27),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)
            cv2.putText(canvas,"Click to select or ESC to cancel",
                        (sx,ny+bh+30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(100,100,100),1)
            cv2.imshow(wn,canvas)
            if cv2.waitKey(50)&0xFF==27:
                self.class_window_active=False; self.selected_class=None; break
        cv2.destroyWindow(wn); return self.selected_class

    def segment_with_sam2(self,image_path,point):
        results=self.model(image_path,points=[point],labels=[1])
        if results and results[0].masks is not None:
            return results[0].masks.data[0].cpu().numpy()
        return None

    def mask_to_bbox(self,mask):
        contours,_=cv2.findContours((mask*255).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        x,y,w,h=cv2.boundingRect(max(contours,key=cv2.contourArea))
        return [x,y,x+w,y+h]

    def bbox_to_yolo_format(self,bbox,class_id):
        x1,y1,x2,y2=bbox
        return {'class_id':class_id,
                'center_x':((x1+x2)/2)/self.original_width,
                'center_y':((y1+y2)/2)/self.original_height,
                'width':(x2-x1)/self.original_width,
                'height':(y2-y1)/self.original_height}

    def save_yolo_labels(self,image_path,labels):
        base=os.path.splitext(os.path.basename(image_path))[0]
        ld=os.path.join(os.path.dirname(image_path),"labels")
        os.makedirs(ld,exist_ok=True)
        lp=os.path.join(ld,f"{base}.txt")
        with open(lp,'w',encoding='utf-8') as f:
            for lbl in labels:
                f.write(f"{lbl['class_id']} {lbl['center_x']:.6f} {lbl['center_y']:.6f}"
                        f" {lbl['width']:.6f} {lbl['height']:.6f}\n")
        print(f"✓ 儲存: {lp}")

    def show_result_window(self,bbox,class_id,class_name):
        wn="Result - Adjust BBox"; cv2.namedWindow(wn)
        self.temp_bbox=list(bbox); self.adjusting_bbox=False; self.bbox_edge=None
        cv2.setMouseCallback(wn,self.bbox_adjust_mouse_callback)
        print(f"✓ SAM2: {class_name} | 拖曳邊緣調整 | C=接受  X=取消")
        while True:
            ri=self.current_image.copy()
            ri=self.draw_adjustable_bbox(ri,self.temp_bbox,(0,255,0),self.bbox_edge)
            ov=ri.copy(); cv2.rectangle(ov,(10,10),(600,130),(0,0,0),-1)
            cv2.addWeighted(ov,0.7,ri,0.3,0,ri)
            for txt,y,col in [(f"Class: {class_name}",40,(0,255,0)),
                               ("Drag edges to adjust",70,(255,255,255)),
                               ("C - Accept & Save",95,(0,255,0)),
                               ("X - Cancel",120,(0,0,255))]:
                cv2.putText(ri,txt,(20,y),cv2.FONT_HERSHEY_SIMPLEX,0.7 if y==40 else 0.6,col,2)
            w=self.temp_bbox[2]-self.temp_bbox[0]; h=self.temp_bbox[3]-self.temp_bbox[1]
            cv2.putText(ri,f"{w}x{h}",(self.temp_bbox[0],self.temp_bbox[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            cv2.imshow(wn,self.resize_image_for_display(ri))
            key=cv2.waitKey(10)&0xFF
            if key==ord('c'):
                cv2.destroyWindow(wn)
                return 'accept', self.bbox_to_yolo_format(self.temp_bbox,class_id)
            elif key in (ord('x'),27):
                cv2.destroyWindow(wn); return 'cancel',None

    def process_image(self,image_path,current_idx,total_images):
        self.current_image_path=image_path
        self.current_image=cv2.imread(image_path)
        if self.current_image is None: print(f"❌ 無法載入: {image_path}"); return 'error'
        self.original_height,self.original_width=self.current_image.shape[:2]
        self.current_labels=self.load_existing_labels(image_path)
        self.update_display_with_labels()
        wn="Image"; cv2.namedWindow(wn)
        cv2.setWindowTitle(wn,f"[{current_idx+1}/{total_images}] {os.path.basename(image_path)}")
        cv2.setMouseCallback(wn,self.mouse_callback)
        print(f"\n[{current_idx+1}/{total_images}] {os.path.basename(image_path)}"
              f"  {self.original_width}x{self.original_height}  標註:{len(self.current_labels)}")
        print("  點擊→s標註  r重置  Backspace刪除  a上一  d下一  q退出")
        while True:
            cv2.imshow(wn,self.display_image)
            key=cv2.waitKey(1)&0xFF
            if   key==ord('q'): cv2.destroyAllWindows(); return 'quit'
            elif key==ord('d'): cv2.destroyAllWindows(); return 'next'
            elif key==ord('a'): cv2.destroyAllWindows(); return 'prev'
            elif key==ord('r'): self.point=None; self.update_display_with_labels()
            elif key==8:
                if self.current_labels:
                    d=self.current_labels.pop(); self.save_yolo_labels(image_path,self.current_labels)
                    self.update_display_with_labels()
                    cn=self.classes[d['class_id']] if d['class_id']<len(self.classes) else f"cls{d['class_id']}"
                    print(f"🗑  刪除: {cn}（剩{len(self.current_labels)}）")
                else: print("⚠️  無標註可刪除")
            elif key==ord('s') and self.point:
                result=self.show_class_selection_window()
                if result:
                    class_id,class_name=result
                    mask=self.segment_with_sam2(image_path,self.point)
                    if mask is not None:
                        bbox=self.mask_to_bbox(mask)
                        if bbox:
                            choice,ld=self.show_result_window(bbox,class_id,class_name)
                            if choice=='accept' and ld:
                                self.current_labels.append(ld)
                                self.save_yolo_labels(image_path,self.current_labels)
                                self.update_display_with_labels()
                        else: print("❌ 轉換失敗")
                    else: print("❌ 分割失敗")
                self.point=None; self.update_display_with_labels()
        cv2.destroyAllWindows(); return 'next'


# ─────────────────────────────────────────────────────────────
# 起始索引
# ─────────────────────────────────────────────────────────────
def get_start_index(total):
    root=tk.Tk(); root.title("選擇起始照片"); root.geometry("350x150")
    root.update_idletasks()
    root.geometry(f'350x150+{root.winfo_screenwidth()//2-175}+{root.winfo_screenheight()//2-75}')
    tk.Label(root,text=f"總共有 {total} 張照片",font=("Arial",12)).pack(pady=10)
    tk.Label(root,text="請輸入起始照片編號 (1-based):",font=("Arial",10)).pack(pady=5)
    f=tk.Frame(root); f.pack(pady=5)
    entry=tk.Entry(f,width=15,font=("Arial",12)); entry.insert(0,"1"); entry.pack()
    result={'index':0,'confirmed':False}
    def ok():
        try:
            idx=int(entry.get())
            if 1<=idx<=total: result.update({'index':idx-1,'confirmed':True}); root.destroy()
            else: messagebox.showerror("錯誤",f"請輸入 1~{total}")
        except: messagebox.showerror("錯誤","請輸入有效數字")
    bf=tk.Frame(root); bf.pack(pady=10)
    tk.Button(bf,text="確認",command=ok,width=10,bg="#4CAF50",fg="white").pack(side=tk.LEFT,padx=5)
    tk.Button(bf,text="取消",command=root.destroy,width=10,bg="#f44336",fg="white").pack(side=tk.LEFT,padx=5)
    entry.bind('<Return>',lambda e:ok()); entry.focus()
    root.mainloop(); return result


# ─────────────────────────────────────────────────────────────
# 主選單
# ─────────────────────────────────────────────────────────────
def show_main_menu():
    print("\n" + "="*60)
    print("YOLO 標註工具集")
    print("="*60)
    print("  1. SAM2 輔助標註（手動點擊）")
    print("  2. 資料集分割（Train / Val / Test）")
    print("  3. YOLO 補標（偵測缺漏 Class + 逐張複判）")
    print("  4. 退出")
    print("="*60)
    return input("請輸入選項 (1-4): ").strip()


def main():
    while True:
        choice = show_main_menu()

        if choice == '1':
            tool = YOLOLabelTool(model_path="sam2_b.pt", display_size=(1280,720))
            tool.load_classes("classes.txt")
            root=tk.Tk(); root.withdraw()
            folder=filedialog.askdirectory(title="選擇包含圖片的資料夾")
            root.destroy()
            if not folder: print("❌ 未選擇資料夾"); continue
            files=[f for f in sorted(os.listdir(folder))
                   if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))]
            if not files: print("❌ 沒有圖片"); continue
            sr=get_start_index(len(files))
            if not sr['confirmed']: continue
            idx=sr['index']
            while 0<=idx<len(files):
                r=tool.process_image(os.path.join(folder,files[idx]),idx,len(files))
                if   r=='quit': print("✓ 退出"); break
                elif r=='next': idx+=1
                elif r=='prev': idx=max(0,idx-1)
                elif r=='error': idx+=1

        elif choice == '2':
            DatasetSplitter().interactive_split()

        elif choice == '3':
            YOLOAutoLabeler().run()

        elif choice == '4':
            print("\n👋 再見！"); break

        else:
            print("\n❌ 無效選項")


if __name__ == "__main__":
    main()