import os
import sys
import time
import cv2
import mss
import numpy as np
import threading
from pynput import keyboard
import ctypes
from multiprocessing import Process, freeze_support
from PyQt5 import QtCore, QtWidgets, QtGui
# Windows API for always-on-top
try:
    import win32gui
    import win32con
except ImportError:
    win32gui = None
    win32con = None


# ----- 配置 -----
TEMPLATE_NAMES = ['up', 'down', 'left', 'right']
TEMPLATE_THRESHOLD = 0.5
BASE_DIR = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(__file__)
TEMPLATE_PATH = os.path.join(BASE_DIR, '')
REGION_FILE = os.path.join(BASE_DIR, 'regions.txt')
TEST_IMAGE = os.path.join(BASE_DIR, 'test.png')

KEY_MAP = {'up': 'w', 'down': 's', 'left': 'a', 'right': 'd'}
ADD_REGION_HOTKEY = {keyboard.Key.ctrl_l, keyboard.Key.shift}
current_keys = set()
regions = []

ctrl_pressed = True

# 4K 基准
BASE_WIDTH = 3840
BASE_HEIGHT = 2160

#ROI
base_rois = [
    (220, 205, 350, 55),
    (220, 310, 350, 55),
    (220, 415, 350, 55),
    (220, 515, 350, 55),
    (220, 620, 350, 55),
    (220, 725, 350, 55),
    (220, 830, 350, 55),
    (220, 940, 350, 55),
]

# 转换为百分比格式
relative_rois = [
    (x / BASE_WIDTH, y / BASE_HEIGHT, w / BASE_WIDTH, h / BASE_HEIGHT)
    for x, y, w, h in base_rois
]

def load_regions():
    rs = base_rois
    return rs

def save_region(region):
    with open(REGION_FILE, 'a') as f:
        f.write(','.join(map(str, region)) + '\n')
    print(f"已保存新区域: {region}")

def select_region():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screenshot = np.array(sct.grab(monitor))
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
        roi = cv2.selectROI("选择区域", screenshot, False, False)
        cv2.destroyWindow("选择区域")
        return roi if roi[2] > 0 and roi[3] > 0 else None

def grab_gray(x, y, w, h):
    with mss.mss() as sct:
        img = np.array(sct.grab({"top": y, "left": x, "width": w, "height": h}))
    return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

def recognize_sequence_by_sliding_window(cleaned_img, step_ratio=0.85):
    """
    滑动窗口几何分析版
    :param cleaned_img: 预处理后的二值图像
    :param step_ratio: 步长与窗口宽度的比例（建议0.6-0.8）
    :return: 有序的箭头方向序列
    """
    h, w = cleaned_img.shape
    arrow_width = int(h * step_ratio)
    n_arrows =int(w // arrow_width)
    sequence = []
    x = 0

    for i in range(n_arrows):
        roi = cleaned_img[0:h, i*arrow_width:(i+1)*arrow_width]
        #cv2.imwrite(f"roi_{i}.png", roi)
        # 调用几何识别函数
        total_pixel = roi.size
        white_pixel = cv2.countNonZero(roi)
        #白色值小于一定程度则认为不存在箭头且序列结束
        if(total_pixel-white_pixel)/total_pixel < 0.5:
            break
        direction = recognize_img_by_geometry(roi, i)
        #只有成功识别的箭头才加入序列，如果任意一个识别失败则认为队列已经结束
        if direction is not None:
            sequence.append(direction)
        else:
            break
    return sequence

def recognize_img_by_geometry(squared_img, index):
    vis = cv2.cvtColor(squared_img, cv2.COLOR_GRAY2BGR)
    def vertex_angle(a, b, c):
        v1 = a - b
        v2 = c - b
        cosA = (v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        return np.arccos(np.clip(cosA, -1, 1))
    # Process contours
    cnts, _ = cv2.findContours(squared_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in cnts:
        # Approximate polygon
        peri = cv2.arcLength(cnt, True)
        pts = cv2.approxPolyDP(cnt, 0.02 * peri, True).reshape(-1, 2)

        # Compute angles and find right-angle vertices
        angles = [(vertex_angle(pts[i - 1], pts[i], pts[(i + 1) % len(pts)]), tuple(pts[i])) for i in range(len(pts))]
        right_pts = [p for ang, p in angles if abs(ang - np.pi / 2) < 0.15]
        # Compute base centroid (mean of right-angle points)
        right_pts_arr = np.array(right_pts)
        if len(right_pts_arr) == 0:
            print(f"[Warn] 未找到有效的直角点，跳过 index={index}"f"[Warn] 未找到有效的直角点，跳过 index={index}")
            return None
        bx, by = np.mean(right_pts_arr, axis=0).astype(int)
        # Compute shape centroid
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            print(f"[Warn] 面积为0的轮廓跳过，index={index}")
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        dx = cx - bx
        dy = cy - by
        #根据向量方向判断箭头方向
        # 判断方向
        if abs(dx) > abs(dy):
            direction = 'right' if dx > 0 else 'left'
        else:
            direction = 'down' if dy > 0 else 'up'
        # Draw contour
        cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 1)
        # Draw right-angle points
        #for p in right_pts:
        #    cv2.circle(vis, p, 6, (0, 255, 255), -1)
        # Draw base centroid
        #cv2.circle(vis, (bx, by), 6, (0, 165, 255), -1)
        # Draw shape centroid
        #cv2.circle(vis, (cx, cy), 6, (255, 0, 0), -1)

        # Draw direction arrow: vector from base centroid to shape centroid
        #dx, dy = cx - bx, cy - by
        #arrow_end = (bx + int(dx * 0.5), by + int(dy * 0.5))
        #cv2.arrowedLine(vis, (bx, by), arrow_end, (255, 0, 255), 2)
        #cv2.imwrite(f'roi_{index}.png', vis)
        return direction
    return None

def visualize_matches(image, sequence, debug_prefix):
    color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for idx, name in enumerate(sequence):
        cv2.putText(color, name, (10 + idx * 50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(f"{debug_prefix}_05_visual.png", color)

#识别按键序列
def recon_seq(gray_img, debug_prefix=""):
    #处理图片
    #cv2.imwrite(f"{debug_prefix}_01_original.png", gray_img)
    _, binary = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY)
    #cv2.imwrite(f"{debug_prefix}_02_binary.png", binary)
    #调用滑动窗口处理单个战备技能的roi，获得序列
    seq = recognize_sequence_by_sliding_window(binary)
    return seq

#根据传入序列模拟按键输入
def press_sequence(sequence):
    kb = keyboard.Controller()
    for d in sequence:
        #当检测到None认为输入已结束，退出循环
        if d is None:
            return None
        else:
            print(KEY_MAP[d])
            kb.press(KEY_MAP[d])
            time.sleep(0.05)
            kb.release(KEY_MAP[d])
            time.sleep(0.05)
    return None

def run_mode():
    print("[✓] 进入运行模式，按 Ctrl+1~6 触发对应识别区域。按 Esc 退出。")
    regions = load_regions()
    start_osd_process()
    if not regions:
        print("未配置任何识别区域，请先进入 setup 模式。")
        return

    pressed_keys = set()

    def on_press(key):
        pressed_keys.add(key)
        # 获取键的虚拟键码（vk）
        try:
            vk = key.vk
        except AttributeError:
            vk = None

        # 检查是否是数字键1-6（主键盘）
        if vk is not None and 49 <= vk <= 56:  # 1~8的vk码是49~56
            digit = vk - 48  # 转换为1~6
            # 检查Ctrl键是否被按住
            ctrl_pressed = any(k in pressed_keys for k in [
                keyboard.Key.ctrl,
                keyboard.Key.ctrl_l,
                keyboard.Key.ctrl_r
            ])
            if ctrl_pressed:
                #-1按键错位了因此多减一位
                idx = digit - 1
                if idx < len(regions):
                    #根据预先划分好的每个编号对应的区域截取当前这个区域的图像
                    x, y, w, h = regions[idx]
                    gray = grab_gray(x, y, w, h)

                    roi_filename = f"roi_run_{idx + 1}.png"
                    #cv2.imwrite(roi_filename, gray)
                    #print(f"[💾] 已保存当前识别区域截图为 {roi_filename}")
                    #调用识别函数
                    seq = recon_seq(gray)
                    if seq:
                        print(f"[→] 执行按键序列：{[' ' + d for d in seq]}")
                        press_sequence(seq)
                        return None
                    else:
                        print("[!] 未识别到箭头序列，跳过。")
                        return None
                return None
            return None
        return None

    def on_release(key):
        if key in pressed_keys:
            pressed_keys.remove(key)

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

def debug_mode():
    print("进入测试模式，读取 test.png 中的区域并尝试识别箭头。")
    if not os.path.exists(TEST_IMAGE):
        print("找不到 test.png 文件")
        return
    test_img = cv2.imread(TEST_IMAGE, cv2.IMREAD_GRAYSCALE)
    if test_img is None:
        print("加载 test.png 失败")
        return
    regions = load_regions()
    if not regions:
        print("未定义任何识别区域")
        return
    idx = int(input(f"区域编号 (1~{len(regions)}): ")) - 1
    if idx < 0 or idx >= len(regions):
        print("编号超出范围")
        return
    x, y, w, h = regions[idx]
    roi = test_img[y:y+h, x:x+w]
    #cv2.imwrite(f"roi_debug_{idx+1}.png", roi)
    seq = recon_seq(roi, debug_prefix=f"debug_region_{idx + 1}")
    print(f"[识别结果] 区域 {idx+1}: {seq}")

def on_press(key):
    current_keys.add(key)
    if ADD_REGION_HOTKEY <= current_keys:
        print("进入区域添加模式...")
        region = select_region()
        if region:
            regions.append(region)
            save_region(region)

def on_release(key):
    current_keys.discard(key)

def setup_mode():
    print("[⚙] 清除已有区域配置")
    if input("确认清除并重新配置？(y/n): ").strip().lower() != 'y':
        return
    with open(REGION_FILE, 'w') as f:
        pass
    while True:
        cmd = input(">>> ").strip().lower()
        if cmd == 'exit':
            break
        elif cmd == 'add':
            region = select_region()
            if region:
                save_region(region)

# ---------------- OSD Overlay ----------------
class OSDOverlay(QtWidgets.QWidget):
    def __init__(self, relative_rois, alpha=0.7):
        super().__init__()
        self.relative_rois = relative_rois
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint |
            QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.Tool
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)

        user32 = ctypes.windll.user32
        self.screen_w = user32.GetSystemMetrics(0)
        self.screen_h = user32.GetSystemMetrics(1)
        self.setGeometry(0, 0, self.screen_w, self.screen_h)
        self.alpha = alpha
        self.show()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setOpacity(self.alpha)
        pen = QtGui.QPen(QtGui.QColor(255, 255, 0), 2)
        painter.setPen(pen)
        font = QtGui.QFont('Arial', 20)
        painter.setFont(font)

        for idx, (rx, ry, rw, rh) in enumerate(self.relative_rois):
            x = int(rx * self.screen_w)
            y = int(ry * self.screen_h)
            w = int(rw * self.screen_w)
            h = int(rh * self.screen_h)
            painter.drawRect(x-4, y-4, w+8, h+8)
            painter.drawText(x + 375, y + int(h / 2)+10, str(idx + 1))

def osd_process_entry():
    app = QtWidgets.QApplication(sys.argv)
    overlay = OSDOverlay(relative_rois)
    sys.exit(app.exec_())

def start_osd_process():
    p = Process(target=osd_process_entry)
    p.daemon = True
    p.start()
    return p

def main():
    #mode = input("选择模式：run / debug / setup >>> ").strip().lower()
    #if mode == "run":
        run_mode()
    #elif mode == "debug":
    #    debug_mode()
    #elif mode == "setup":
    #    setup_modesetup_mode()
    #else:
    #    print("未知模式")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()