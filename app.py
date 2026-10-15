import pyautogui
import cv2
import numpy as np
from ultralytics import YOLO
import time

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05

model = YOLO('runs/detect/cookie_detector/weights/best.pt')
CLASS_NAMES = {0: "Закрытие рекламы", 1: "Покупка улучшения", 2: "Бонусы"}
def find_and_click():
    
    screenshot = pyautogui.screenshot()
    screenshot_np = np.array(screenshot)
    screenshot_np = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
    
    
    results = model(screenshot_np)
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            confidence = box.conf[0]

            if confidence > 0.7:
                class_id = int(box.cls[0])  
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                
                for i in range(10):
                    pyautogui.click(center_x, center_y)
                    time.sleep(0.1)
                action_name = CLASS_NAMES.get(class_id, "Неизвестный объект")
                print(f"Выполнено: {action_name} -> Клик по координатам: ({center_x}, {center_y})")
                return True

    return False

while True:
    if find_and_click():
        time.sleep(1)
    else:
        time.sleep(0.5)