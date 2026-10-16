import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# Models load karo
print("Loading models... wait")
classifier = MobileNetV2(weights="imagenet")
detector = YOLO("yolov8n.pt") # pehli baar internet se download hoga

# Camera start
camera = cv2.VideoCapture(0)
mode = "mobilenet" # default mode

print("Press m = MobileNet, y = YOLO, q = Quit")

while camera.isOpened():
    ok, frame = camera.read()
    if not ok:
        break

    if mode == "yolo":
        # YOLO object detection
        results = detector(frame, verbose=False)
        frame = results[0].plot()
    else:
        # MobileNetV2 Image Classification
        # 1. Resize to 224x224
        resized = cv2.resize(frame, (224, 224))
        # 2. BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # 3. Preprocess
        img = preprocess_input(np.expand_dims(rgb.astype(float), 0))

        preds = classifier.predict(img, verbose=0)
        _, name, score = decode_predictions(preds, top=1)[0][0]

        cv2.putText(frame, f"{name} (score: {score:.0%})", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("m=MobileNet, y=YOLO, q=quit", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    # Mode switch
    mode = {"m": "mobilenet", "y": "yolo"}.get(chr(key), mode)

camera.release()
cv2.destroyAllWindows()