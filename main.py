from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")

video = cv2.VideoCapture("test_video.mp4")

while True:
    ret, frame = video.read()

    if not ret:
        break

    results = model(frame, conf=0.5)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()