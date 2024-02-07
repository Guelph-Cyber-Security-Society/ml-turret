import cv2
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

# Load a pre-trained model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Set the model to evaluation mode

# COCO dataset class names
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Define a function to predict and draw boxes
def predict_and_draw_boxes(frame):
    # Convert the image from BGR (OpenCV format) to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Transform the image to tensor
    image = F.to_tensor(rgb_frame)
    
    # Perform inference
    with torch.no_grad():
        prediction = model([image])
    
    # Draw boxes and labels on the frame
    for element in zip(prediction[0]['boxes'], prediction[0]['labels'], prediction[0]['scores']):
        boxes, label, score = element
        if score > 0.5:  # Filter out low confidence detections
            boxes = boxes.detach().numpy().astype(np.int32)
            label_text = f'{COCO_INSTANCE_CATEGORY_NAMES[label]} ({score:.2f})' if label < len(COCO_INSTANCE_CATEGORY_NAMES) else f'Unknown ({score:.2f})'
            cv2.rectangle(frame, (boxes[0], boxes[1]), (boxes[2], boxes[3]), (0, 255, 0), 2)
            cv2.putText(frame, label_text, (boxes[0], boxes[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

# Define a function to process video frames
def process_video(video_path, frame_skip=5):
    # Capture video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames to improve performance
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # Predict and draw boxes on the frame
        output_frame = predict_and_draw_boxes(frame)

        # Return the processed frame
        yield output_frame

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
