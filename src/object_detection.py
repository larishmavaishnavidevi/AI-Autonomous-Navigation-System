import cv2
import numpy as np
import os

def run_object_detection():
    print("Loading YOLOv3 Model...")
    
    # Paths to your downloaded files (ensure these are in the 'models' folder)
    weights_path = os.path.join("..", "models", "yolov3.weights")
    config_path = os.path.join("..", "models", "yolov3.cfg")
    labels_path = os.path.join("..", "models", "coco.names")
    
    # Check if files exist before crashing
    if not os.path.exists(weights_path) or not os.path.exists(config_path):
        print("Error: YOLO model files not found in the 'models' directory.")
        print("Please download yolov3.weights and yolov3.cfg.")
        return

    # Load the YOLO network
    net = cv2.dnn.readNet(weights_path, config_path)
    
    # Load the class labels (e.g., 'person', 'car')
    with open(labels_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
        
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    print("Starting camera... Press 'q' to exit.")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, channels = frame.shape

        # Preprocess the frame for the neural network
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Process the network's outputs
        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # If the AI is more than 50% sure it found something
                if confidence > 0.5:
                    # Calculate bounding box coordinates
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply Non-Max Suppression to remove overlapping boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Draw the bounding boxes on the screen
        font = cv2.FONT_HERSHEY_PLAIN
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence_text = f"{round(confidences[i]*100, 2)}%"
                
                # Draw a green box and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence_text}", (x, y - 5), font, 2, (0, 255, 0), 2)

        # Show the output
        cv2.imshow("Object Detection - Autonomous Nav", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_object_detection()