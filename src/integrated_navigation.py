import cv2
import numpy as np
import os

# --- VIRTUAL GRID SETTINGS ---
GRID_SIZE = 10  # A 10x10 navigation grid
FRAME_WIDTH = 640 # Default camera width
FRAME_HEIGHT = 480 # Default camera height

def pixel_to_grid(x, y):
    """Translates a camera pixel coordinate into an A* grid coordinate."""
    grid_x = int((x / FRAME_WIDTH) * GRID_SIZE)
    grid_y = int((y / FRAME_HEIGHT) * GRID_SIZE)
    
    # Ensure coordinates stay within the 0-9 range
    grid_x = max(0, min(GRID_SIZE - 1, grid_x))
    grid_y = max(0, min(GRID_SIZE - 1, grid_y))
    return grid_x, grid_y

def run_integrated_system():
    print("Loading YOLOv3 System and Virtual Grid...")
    
    weights_path = os.path.join("..", "models", "yolov3.weights")
    config_path = os.path.join("..", "models", "yolov3.cfg")
    labels_path = os.path.join("..", "models", "coco.names")
    
    net = cv2.dnn.readNet(weights_path, config_path)
    with open(labels_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
        
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    print("System Active. Press 'q' on the video window to exit.")
    cap = cv2.VideoCapture(0)
    
    # Force camera resolution for consistent grid mapping
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame so it acts like a mirror
        frame = cv2.flip(frame, 1)

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids, confidences, boxes = [], [], []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Using a higher confidence threshold (60%) to prevent ghost detections
                if confidence > 0.6:
                    center_x = int(detection[0] * FRAME_WIDTH)
                    center_y = int(detection[1] * FRAME_HEIGHT)
                    w = int(detection[2] * FRAME_WIDTH)
                    h = int(detection[3] * FRAME_HEIGHT)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                
                # --- INTEGRATION LOGIC ---
                # Find the center-bottom pixel of the object (where it touches the ground)
                ground_x = x + (w // 2)
                ground_y = y + h
                
                # Translate that pixel into an A* grid coordinate
                grid_x, grid_y = pixel_to_grid(ground_x, ground_y)
                
                # Print the translation to the terminal (This is where A* takes over!)
                print(f"ALERT: {label.upper()} detected! Updating A* Grid -> Obstacle placed at [{grid_x}, {grid_y}]")
                
                # Draw visual feedback on the camera
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, f"{label} Grid:[{grid_x},{grid_y}]", (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                
                # Draw a target on the ground coordinate
                cv2.circle(frame, (ground_x, ground_y), 5, (0, 255, 255), -1)

        cv2.imshow("Integrated Navigation System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_integrated_system()