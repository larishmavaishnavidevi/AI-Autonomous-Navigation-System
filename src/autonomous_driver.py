import cv2
import numpy as np
import os
import heapq
import math

# --- SYSTEM SETTINGS ---
GRID_SIZE = 10
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
START = (0, 0) # Simulated car position
GOAL = (9, 9)  # Simulated destination

class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g, self.h, self.f = 0, 0, 0
    def __lt__(self, other):
        return self.f < other.f

def astar(grid, start, end):
    open_list, closed_set = [], set()
    heapq.heappush(open_list, (0, Node(start)))

    while open_list:
        _, current = heapq.heappop(open_list)
        if current.position == end:
            path = []
            while current:
                path.append(current.position)
                current = current.parent
            return path[::-1]
        
        closed_set.add(current.position)
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]:
            nx, ny = current.position[0] + dx, current.position[1] + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and grid[ny][nx] == 0:
                if (nx, ny) not in closed_set:
                    neighbor = Node((nx, ny), current)
                    neighbor.g = current.g + 1
                    neighbor.h = abs(nx - end[0]) + abs(ny - end[1])
                    neighbor.f = neighbor.g + neighbor.h
                    heapq.heappush(open_list, (neighbor.f, neighbor))
    return []

def calculate_controls(path):
    if len(path) < 2:
        return "STOP", "0%"

    current_pos = path[0]
    next_pos = path[1]
    dx = next_pos[0] - current_pos[0]
    dy = next_pos[1] - current_pos[1]
    angle = math.degrees(math.atan2(dy, dx))
    
    if angle == 0: steering = "HARD RIGHT"
    elif angle == 45: steering = "SLIGHT RIGHT"
    elif angle == 90: steering = "STRAIGHT AHEAD"
    elif angle == 135: steering = "SLIGHT LEFT"
    elif angle == 180: steering = "HARD LEFT"
    else: steering = f"TURN {int(angle)}°"

    throttle = "100% (Cruising)" if angle == 90 else "50% (Cornering)"
    return steering, throttle

def pixel_to_grid(x, y):
    grid_x = int((x / FRAME_WIDTH) * GRID_SIZE)
    grid_y = int((y / FRAME_HEIGHT) * GRID_SIZE)
    return max(0, min(GRID_SIZE - 1, grid_x)), max(0, min(GRID_SIZE - 1, grid_y))

def draw_dashboard(frame, steering, throttle, path_status):
    # Background Bar
    cv2.rectangle(frame, (0, 0), (FRAME_WIDTH, 80), (0, 0, 0), -1)
    # Telemetry Text (Indented correctly inside the function)
    cv2.putText(frame, f"STATUS: {path_status}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"STEERING: {steering}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"THROTTLE: {throttle}", (350, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

def run_autonomous_driver():
    print("Initializing Student Project: Autonomous Stack...")
    
    weights_path = os.path.join("..", "models", "yolov3.weights")
    config_path = os.path.join("..", "models", "yolov3.cfg")
    labels_path = os.path.join("..", "models", "coco.names")
    
    net = cv2.dnn.readNet(weights_path, config_path)
    with open(labels_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
        
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        boxes, confidences, class_ids = [], [], []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.6:
                    center_x, center_y = int(detection[0] * FRAME_WIDTH), int(detection[1] * FRAME_HEIGHT)
                    w, h = int(detection[2] * FRAME_WIDTH), int(detection[3] * FRAME_HEIGHT)
                    x, y = int(center_x - w / 2), int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                gx, gy = pixel_to_grid(x + (w // 2), y + h)
                grid[gy][gx] = 1 
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        current_path = astar(grid, START, GOAL)
        status = "AUTOPILOT ENGAGED" if current_path else "PATH BLOCKED"
        steer, throt = calculate_controls(current_path)
        draw_dashboard(frame, steer, throt, status)

        cv2.imshow("Autonomous Driver - Student Project", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_autonomous_driver()