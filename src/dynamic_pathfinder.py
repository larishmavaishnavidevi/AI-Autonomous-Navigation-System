import cv2
import numpy as np
import os
import heapq

# --- VIRTUAL GRID & PATH SETTINGS ---
GRID_SIZE = 10
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
START = (0, 0) # Top-left of the grid
GOAL = (9, 9)  # Bottom-right of the grid

# --- A* PATHFINDING LOGIC ---
class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0
    def __lt__(self, other):
        return self.f < other.f

def astar(grid, start, end):
    """Calculates the shortest path from start to end, avoiding obstacles."""
    open_list = []
    closed_set = set()
    start_node = Node(start)
    heapq.heappush(open_list, (0, start_node))

    while open_list:
        _, current_node = heapq.heappop(open_list)

        if current_node.position == end:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1] # Return reversed path

        closed_set.add(current_node.position)

        # Check all 8 surrounding directions
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]:
            node_pos = (current_node.position[0] + dx, current_node.position[1] + dy)

            # Ensure within grid bounds
            if 0 <= node_pos[0] < GRID_SIZE and 0 <= node_pos[1] < GRID_SIZE:
                if grid[node_pos[1]][node_pos[0]] == 1: # 1 means Obstacle
                    continue
                if node_pos in closed_set:
                    continue

                neighbor = Node(node_pos, current_node)
                neighbor.g = current_node.g + 1
                neighbor.h = abs(node_pos[0] - end[0]) + abs(node_pos[1] - end[1])
                neighbor.f = neighbor.g + neighbor.h
                heapq.heappush(open_list, (neighbor.f, neighbor))
    return [] # Return empty if no path exists

def pixel_to_grid(x, y):
    """Translates a camera pixel coordinate into an A* grid coordinate."""
    grid_x = int((x / FRAME_WIDTH) * GRID_SIZE)
    grid_y = int((y / FRAME_HEIGHT) * GRID_SIZE)
    return max(0, min(GRID_SIZE - 1, grid_x)), max(0, min(GRID_SIZE - 1, grid_y))

# --- MAIN SYSTEM LOOP ---
def run_dynamic_navigation():
    print("Loading YOLOv3 and A* Dynamic Path Planner...")
    
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

    last_path = [] # Tracks the previous path so we only print when it changes

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)

        # 1. Reset the map every single frame (0 = clear road)
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids, confidences, boxes = [], [], []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
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
                
                # 2. Find where the object touches the ground
                ground_x = x + (w // 2)
                ground_y = y + h
                grid_x, grid_y = pixel_to_grid(ground_x, ground_y)
                
                # 3. Mark the object as a solid wall (1) on our virtual map
                grid[grid_y][grid_x] = 1
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, f"Obstacle at [{grid_x},{grid_y}]", (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)

        # 4. DYNAMIC RECALCULATION: Tell A* to find a path using the newly updated map
        current_path = astar(grid, START, GOAL)
        
        # 5. Alert the terminal ONLY if the path was forced to change
        if current_path != last_path:
            if not current_path:
                print("\n⚠️ ALERT: PATH BLOCKED! No valid route to goal.")
            else:
                print(f"\n✅ REROUTING: New path calculated -> {current_path}")
            last_path = current_path

        cv2.imshow("Dynamic Path Planning System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_dynamic_navigation()