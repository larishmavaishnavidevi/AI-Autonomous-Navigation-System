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

# --- THE MUSCLES: CONTROL MODULE ---
def calculate_controls(path):
    """Translates A* coordinates into Steering and Throttle commands."""
    if len(path) < 2:
        return "STOP", "0%" # Reached the goal or path blocked

    current_pos = path[0]
    next_pos = path[1]

    # Calculate the change in X and Y
    dx = next_pos[0] - current_pos[0]
    dy = next_pos[1] - current_pos[1]

    # Calculate steering angle using Arctangent
    angle = math.degrees(math.atan2(dy, dx))
    
    # Translate math into English driving commands
    if angle == 0:
        steering = "HARD RIGHT"
    elif angle == 45:
        steering = "SLIGHT RIGHT"
    elif angle == 90:
        steering = "STRAIGHT AHEAD"
    elif angle == 135:
        steering = "SLIGHT LEFT"
    elif angle == 180:
        steering = "HARD LEFT"
    else:
        steering = f"TURN {int(angle)}°"

    # If the path is clear and straight, throttle is 100%. If turning, slow down to 50%.
    throttle = "100% (Cruising)" if angle == 90 else "50% (Cornering)"

    return steering, throttle

def pixel_to_grid(x, y):
    grid_x = int((x / FRAME_WIDTH) * GRID_SIZE)
    grid_y = int((y / FRAME_HEIGHT) * GRID_SIZE)
    return max(0, min(GRID_SIZE - 1, grid_x)), max(0, min(GRID_SIZE - 1, grid_y))

def draw_dashboard(frame, steering, throttle, path_status):
    """Draws a professional self-driving HUD on the camera feed."""
    # Draw a black background bar at the top
    cv2.rectangle(frame, (0, 0), (FRAME_WIDTH, 80), (0, 0, 0), -1)