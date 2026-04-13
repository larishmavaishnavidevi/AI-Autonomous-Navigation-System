import numpy as np
import matplotlib.pyplot as plt
import heapq

# Import the grid we created in the previous file
from env_setup import create_environment

def heuristic(a, b):
    """Calculates the Manhattan distance between two points."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid, start, goal):
    """A* Pathfinding algorithm to find the shortest path."""
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)] 
    
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []
    
    heapq.heappush(oheap, (fscore[start], start))
    
    while oheap:
        current = heapq.heappop(oheap)[1]
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
            
        close_set.add(current)
        
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            
            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
                if grid[neighbor[0]][neighbor[1]] == 1:
                    continue
            else:
                continue
                
            if neighbor in close_set and gscore.get(current, 0) + 1 >= gscore.get(neighbor, 0):
                continue
                
            tentative_g_score = gscore.get(current, 0) + 1
            
            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
                
    return False

def visualize_path(grid, start, goal, path):
    """Plots the grid and the calculated optimal path."""
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap='Greys', origin='upper')
    
    if path:
        path_y = [p[0] for p in path]
        path_x = [p[1] for p in path]
        plt.plot(path_x, path_y, color='blue', linewidth=3, label='A* Path')
    
    plt.scatter(start[1], start[0], marker='s', color='green', s=100, label='Start')
    plt.scatter(goal[1], goal[0], marker='s', color='red', s=100, label='Goal')
    
    plt.grid(True, which='both', color='lightgrey', linewidth=0.5)
    plt.xticks(np.arange(-0.5, grid.shape[1], 1), [])
    plt.yticks(np.arange(-0.5, grid.shape[0], 1), [])
    plt.title("AI Autonomous Navigation - A* Path Planning")
    plt.legend(loc="upper right")
    plt.show()

if __name__ == "__main__":
    start_pos = (2, 2)
    goal_pos = (18, 17)
    
    print("Loading environment...")
    env_grid = create_environment()
    
    print("Calculating optimal path using A*...")
    optimal_path = astar(env_grid, start_pos, goal_pos)
    
    if optimal_path:
        print(f"Success! Path found in {len(optimal_path)} steps.")
        visualize_path(env_grid, start_pos, goal_pos, optimal_path)
    else:
        print("Error: No path found! The goal is completely blocked.")