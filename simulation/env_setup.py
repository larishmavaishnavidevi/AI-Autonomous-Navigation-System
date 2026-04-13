import numpy as np
import matplotlib.pyplot as plt

def create_environment(grid_size=20):
    """Creates a 2D grid with obstacles."""
    # Create an empty grid filled with 0s (0 = empty space, 1 = obstacle)
    grid = np.zeros((grid_size, grid_size))
    
    # Add some digital obstacles (simulating walls or blocked paths)
    grid[5:15, 5] = 1  # Vertical wall
    grid[15, 5:15] = 1 # Horizontal wall
    grid[3:8, 12] = 1  # Another small wall
    
    return grid

def visualize_environment(grid, start, goal):
    """Visualizes the grid, start point, and goal point."""
    plt.figure(figsize=(8, 8))
    
    # Plot the grid using a grayscale colormap
    plt.imshow(grid, cmap='Greys', origin='upper')
    
    # Mark the start (Green) and goal (Red)
    plt.scatter(start[1], start[0], marker='s', color='green', s=100, label='Start')
    plt.scatter(goal[1], goal[0], marker='s', color='red', s=100, label='Goal')
    
    # Add grid lines
    plt.grid(True, which='both', color='lightgrey', linewidth=0.5)
    plt.xticks(np.arange(-0.5, grid.shape[1], 1), [])
    plt.yticks(np.arange(-0.5, grid.shape[0], 1), [])
    
    plt.title("AI Autonomous Navigation - Virtual Environment")
    plt.legend(loc="upper right")
    plt.show()

if __name__ == "__main__":
    # Define start and goal coordinates (row, column)
    start_pos = (2, 2)
    goal_pos = (18, 17)
    
    print("Generating simulation environment...")
    env_grid = create_environment()
    visualize_environment(env_grid, start_pos, goal_pos)