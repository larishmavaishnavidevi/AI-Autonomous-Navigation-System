# AI-Based Autonomous Navigation System

## 🚀 Project Overview
This project is an AI-powered autonomous navigation system that enables a virtual agent (simulating a robot or self-driving vehicle) to navigate from a start point to a destination without human intervention. [cite_start]It detects obstacles, calculates the optimal route using path-planning algorithms, and safely navigates a simulated environment[cite: 317, 318, 469, 470, 471].

## 💡 Problem Statement & Industry Relevance
[cite_start]**Problem:** Autonomous movement in dynamic environments requires systems to move intelligently without manual control[cite: 474, 475].
[cite_start]**Industry Relevance:** This core logic is actively used by companies like Tesla, Waymo, NVIDIA, and Open AI Robotics for self-driving cars, warehouse automation, delivery bots, and drones[cite: 319, 320, 475].

## 🛠️ Tech Stack
* [cite_start]**Language:** Python 3.x [cite: 345, 478]
* [cite_start]**Environment/Math:** NumPy, Matplotlib [cite: 478]
* [cite_start]**Path Planning:** A* (A-Star) Algorithm [cite: 486, 487]

## 📂 Folder Structure
```text
AI-Autonomous-Navigation-System/
│
├── simulation/            # Virtual environment and layout scripts
│   ├── env_setup.py       # Generates the 2D grid and obstacles
│   └── path_planning.py   # Executes the A* algorithm
├── outputs/               # Saved screenshots and results
├── src/                   # Core source code (Perception/Control)
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation

## 🚀 Final Phase: Full Autonomous Integration
The final system fuses Computer Vision, Path Planning, and Control Theory into a single real-time loop.
- **Perception:** YOLOv3 detects obstacles and maps them to a 10x10 navigation grid.
- **Planning:** A* algorithm recalculates the optimal path in milliseconds when obstacles appear.
- **Control:** The system calculates real-time steering angles and throttle percentages based on the next path node.

🧠 Core Module Logic
Vision (YOLOv3): Processes the camera feed using a Deep Neural Network to classify and locate objects. Coordinates are mapped from 2D pixels to a 2D navigation grid.

Brain (A Algorithm):* Uses a heuristic-based search to find the shortest path while avoiding grid cells marked as "blocked" by the Vision module.

Actuation (Trigonometric Control): Uses the atan2 function to calculate the required heading from the current position to the next target coordinate, outputting steering angles.pip freeze > requirements.txt

## 🛠️ How to Run
1. Clone the repo: `git clone https://github.com/YOUR_USERNAME/AI-Autonomous-Navigation-System.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the system: `python src/autonomous_driver.py`

## 🔮 Future Improvements
- **Sensor Fusion:** Integrating LiDAR data alongside camera feed for better depth accuracy.
- **SLAM Implementation:** Moving from a static 10x10 grid to Simultaneous Localization and Mapping.
- **Performance Optimization:** Converting the model to TensorRT for higher FPS on edge devices.