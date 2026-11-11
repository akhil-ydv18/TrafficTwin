# TrafficTwin — Hybrid GA–PSO Traffic Simulation Calibration

This project implements the calibration methodology for microscopic traffic simulations, combining:
* **Global Exploration:** Genetic Algorithm (GA) arithmetic crossover and adaptive Gaussian mutation.
* **Local Exploitation:** Particle Swarm Optimization (PSO) velocity updates with cognitive/social trajectory attractors.
* **Objective:** Minimizing Root-Mean-Square Error (RMSE) against loop detector hourly counts.

---

## Project Structure

```
traffic-twin-calibration/
├── index.html            # Interactive web visualizer (Canvas simulation + convergence plots)
├── run.py                # Unified launcher (web visualizer or Python CLI benchmark)
├── calibrate.py          # Comparative benchmark (Hybrid GA-PSO vs Pure PSO vs Pure GA)
├── hybrid_ga_pso.py      # Core Hybrid GA-PSO optimization algorithm
├── traffic_model.py      # Microscopic IDM car-following and multi-detector corridor engine
├── requirements.txt      # Python dependencies (numpy, matplotlib)
├── .vscode/
│   └── launch.json       # Preconfigured VS Code Run & Debug profiles
└── README.md             # Project guide
```

---

## How to Run in VS Code

### Step 1: Open the Project in VS Code
1. In VS Code, go to **File > Open Folder...**
2. Select the directory:
   ```
   C:\Users\Lenovo\.gemini\antigravity\scratch\traffic-twin-calibration
   ```

### Step 2: Install Python Dependencies (Optional for CLI)
Open the VS Code integrated terminal (`Ctrl + ~`) and install requirements:
```bash
pip install -r requirements.txt
```

### Step 3: Run the Project

#### Option A: Run the Interactive Web Visualizer
* **Via Python:** Run `python run.py` in the terminal, then visit `http://localhost:8080/`.
* **Via VS Code "Run and Debug" (F5):** Select **"TrafficTwin: Start Visualizer Web Server"**.
* **Via Live Server extension:** Right-click `index.html` and choose **"Open with Live Server"**.

#### Option B: Run the Python Calibration Engine
* In the terminal, execute:
  ```bash
  python calibrate.py
  ```
  *(or `python run.py --cli`)*
* This runs all three solvers (**Hybrid GA-PSO**, **Pure PSO**, and **Pure GA**), prints tabular RMSE metrics, and outputs `convergence_plot.png`.
