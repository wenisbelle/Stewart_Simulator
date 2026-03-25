# Stewart Platform Motion Simulator

A simple **motion cueing simulation pipeline** that models the full chain from a generic vehicle dynamics inputs to physical Stewart platform response — including a classical washout filter, 6-DOF inverse/forward kinematics, and PID actuator control.

The main goal is to use Optimization Algorithms to find the best washout algorithms to this platform. 
---

## Overview

This project simulates the behavior of a **6-DOF Stewart platform** as used in flight/driving simulators. It takes raw vehicle motion data (linear accelerations and angular (actually the derivative of the euler angles)) and processes them through a **classical washout filter**, which generates optimal pose commands for the platform. The platform then tracks those commands via PID-controlled linear actuators, with the full response recorded and plotted for analysis.

```
Vehicle Motion Data
      │
      ▼
 Input Generator       ← step / sine / ramp / CSV
      │
      ▼
 Washout Filter        ← high-pass, low-pass, tilt coordination, integration
      │
      ▼
 Pose Commands [x, y, z, roll, pitch, yaw]
      │
      ▼
 Stewart Platform      ← IK → PID actuator control → FK
      │
      ▼
 Recorded Response     ← actuator lengths + estimated pose
```

---

## Features

- **Flexible input generation** — step, sine, ramp, or CSV (e.g. real flight data)
- **Classical washout filter** with 18 tunable parameters:
  - 1st-order high-pass (translational, per axis)
  - 2nd-order high-pass (translational and rotational, per axis)
  - 1st-order low-pass for tilt coordination
  - Double integration (translational) and single integration (rotational)
  - Tilt coordination channel: sustained low-frequency accelerations mapped to gravity-perceived tilt
- **Stewart platform kinematics**:
  - Analytical inverse kinematics (IK)
  - Numerical forward kinematics (FK) via `scipy.optimize.least_squares`
  - 6 configurable base and top anchor positions
- **PID actuator control** very simple PID to track position errors with configurable `kp`, `ki`, `kd` gains
- **Multi-rate simulation**: washout runs at 100 Hz, platform control loop at 1000 Hz

---

## Project Structure

```
├── main.py               # Entry point: simulation configuration and orchestration
├── washout.py            # Classical washout filter implementation
├── stewart_platform.py   # Stewart platform kinematics, PID control, threaded simulator
└── vestibular_system.py  # (Vestibular model — future integration)
```

---

## Getting Started

### Requirements

```bash
pip install numpy scipy matplotlib
```

### Run the Simulation

```bash
python main.py
```

This will:
1. Generate the configured input signal
2. Run it through the washout filter
3. Simulate the Stewart platform tracking the washout commands
4. Plot the results

---

## Configuration

All simulation parameters are set in the `CONFIG` dictionary in `main.py`:

```python
CONFIG = {
    "dt":         0.01,    # Sample period (s) — washout runs at 100 Hz
    "duration":   30.0,    # Total simulation time (s)
    "input_type": "step",  # "step" | "sine" | "ramp" | "csv"

    "step": {
        "accel_amplitudes":      [2.0, 0.0, 0.0],  # x, y, z (m/s²)
        "omega_amplitudes":      [0.0, 0.0, 0.5],  # roll, pitch, yaw (rad/s)
        "start_time_linear":     2.0,               # (s)
        "start_time_angular":    10.0,              # (s)
    },
    ...
}
```

### Washout Filter Parameters

The filter is controlled by an 18-element parameter list passed to `Washout(dt, washout_parameters)`:

| Index | Parameter | Description |
|-------|-----------|-------------|
| 0–2   | `hp1_w`   | 1st-order HP cutoff freq (rad/s) for x, y, z |
| 3–5   | `hp2_w`   | 2nd-order HP natural freq (rad/s) for x, y, z |
| 6–8   | `hp2_e`   | 2nd-order HP damping ratios for x, y, z |
| 9–11  | `lp_w`    | 1st-order LP cutoff freq (rad/s) — tilt coordination |
| 12–14 | `ang_w`   | 2nd-order HP natural freq (rad/s) for roll, pitch, yaw |
| 15–17 | `ang_e`   | 2nd-order HP damping ratios for roll, pitch, yaw |

### Platform Geometry

Configured in `make_platform()` inside `main.py`. Uses a standard 6-leg arrangement with separate base and top anchor radii:

```python
rp = 1.5   # Top plate radius (m)
rb = 1.75  # Base plate radius (m)
h  = 0.35  # Nominal platform height (m)
```

---

## License

MIT License. See `LICENSE` for details.