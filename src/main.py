import numpy as np
import time
import matplotlib.pyplot as plt
from washout import Washout
from stewart_platform import StewartPlatform, ThreadedSimulator
from vestibular_system import VestibularSystem

# Simulation configuration 

CONFIG = {
    # Time
    "dt":       0.01,    # sample period (s)
    "duration": 30.0,    # total simulation time (s)

    # Input type: "step", "sine", "ramp", "csv"
    "input_type": "step",

    # Step input parameters
    "step": {
        "accel_amplitudes": [2.0, 0.0, 0.0],   # x, y, z (m/s²)
        "omega_amplitudes": [0.0, 0.0, 0.5],   # roll, pitch, yaw (rad/s)
        "start_time_linear":       2.0,        # (s)
        "start_time_angular":      3.0,
    },

    # Sine input parameters
    "sine": {
        "accel_amplitudes": [2.0, 1.0, 0.5],   # x, y, z (m/s²)
        "omega_amplitudes": [0.3, 0.2, 0.1],   # roll, pitch, yaw (rad/s)
        "accel_freqs_hz":   [0.5, 0.3, 0.2],   # one per axis
        "omega_freqs_hz":   [0.4, 0.3, 0.2],
    },

    # Ramp input parameters
    "ramp": {
        "accel_slopes": [0.0, 0.0, 0.0],        # m/s² per second
        "omega_slopes": [0.0, 0.0, 0.1],        # rad/s per second
    },

    # CSV input — rows: t, ax, ay, az, roll_rate, pitch_rate, yaw_rate
    "csv": {
        "filepath": "data/flight_data.csv",
    },
}

#  Input generator 

class InputGenerator:
    def __init__(self, config: dict):
        self.dt       = config["dt"]
        self.duration = config["duration"]
        self.N        = int(self.duration / self.dt)
        self.t        = np.linspace(0, self.duration, self.N)
        self.cfg      = config

    def generate(self):
        """Returns linear_accel (3, N) and angular_vel (3, N)."""
        mode = self.cfg["input_type"]
        if   mode == "step": return self._step()
        elif mode == "sine": return self._sine()
        elif mode == "ramp": return self._ramp()
        elif mode == "csv":  return self._csv()
        else:
            raise ValueError(f"Unknown input_type: '{mode}'")

    def _step(self):
        p = self.cfg["step"]
        accel = np.zeros((3, self.N))
        omega = np.zeros((3, self.N))
        start_linear = int(p["start_time_linear"] / self.dt)
        start_angular = int(p["start_time_angular"] / self.dt)
        for i in range(3):
            accel[i, start_linear:] = p["accel_amplitudes"][i]
            omega[i, start_angular:] = p["omega_amplitudes"][i]
        return accel, omega

    def _sine(self):
        p = self.cfg["sine"]
        accel = np.array([
            p["accel_amplitudes"][i] * np.sin(2 * np.pi * p["accel_freqs_hz"][i] * self.t)
            for i in range(3)
        ])
        omega = np.array([
            p["omega_amplitudes"][i] * np.sin(2 * np.pi * p["omega_freqs_hz"][i] * self.t)
            for i in range(3)
        ])
        return accel, omega

    def _ramp(self):
        p = self.cfg["ramp"]
        accel = np.array([p["accel_slopes"][i] * self.t for i in range(3)])
        omega = np.array([p["omega_slopes"][i] * self.t for i in range(3)])
        return accel, omega

    def _csv(self):
        import pandas as pd
        df    = pd.read_csv(self.cfg["csv"]["filepath"])
        accel = df[["ax", "ay", "az"]].to_numpy().T           # (3, N)
        omega = df[["roll_rate", "pitch_rate", "yaw_rate"]].to_numpy().T
        # Resample to match dt if needed
        self.N = accel.shape[1]
        self.t = np.arange(self.N) * self.dt
        return accel, omega

    def _plot_input(self, linear_accel, angular_vel):
        labels_accel = ['ax', 'ay', 'az']
        labels_omega = ['roll rate', 'pitch rate', 'yaw rate']
        colors       = ['tab:blue', 'tab:green', 'tab:red']

        fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex=True)
        fig.suptitle('Washout Filter Inputs', fontsize=14)   
        
        for i in range(3):
            # Linear accelerations (top row)
            axes[0, i].plot(self.t, linear_accel[i], color=colors[i], linewidth=0.8)
            axes[0, i].set_title(labels_accel[i])
            axes[0, i].set_ylabel('m/s²')
            axes[0, i].grid(True, alpha=0.3)
            axes[0, i].axhline(0, color='black', linewidth=0.5, linestyle='--')

            # Angular velocities (bottom row)
            axes[1, i].plot(self.t, angular_vel[i], color=colors[i], linewidth=0.8)
            axes[1, i].set_title(labels_omega[i])
            axes[1, i].set_ylabel('rad/s')
            axes[1, i].set_xlabel('time (s)')
            axes[1, i].grid(True, alpha=0.3)
            axes[1, i].axhline(0, color='black', linewidth=0.5, linestyle='--')

        plt.tight_layout()
        plt.show()

def plot_washout(commands: list[float]):
    labels_linear = ['x', 'y', 'z']
    labels_angular = ['roll', 'pitch', 'yaw']
    colors       = ['tab:blue', 'tab:green', 'tab:red']
    fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex=True)
    fig.suptitle('Washout Filter Outputs', fontsize=14) 

    dt       = 0.01
    duration = 30.0
    N        = int(duration / dt)
    t        = np.linspace(0,duration, N)

    for i in range(3):
        # Linear positions (top row)
        axes[0, i].plot(t, commands[i], color=colors[i], linewidth=0.8)
        axes[0, i].set_title(labels_linear[i])
        axes[0, i].set_ylabel('m')
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].axhline(0, color='black', linewidth=0.5, linestyle='--')
        # Angular velocities (bottom row)
        axes[1, i].plot(t, commands[i+3], color=colors[i], linewidth=0.8)
        axes[1, i].set_title(labels_angular[i])
        axes[1, i].set_ylabel('rad')
        axes[1, i].set_xlabel('time (s)')
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.tight_layout()
    plt.show()

def make_platform(kp=20.0, ki=1.0, kd=0.0):
    """
    Standard 6x6 Stewart platform with evenly spaced anchors.
    Top plate radius = 1.5 m, base plate radius = 1.75 m.
    Anchors are placed at angles 0°, 60°, 120°, 180°, 240°, 300°.
    """
    rp = 1.5
    rb = 1.75
    h  = 0.35    # nominal platform height (m)

    base_angles = np.radians([10, 50, 130, 170, 250, 290])
    top_angles  = np.radians([350, 70, 110, 190, 230, 310])

    top_anchors  = np.array([[rp * np.cos(a), rp * np.sin(a), 0.0] for a in top_angles])
    base_anchors = np.array([[rb * np.cos(a), rb * np.sin(a), -h] for a in base_angles])

    return StewartPlatform(base_anchors, top_anchors, kp=kp, ki=ki, kd=kd)


def plot_platform_response(commands, recorded_poses, dt):
    N      = recorded_poses.shape[1]
    t      = np.linspace(0, N * dt, N)
    labels = ['x (m)', 'y (m)', 'z (m)', 'roll (rad)', 'pitch (rad)', 'yaw (rad)']

    fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex=True)
    fig.suptitle('Washout command vs platform response')
    axes = axes.flatten()

    for i in range(6):
        axes[i].plot(t, commands[i],          label='washout command', linewidth=1.0)
        axes[i].plot(t, recorded_poses[i, :], label='platform pose',   linewidth=1.0, linestyle='--')
        axes[i].set_title(labels[i])
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)
        axes[i].axhline(0, color='black', linewidth=0.5, linestyle='--')

    plt.tight_layout()
    plt.show()

def plot_lengths(recorded_lengths, dt):
    N      = recorded_lengths.shape[1]
    t      = np.linspace(0, N * dt, N)
    labels = ['l1 (m)', 'l2 (m)', 'l3 (m)','l4 (m)', 'l5 (m)', 'l6 (m)']

    fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex=True)
    fig.suptitle('Platform lenght')
    axes = axes.flatten()

    for i in range(6):
        axes[i].plot(t, recorded_lengths[i, :], label='platform lengths',   linewidth=1.0, linestyle='--')
        axes[i].set_title(labels[i])
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)
        axes[i].axhline(0, color='black', linewidth=0.5, linestyle='--')

    plt.tight_layout()
    plt.show()    


def main():
    gen          = InputGenerator(CONFIG)
    linear_accel, angular_vel = gen.generate()

    gen._plot_input(linear_accel, angular_vel)
    washout_param = [2.8, 3, 6,    # HP1 cutoff frequencies (rad/s) for x, y, z accelerations (1st-order high-pass)
                    0.7, 0.6, 0.6, # HP2 natural frequencies (rad/s) for x, y, z accelerations (2nd-order high-pass)
                    0.8, 0.8, 0.8, # HP2 damping ratios for x, y, z accelerations (2nd-order high-pass)
                    0.5, 0.5, 0.5, # LP cutoff frequencies (rad/s) for x, y, z accelerations (1st-order low-pass
                    1.5, 1.5, 1.5, # HP natural frequencies (rad/s) for roll, pitch, yaw (2nd-order high-pass)
                    0.5, 0.5, 0.5  # HP damping ratios for roll, pitch, yaw (2nd-order high-pass)
                    ]

    ### Washout commands 
    washout  = Washout(CONFIG["dt"], washout_parameters=washout_param)
    commands = washout._washout_output(linear_accel, angular_vel) 
    #plot_washout(commands)

    ### Initialize the base platform and the threaded simulator running at higher frequency
    platform = make_platform()

    ### Storage for results
    N = linear_accel.shape[1]
    print(f"Number of points: {N}")
    recorded_poses = np.zeros((6, N))
    recorded_lengths = np.zeros((6,N))

    # Calculate how many high-frequency steps fit into one washout step
    washout_dt = CONFIG["dt"]   # 0.01 s
    sim_dt = 0.001              # 1000 Hz simulation
    steps_per_loop = int(washout_dt / sim_dt) # 10 steps

    for k in range(N):
        # Extract the pose command for this timestep
        target = [float(commands[dof][k]) for dof in range(6)]
        platform.set_target_pose(*target)
        
        for _ in range(steps_per_loop):
            # This advances the platform's internal state purely mathematically
            actual_length = platform.control_step(sim_dt)

        # 3. Record the pose at the end of the 0.01s window
        recorded_poses[:, k] = platform.get_forward_kinematics()
        recorded_lengths[:, k] = actual_length


    # Post-simulation
    #plot_washout(commands)
    plot_platform_response(commands, recorded_poses, CONFIG["dt"])

    plot_lengths(recorded_lengths, CONFIG["dt"])

if __name__ == "__main__":
    main()











