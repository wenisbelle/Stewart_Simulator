import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
import threading
import time

class StewartPlatform:
    def __init__(self, base_anchors, top_anchors, kp=1.0, ki=0.0, kd=0.0):
        """
        base_anchors: 6x3 numpy array of anchor coordinates on the base (relative to its center).
        top_anchors: 6x3 numpy array of anchor coordinates on the top plate (relative to its center).
        kp, ki, kd: PID gains for the actuators.
        """
        self.base_anchors = np.array(base_anchors)
        self.top_anchors = np.array(top_anchors)

        # State variables
        self.current_pose = [0, 0, 0, 0, 0, 0]  # [x, y, z, roll, pitch, yaw]
        self.target_pose =  [0, 0, 0, 0, 0, 0]  # [x, y, z, roll, pitch, yaw]

        # Actuator lengths
        self.current_lengths = self._calculate_ik(self.current_pose)
        self.target_lengths = np.copy(self.current_lengths)

        # Actuator limits
        self.actuator_minimum_lengths = 0.75 * np.ones(6)
        self.actuator_maximum_lengths = 1.35  * np.ones(6)   

        # PID Controller states
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_error = np.zeros(6)
        self.previous_error = self.target_lengths - self.current_lengths

    def _euler_to_matrix(self, rpy):
        """Converts Roll, Pitch, Yaw to a rotation matrix."""
        return R.from_euler('xyz', rpy, degrees=False).as_matrix()

    def _calculate_ik(self, pose):
        """
        Inverse Kinematics: Calculates exact actuator lengths for a given pose.
        pose: [x, y, z, roll, pitch, yaw]
        """
        pos = pose[:3]
        rpy = pose[3:]
        rot_matrix = self._euler_to_matrix(rpy)

        lengths = np.zeros(6)
        for i in range(6):
            # P_i = T + R * p_i - B_i
            # Where T is translation, R is rotation, p_i is top anchor, B_i is base anchor
            top_pos_global = pos + rot_matrix.dot(self.top_anchors[i])
            leg_vector = top_pos_global - self.base_anchors[i]
            lengths[i] = np.linalg.norm(leg_vector)
            #print(f"Actuator {i} with length {lengths[i]}")
        return lengths

    def set_target_pose(self, x, y, z, roll, pitch, yaw):
        """Sets the desired pose and calculates the required target leg lengths."""
        self.target_pose = np.array([x, y, z, roll, pitch, yaw])
        self.target_lengths = self._calculate_ik(self.target_pose)

    def get_forward_kinematics(self):
        """
        Forward Kinematics: Numerically estimates the current pose based on current leg lengths.
        Uses scipy.optimize.least_squares to minimize the difference between
        the measured lengths and the lengths calculated by IK for a guessed pose.
        """
        def error_function(guess_pose):
            guessed_lengths = self._calculate_ik(guess_pose)
            return guessed_lengths - self.current_lengths

        # Use the previous known pose as the initial guess to speed up convergence
        initial_guess = self.current_pose

        result = least_squares(error_function, initial_guess, method='lm',ftol=1e-5, xtol=1e-5)

        if result.status > 0:
            self.current_pose = result.x
        else:
            print("Warning: Forward kinematics did not converge!")
 
        return self.current_pose

    def control_step(self, dt):
        """
        Advances the simulation by one time step (dt).
        Applies the PID control to calculate actuator velocities/positions,
        updates the actuator lengths, and recalculates the actual platform pose.
        """
        # Calculate errors
        error = self.target_lengths - self.current_lengths
        self.integral_error += error * dt
        derivative_error = (error - self.previous_error) / dt

        # PID Output (treating output as actuator velocity for a simple model)
        actuator_velocity = (self.kp * error) + (self.ki * self.integral_error) + (self.kd * derivative_error)

        # Update current lengths (Kinematic update: pos = pos + v * dt)
        self.current_lengths += actuator_velocity * dt
        
        self.current_lengths = np.clip(
            self.current_lengths,
            self.actuator_minimum_lengths,
            self.actuator_maximum_lengths,
        )
 
        self.previous_error = error
 
        return self.current_lengths

class ThreadedSimulator:
    def __init__(self, platform, dt=0.001):
        """
        Wraps the StewartPlatform to run its control loop in a background thread.
        dt: The time step for the control loop in seconds (e.g., 0.01 for 1000Hz).
        """
        self.platform = platform
        self.dt = dt
        self.running = False
        self.thread = None

        # A lock to prevent the simulation thread and main thread from
        # accessing the platform data at the exact same time.
        self.lock = threading.Lock()

        # Store the latest pose so the main thread can read it safely
        self.current_pose = np.zeros(6)

        # Store the latest lenghts of the actuators
        self.current_lengths = np.zeros(6)

    def _simulation_loop(self):
        """The core loop that runs in the background."""
        while self.running:
            start_time = time.time()

            # Acquire lock to perform the control step safely
            with self.lock:
                self.current_pose, self.current_lengths = self.platform.control_step(self.dt)

            # Calculate how long the math took, and sleep for the remainder of dt
            elapsed_time = time.time() - start_time
            sleep_time = max(0.0, self.dt - elapsed_time)
            time.sleep(sleep_time)

    def start(self):
        """Starts the background simulation thread."""
        if not self.running:
            self.running = True
            # daemon=True ensures the thread dies if the main program exits
            self.thread = threading.Thread(target=self._simulation_loop, daemon=True)
            self.thread.start()
            print("Simulator thread started.")

    def stop(self):
        """Stops the background simulation thread safely."""
        self.running = False
        if self.thread is not None:
            self.thread.join()
            print("Simulator thread stopped.")

    def update_target_pose(self, x, y, z, roll, pitch, yaw):
        """Thread-safe method to update the target pose from the main thread."""
        with self.lock:
            self.platform.set_target_pose(x, y, z, roll, pitch, yaw)

    def get_current_pose(self):
        """Thread-safe method to read the current pose from the main thread."""
        with self.lock:
            return np.copy(self.current_pose)

    def get_current_lengths(self):
        """Thread-safe method to read the current lengths from the main thread."""
        with self.lock:
            return np.copy(self.current_lengths)