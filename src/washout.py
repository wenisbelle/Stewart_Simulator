import numpy as np
from scipy import signal


class Washout:
    def __init__(self, dt: float, washout_parameters: list[float]):
        """
        Represents the classical washout filter for motion cueing.

        Parameters (18 total):
          [0:3]   - HP1 cutoff frequencies (rad/s) for x, y, z accelerations (1st-order high-pass)
          [3:6]   - HP2 natural frequencies (rad/s) for x, y, z accelerations (2nd-order high-pass)
          [6:9]   - HP2 damping ratios for x, y, z accelerations (2nd-order high-pass)
          [9:12]  - LP cutoff frequencies (rad/s) for x, y, z accelerations (1st-order low-pass,
                    feeds tilt coordination channel)
          [12:15] - HP natural frequencies (rad/s) for roll, pitch, yaw (2nd-order high-pass)
          [15:18] - HP damping ratios for roll, pitch, yaw (2nd-order high-pass)

        The classical washout structure per translational axis is:
          - High-frequency channel: accel -> 2nd-order HP -> double-integrate -> position command
          - Tilt-coordination channel: accel -> 1st-order LP -> 1st-order HP -> tilt angle command
            (uses sustained low-frequency acceleration perceived as tilt via gravity)

        The rotational channel per axis is:
          - ang_vel -> 2nd-order HP -> angle command (direct integration implicit in platform IK)
        """
        self.dt = dt

        # Unpack washout parameters
        self.hp1_w  = washout_parameters[0:3]   # 1st-order HP cutoff for x,y,z
        self.hp2_w  = washout_parameters[3:6]   # 2nd-order HP natural freq for x,y,z
        self.hp2_e  = washout_parameters[6:9]   # 2nd-order HP damping for x,y,z
        self.lp_w   = washout_parameters[9:12]  # 1st-order LP cutoff for tilt coord x,y,z
        self.ang_w  = washout_parameters[12:15] # 2nd-order HP natural freq for roll,pitch,yaw
        self.ang_e  = washout_parameters[15:18] # 2nd-order HP damping for roll,pitch,yaw

        # Scaling factors to keep motion within platform workspace
        self.linear_scaling  = 0.8
        self.angular_scaling = 0.8

        # Build and discretise all transfer functions
        # Each dict stores (num_d, den_d) ready for lfilter

        # --- Translational high-frequency channel: 2nd-order HP ---
        # H_HP1(s) = s / (s + w)
        self._tf_hp1_accel = []
        for i in range(3):
            w = self.hp1_w[i]
            num_c = [1.0, 0.0]    # s
            den_c = [1.0, w]      # s + w
            nd, dd, _ = signal.cont2discrete((num_c, den_c), dt, method='bilinear')
            self._tf_hp1_accel.append((nd.flatten(), dd.flatten()))
        
        
        # --- Translational high-frequency channel: 2nd-order HP ---
        # H_HP2(s) = s^2 / (s^2 + 2*e*w*s + w^2)
        self._tf_hp2_accel = []
        for i in range(3):
            w = self.hp2_w[i]
            e = self.hp2_e[i]
            num_c = [1.0, 0.0, 0.0]                   # s^2
            den_c = [1.0, 2.0 * e * w, w ** 2]        # (s^2 + 2*e*w*s + w^2)
            nd, dd, _ = signal.cont2discrete((num_c, den_c), dt, method='bilinear')
            self._tf_hp2_accel.append((nd.flatten(), dd.flatten()))

        # Double integrator in discrete time: H_int(z) = dt^2*(z+1)^2 / (4*(z-1)^2)
        # Implemented as two cascaded 1st-order integrators using the Tustin rule:
        #   H_int1(z) = (dt/2) * (z+1)/(z-1)
        num_int = np.array([dt / 2.0,  dt / 2.0])
        den_int = np.array([1.0,      -1.0      ])
        self._tf_integrator = (num_int, den_int)  # applied twice per axis

        # --- Tilt coordination channel ---
        # Low-pass: H_LP(s) = w_lp / (s + w_lp)
        self._tf_lp = []
        for i in range(3):
            w = self.lp_w[i]
            num_c = [w]
            den_c = [1.0, w]
            nd, dd, _ = signal.cont2discrete((num_c, den_c), dt, method='bilinear')
            self._tf_lp.append((nd.flatten(), dd.flatten()))


        # --- Rotational channel: 2nd-order HP ---
        # H_HP_ang(s) = s^2 / (s^2 + 2*e*w*s + w^2)
        self._tf_hp2_ang = []
        for i in range(3):
            w = self.ang_w[i]
            e = self.ang_e[i]
            num_c = [1.0, 0.0, 0.0]             # s^2
            den_c = [1.0, 2.0 * e * w, w ** 2]  # (s^2 + 2*e*w*s + w^2)
            nd, dd, _ = signal.cont2discrete((num_c, den_c), dt, method='bilinear')
            self._tf_hp2_ang.append((nd.flatten(), dd.flatten()))

        # Single integrator for angular channel (vel -> angle)
        self._tf_ang_integrator = (num_int.copy(), den_int.copy())

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def _washout_output(
        self,
        linear_accelerations: list[float],
        angular_velocities: list[float],
    ) -> list[float]:
        """
        Processes arrays of linear accelerations and angular velocities through
        the classical washout filter and returns the desired Stewart platform pose.

        Parameters
        ----------
        linear_accelerations : array-like, shape (3, N)
            Rows are x, y, z acceleration time-series [m/s²].
        angular_velocities : array-like, shape (3, N)
            Rows are roll, pitch, yaw rate time-series [rad/s].

        Returns
        -------
        pose_commands : list of 6 arrays, each length N
            [x, y, z, roll, pitch, yaw] platform pose commands.
            Translations in metres, rotations in radians.
        """
        linear_accelerations = np.asarray(linear_accelerations)  # (3, N)
        angular_velocities   = np.asarray(angular_velocities)    # (3, N)

        # ---- Translational axes ----------------------------------------
        position_commands = []
        tilt_commands     = []  # tilt-coordination angles (used for roll/pitch from sustained accel)

        for i in range(3):
            accel_i = linear_accelerations[i]
        
            # 1. 1st-order HP
            hp1_num, hp1_den = self._tf_hp1_accel[i]
            hp1_out = signal.lfilter(hp1_num, hp1_den, accel_i)
        
            # 2. 2nd-order HP (in series after 1st-order HP)
            hp2_num, hp2_den = self._tf_hp2_accel[i]
            hp2_out = signal.lfilter(hp2_num, hp2_den, hp1_out)  # feeds from hp1, not accel_i
        
            # 3. Double integrate
            int1_num, int1_den = self._tf_integrator
            vel_hf = signal.lfilter(int1_num, int1_den, hp2_out)
            pos_hf = signal.lfilter(int1_num, int1_den, vel_hf)
        
            position_commands.append(self.linear_scaling * pos_hf)
        
            # Tilt coordination: LP only, then directly to angle
            lp_num, lp_den = self._tf_lp[i]
            lp_out = signal.lfilter(lp_num, lp_den, accel_i)
            tilt_commands.append(self.angular_scaling * np.arctan(lp_out / 9.81))

        # ---- Rotational axes -------------------------------------------
        angle_commands = []

        for i in range(3):
            omega_i = angular_velocities[i]

            # 2nd-order HP filter on angular velocity
            hp2_num, hp2_den = self._tf_hp2_ang[i]
            hp2_out = signal.lfilter(hp2_num, hp2_den, omega_i)

            # Integrate to get angle
            int_num, int_den = self._tf_ang_integrator
            angle_hf = signal.lfilter(int_num, int_den, hp2_out)

            angle_commands.append(self.angular_scaling * angle_hf)

        # ---- Combine channels ------------------------------------------
        # Tilt coordination contributes to roll (from y accel) and pitch (from x accel).
        # Convention: pose = [x, y, z, roll, pitch, yaw]
        x_cmd     = position_commands[0]
        y_cmd     = position_commands[1]
        z_cmd     = position_commands[2]

        roll_cmd  = angle_commands[0] + tilt_commands[1]  # rot channel + tilt from y accel
        pitch_cmd = angle_commands[1] + tilt_commands[0]  # rot channel + tilt from x accel
        yaw_cmd   = angle_commands[2]                     # yaw has no tilt coordination

        return [x_cmd, y_cmd, z_cmd, roll_cmd, pitch_cmd, yaw_cmd]