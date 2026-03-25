import numpy as np
from scipy import signal


class VestibularSystem:
    def __init__(self, discrete_time):
        """
        Represents the human vestibular system, perception for linear accelerations and
        angular velocities.
        """
        PI = 3.1415

        self.dt = discrete_time

        # System thresholds
        # ax, ay, az (m/s²)   roll, pitch, yaw (rad/s)
        self.thresholds = [0.17, 0.17, 0.28, 2 * PI / 180, 2 * PI / 180, 1.6 * PI / 180]

        # Model parameters for the transfer functions
        self.accel_param   = [[0.0] * 4 for _ in range(3)]
        self.angular_param = [[0.0] * 5 for _ in range(3)]

        # x, y, z — each row: [a0, b0, b1, K]
        self.accel_param[0][:] = [0.08, 0.18, 1.51, 1.5]
        self.accel_param[1][:] = [0.08, 0.18, 1.51, 1.5]
        self.accel_param[2][:] = [0.08, 0.18, 1.51, 1.5]

        # roll, pitch, yaw — each row: [t1, t2, ta, tl, G]
        self.angular_param[0][:] = [6.1,  0.1, 30, 0, 1]
        self.angular_param[1][:] = [5.3,  0.1, 30, 0, 1]
        self.angular_param[2][:] = [10.2, 0.0, 30, 0, 1]

    def _apply_threshold(self, signal_array: np.ndarray, threshold: float) -> np.ndarray:
        """Zero out signal values whose absolute value is below the perception threshold."""
        return np.where(np.abs(signal_array) >= threshold, signal_array, 0.0)

    def _vestibular_output(
        self,
        linear_accelerations: list,
        angular_velocities: list,
    ) -> list:
        """
        Receives arrays of linear accelerations and angular velocities and returns
        the vestibular system outputs (perceived accelerations and angular rates).

        Parameters
        ----------
        linear_accelerations : array-like, shape (3, N)
            Rows are x, y, z acceleration time-series [m/s²].
        angular_velocities : array-like, shape (3, N)
            Rows are roll, pitch, yaw rate time-series [rad/s].

        Returns
        -------
        [accel_output, ang_output]
            Each is a list of 3 filtered arrays.
        """
        linear_accelerations = np.asarray(linear_accelerations)
        angular_velocities   = np.asarray(angular_velocities)

        # ---- Linear acceleration transfer functions -------------------------
        # H_accel(s) = K*(s + a0) / (s + b0)*(s + b1)
        num_accel = []
        den_accel = []
        for i in range(3):
            a0, b0, b1, K = self.accel_param[i]
            num_accel.append([K,        K * a0     ])   # K*s + K*a0
            den_accel.append([1.0, b0 + b1, b0 * b1])   # s^2 + (b0+b1)*s + b0*b1

        # ---- Angular velocity transfer functions ----------------------------
        # H_ang(s) = G*t1*ta*tl*s^3 + G*t1*ta*s^2
        #            -----------------------------------------------
        #            ta*t1*t2*s^3 + [(t1*t2)+(ta*t1)+(ta*t2)]*s^2 + [ta+t1+t2]*s + 1
        num_ang = []
        den_ang = []
        for i in range(3):
            t1, t2, ta, tl, G = self.angular_param[i]
            num_ang.append([
                G * t1 * ta * tl,  # s^3
                G * t1 * ta,       # s^2
                0.0,               # s^1
                0.0,               # s^0
            ])
            den_ang.append([
                ta * t1 * t2,
                (t1 * t2) + (ta * t1) + (ta * t2),
                ta + t1 + t2,
                1.0,
            ])

        # ---- Discretise and filter -----------------------------------------
        accel_system_output = []
        for i in range(3):
            # BUG FIX: use correct variable names (was num_d / den_d / omega_i_data)
            sys_d = signal.cont2discrete((num_accel[i], den_accel[i]), self.dt, method='bilinear')
            num_d = sys_d[0].flatten()   # BUG FIX: .flatten() avoids shape issues
            den_d = sys_d[1].flatten()

            filtered = signal.lfilter(num_d, den_d, linear_accelerations[i])

            # Apply perception threshold
            filtered = self._apply_threshold(filtered, self.thresholds[i])
            accel_system_output.append(filtered)

        ang_system_output = []
        for i in range(3):
            # BUG FIX: use correct variable names
            sys_d = signal.cont2discrete((num_ang[i], den_ang[i]), self.dt, method='bilinear')
            num_d = sys_d[0].flatten()
            den_d = sys_d[1].flatten()

            filtered = signal.lfilter(num_d, den_d, angular_velocities[i])

            # Apply perception threshold (index offset by 3 for angular)
            filtered = self._apply_threshold(filtered, self.thresholds[3 + i])
            ang_system_output.append(filtered)

        return [accel_system_output, ang_system_output]