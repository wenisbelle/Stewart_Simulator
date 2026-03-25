"""
test_washout.py

Tests for Washout filter class, using the architecture from:
  Volkaner et al., "Realization of a Desktop Flight Simulation System
  for Motion-cueing Studies", Int J Adv Robot Syst, 2016.

Architecture (per translational axis):
  accel → [1st-order HP] → [2nd-order HP] → [∫∫] → position
  accel → [1st-order LP] → arctan(a/g)           → tilt angle

Rotational axis:
  omega → [2nd-order HP] → [∫] → angle

Run with:  python test_washout.py
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from washout import Washout

# ── helpers ──────────────────────────────────────────────────────────────────

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"

def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"{status}  {name}{suffix}")
    if not condition:
        raise AssertionError(f"Test failed: {name}")


# ── fixtures ──────────────────────────────────────────────────────────────────

DT  = 0.01          # 100 Hz
T   = 30.0
N   = int(T / DT)
t   = np.linspace(0, T, N)

# Parameters from Table 7 of the paper (rad/s)
# [hp1_x, hp1_y, hp1_z,
#  hp2_x, hp2_y, hp2_z,
#  e_x,   e_y,   e_z,
#  lp_x,  lp_y,  lp_z,
#  ang_w_roll, ang_w_pitch, ang_w_yaw,
#  ang_e_roll, ang_e_pitch, ang_e_yaw]
PAPER_PARAMS = [
    2.8,  3.0,  6.0,    # 1st-order HP: x, y, z
    0.7,  0.6,  0.6,    # 2nd-order HP natural freq: x, y, z
    1.0,  1.0,  1.0,    # 2nd-order HP damping (paper uses zeta=1)
    0.0,  0.5,  0.5,    # 1st-order LP: x, y, z  (0 means no tilt for x)
    1.5,  1.5,  1.5,    # angular 2nd-order HP natural freq
    1.0,  1.0,  1.0,    # angular 2nd-order HP damping
]

def make_washout(params=None):
    return Washout(DT, params if params is not None else PAPER_PARAMS)

def zero_inputs():
    return np.zeros((3, N)), np.zeros((3, N))

def sine_accel(freq_hz=2.0, amp=1.0):
    sig = amp * np.sin(2 * np.pi * freq_hz * t)
    return np.vstack([sig, sig, sig]), np.zeros((3, N))

def sine_omega(freq_hz=2.0, amp=0.5):
    sig = amp * np.sin(2 * np.pi * freq_hz * t)
    return np.zeros((3, N)), np.vstack([sig, sig, sig])

def sustained_accel(amp=2.0):
    """Constant (DC) acceleration — exercises tilt coordination."""
    return np.ones((3, N)) * amp, np.zeros((3, N))


# ── tests ─────────────────────────────────────────────────────────────────────

def test_instantiation():
    """Washout can be created without errors; all filter banks are populated."""
    w = make_washout()
    check("Instantiation", w is not None)
    check("dt stored", w.dt == DT)
    check("3 HP1 accel filters",  len(w._tf_hp1_accel) == 3)
    check("3 HP2 accel filters",  len(w._tf_hp2_accel) == 3)
    check("3 LP filters",         len(w._tf_lp) == 3)
    check("3 HP2 angular filters", len(w._tf_hp2_ang) == 3)
    check("Integrator filter set", w._tf_integrator is not None)
    check("Angular integrator set", w._tf_ang_integrator is not None)


def test_output_structure():
    """_washout_output returns a list of 6 arrays each of length N."""
    w = make_washout()
    accel, omega = sine_accel()
    result = w._washout_output(accel, omega)

    check("Output is a list", isinstance(result, list))
    check("Output has 6 channels", len(result) == 6,
          f"got {len(result)}")
    for i, name in enumerate(["x", "y", "z", "roll", "pitch", "yaw"]):
        check(f"Channel {name} length == N",
              len(result[i]) == N,
              f"got {len(result[i])}")


def test_zero_input_gives_zero_output():
    """Zero inputs must produce zero outputs."""
    w = make_washout()
    accel, omega = zero_inputs()
    result = w._washout_output(accel, omega)

    names = ["x", "y", "z", "roll", "pitch", "yaw"]
    for i, name in enumerate(names):
        check(f"Zero input → zero output [{name}]",
              np.allclose(result[i], 0.0, atol=1e-10),
              f"max={np.max(np.abs(result[i])):.2e}")


def test_output_is_finite():
    """No NaN or Inf values in the output for typical inputs."""
    w = make_washout()
    accel, omega = sine_accel(freq_hz=2.0, amp=5.0)
    result = w._washout_output(accel, omega)

    names = ["x", "y", "z", "roll", "pitch", "yaw"]
    for i, name in enumerate(names):
        check(f"Output finite [{name}]",
              np.all(np.isfinite(result[i])))


def test_hp_channels_attenuate_dc():
    """
    The high-frequency translational channel must attenuate a sustained
    (DC) acceleration — position should decay back toward zero.
    High-pass filters block DC, so the platform must return to neutral.
    """
    w = make_washout()
    accel, omega = sustained_accel(amp=2.0)
    x_cmd = w._washout_output(accel, omega)[0]  # x position command

    early_peak = np.max(np.abs(x_cmd[:200]))    # first 2 s
    late_value = np.mean(np.abs(x_cmd[-200:]))  # last 2 s

    check("HP channel attenuates DC (position decays)",
          late_value < early_peak,
          f"early_peak={early_peak:.4f}, late_mean={late_value:.4f}")


def test_hp_channels_pass_high_freq():
    """
    A high-frequency acceleration (above corner frequency) should produce
    a non-negligible position command.
    """
    w = make_washout()
    # 2 Hz >> corner freq (0.7 rad/s ≈ 0.11 Hz)
    accel, omega = sine_accel(freq_hz=2.0, amp=2.0)
    result = w._washout_output(accel, omega)

    for i, name in enumerate(["x", "y", "z"]):
        peak = np.max(np.abs(result[i]))
        check(f"High-freq accel produces non-zero position [{name}]",
              peak > 1e-5,
              f"peak={peak:.4e}")


def test_tilt_coordination_from_sustained_accel():
    """
    A sustained acceleration along y must produce a non-zero roll command
    (tilt coordination), and sustained x acceleration must produce pitch.
    Axes with lp_w = 0 (x in paper) should produce zero tilt.
    """
    w = make_washout()
    amp = 2.0
    # Sustained acceleration only on y axis
    accel = np.zeros((3, N))
    accel[1] = amp  # y-axis acceleration
    omega = np.zeros((3, N))

    result = w._washout_output(accel, omega)
    roll_cmd  = result[3]
    pitch_cmd = result[4]

    check("Y-accel produces roll tilt command",
          np.max(np.abs(roll_cmd)) > 1e-4,
          f"max_roll={np.max(np.abs(roll_cmd)):.4e}")

    # x-axis LP is 0 rad/s → no pitch tilt from y-axis accel
    check("Y-accel alone does not affect pitch via HP channel",
          np.max(np.abs(pitch_cmd)) < 0.1,
          f"max_pitch={np.max(np.abs(pitch_cmd)):.4e}")


def test_rotational_channel_passes_high_freq():
    """
    High-frequency angular velocity input must produce a non-zero
    angle command (2nd-order HP passes high frequencies).
    """
    w = make_washout()
    accel, omega = sine_omega(freq_hz=2.0, amp=1.0)
    result = w._washout_output(accel, omega)

    for i, name in enumerate(["roll", "pitch", "yaw"]):
        peak = np.max(np.abs(result[3 + i]))
        check(f"High-freq omega produces angle command [{name}]",
              peak > 1e-5,
              f"peak={peak:.4e}")


def test_rotational_channel_attenuates_dc():
    """
    A constant (DC) angular velocity must wash out over time — the
    angle command must decay back toward zero.
    """
    w = make_washout()
    omega = np.ones((3, N)) * 0.5  # 0.5 rad/s constant
    accel = np.zeros((3, N))
    result = w._washout_output(accel, omega)

    for i, name in enumerate(["roll", "pitch", "yaw"]):
        cmd = result[3 + i]
        early = np.max(np.abs(cmd[:200]))
        late  = np.mean(np.abs(cmd[-200:]))
        check(f"Angular HP washes out DC [{name}]",
              late < early,
              f"early={early:.4f}, late={late:.4f}")


def test_scaling_applied():
    """
    linear_scaling and angular_scaling must be applied.
    Halving the scaling must approximately halve the output magnitude.
    """
    w1 = make_washout()
    w2 = make_washout()
    w2.linear_scaling  = w1.linear_scaling  * 0.5
    w2.angular_scaling = w1.angular_scaling * 0.5

    accel, omega = sine_accel(freq_hz=2.0, amp=2.0)
    r1 = w1._washout_output(accel, omega)
    r2 = w2._washout_output(accel, omega)

    peak1 = np.max(np.abs(r1[0]))  # x position
    peak2 = np.max(np.abs(r2[0]))
    ratio = peak2 / peak1 if peak1 > 0 else 1.0

    check("Halving linear_scaling halves position output",
          0.4 < ratio < 0.6,
          f"ratio={ratio:.3f}")


def test_parameter_sensitivity():
    """
    Increasing the HP corner frequency should reduce the output for a
    given input frequency (more aggressive high-pass → less passes through).
    """
    params_low  = PAPER_PARAMS.copy()
    params_high = PAPER_PARAMS.copy()
    params_high[3] = 5.0   # hp2_x: raise from 0.7 to 5.0 rad/s

    w_low  = make_washout(params_low)
    w_high = make_washout(params_high)

    accel, omega = sine_accel(freq_hz=0.5, amp=2.0)  # 0.5 Hz ~ 3.14 rad/s
    r_low  = w_low._washout_output(accel, omega)
    r_high = w_high._washout_output(accel, omega)

    peak_low  = np.max(np.abs(r_low[0]))
    peak_high = np.max(np.abs(r_high[0]))

    check("Higher HP cutoff reduces output at low frequency",
          peak_high < peak_low,
          f"low_cutoff_peak={peak_low:.4e}, high_cutoff_peak={peak_high:.4e}")


# ── runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_instantiation,
        test_output_structure,
        test_zero_input_gives_zero_output,
        test_output_is_finite,
        test_hp_channels_attenuate_dc,
        test_hp_channels_pass_high_freq,
        test_tilt_coordination_from_sustained_accel,
        test_rotational_channel_passes_high_freq,
        test_rotational_channel_attenuates_dc,
        test_scaling_applied,
        test_parameter_sensitivity,
    ]

    print("\n" + "=" * 60)
    print("  WASHOUT FILTER TESTS")
    print("=" * 60)

    passed = 0
    failed = 0
    for test in tests:
        print(f"\n▶ {test.__name__}")
        try:
            test()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"     {e}")
        except Exception as e:
            failed += 1
            print(f"\033[91m  ERROR\033[0m  {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 60 + "\n")
    sys.exit(0 if failed == 0 else 1)