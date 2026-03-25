"""
test_vestibular_system.py

Tests for VestibularSystem class.
Run with:  python test_vestibular_system.py
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from vestibular_system import VestibularSystem

# ── helpers ────────────────────────────────────────────────────────────────

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"

def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"{status}  {name}{suffix}")
    if not condition:
        raise AssertionError(f"Test failed: {name}")

# ── fixtures ────────────────────────────────────────────────────────────────

DT   = 0.01          # 100 Hz
T    = 10.0          # 10 s simulation
N    = int(T / DT)
t    = np.linspace(0, T, N)

def make_system():
    return VestibularSystem(DT)

def zero_inputs():
    accel = np.zeros((3, N))
    omega = np.zeros((3, N))
    return accel, omega

def sine_inputs(freq_hz=2.0, amp_accel=1.0, amp_omega=0.5):
    """Sinusoidal signals well above the perception threshold."""
    accel = np.ones((3, N)) * amp_accel * np.sin(2 * np.pi * freq_hz * t)
    omega = np.ones((3, N)) * amp_omega * np.sin(2 * np.pi * freq_hz * t)
    return accel, omega

def step_inputs(amp_accel=1.0, amp_omega=0.5):
    """Step signals above the perception threshold."""
    accel = np.ones((3, N)) * amp_accel
    omega = np.ones((3, N)) * amp_omega
    return accel, omega

# ── tests ───────────────────────────────────────────────────────────────────

def test_instantiation():
    """VestibularSystem can be created without errors."""
    vs = make_system()
    check("Instantiation", vs is not None)
    check("dt stored correctly", vs.dt == DT)
    check("6 thresholds defined", len(vs.thresholds) == 6)
    check("3 accel parameter rows", len(vs.accel_param) == 3)
    check("3 angular parameter rows", len(vs.angular_param) == 3)


def test_output_structure():
    """Output is a list of [accel_output(3), ang_output(3)], each array length N."""
    vs = make_system()
    accel, omega = sine_inputs()
    result = vs._vestibular_output(accel, omega)

    check("Output is a list of 2", isinstance(result, list) and len(result) == 2)

    accel_out, ang_out = result
    check("Accel output has 3 channels", len(accel_out) == 3)
    check("Angular output has 3 channels", len(ang_out) == 3)

    for i in range(3):
        check(f"Accel channel {i} length == N",  len(accel_out[i]) == N)
        check(f"Angular channel {i} length == N", len(ang_out[i]) == N)


def test_zero_input_gives_zero_output():
    """Zero inputs must produce zero outputs (below threshold)."""
    vs = make_system()
    accel, omega = zero_inputs()
    accel_out, ang_out = vs._vestibular_output(accel, omega)

    for i in range(3):
        check(f"Zero accel → zero output channel {i}",
              np.allclose(accel_out[i], 0.0),
              f"max={np.max(np.abs(accel_out[i])):.2e}")
        check(f"Zero omega → zero output channel {i}",
              np.allclose(ang_out[i], 0.0),
              f"max={np.max(np.abs(ang_out[i])):.2e}")


def test_threshold_suppresses_small_signals():
    """
    Signals below threshold (ax < 0.17, roll < 2 deg/s) must produce zero output.
    """
    vs = make_system()
    # Slightly below threshold
    accel = np.ones((3, N)) * 0.10   # threshold is 0.17 m/s²
    omega = np.ones((3, N)) * 0.01   # threshold is 2 deg/s = 0.035 rad/s
    accel_out, ang_out = vs._vestibular_output(accel, omega)

    for i in range(3):
        check(f"Sub-threshold accel suppressed ch{i}",
              np.allclose(accel_out[i], 0.0),
              f"max={np.max(np.abs(accel_out[i])):.2e}")
        check(f"Sub-threshold omega suppressed ch{i}",
              np.allclose(ang_out[i], 0.0),
              f"max={np.max(np.abs(ang_out[i])):.2e}")


def test_above_threshold_produces_nonzero_output():
    """
    Signals well above threshold must produce non-zero output.
    """
    vs = make_system()
    accel, omega = sine_inputs(freq_hz=2.0, amp_accel=2.0, amp_omega=1.0)
    accel_out, ang_out = vs._vestibular_output(accel, omega)

    for i in range(3):
        check(f"Above-threshold accel produces output ch{i}",
              np.max(np.abs(accel_out[i])) > 0,
              f"max={np.max(np.abs(accel_out[i])):.4f}")
        check(f"Above-threshold omega produces output ch{i}",
              np.max(np.abs(ang_out[i])) > 0,
              f"max={np.max(np.abs(ang_out[i])):.4f}")


def test_output_is_finite():
    """No NaN or Inf values in the output."""
    vs = make_system()
    accel, omega = sine_inputs()
    accel_out, ang_out = vs._vestibular_output(accel, omega)

    for i in range(3):
        check(f"Accel output finite ch{i}",  np.all(np.isfinite(accel_out[i])))
        check(f"Angular output finite ch{i}", np.all(np.isfinite(ang_out[i])))


def test_output_bounded_by_input():
    """
    The vestibular filters are low-gain — the peak output should not
    greatly exceed the peak input (a sanity check, not a strict bound).
    """
    vs = make_system()
    amp = 5.0
    accel, omega = sine_inputs(amp_accel=amp, amp_omega=amp)
    accel_out, ang_out = vs._vestibular_output(accel, omega)

    for i in range(3):
        peak_a = np.max(np.abs(accel_out[i]))
        peak_o = np.max(np.abs(ang_out[i]))
        # Allow 10× headroom — just checking no runaway gain
        check(f"Accel output not wildly amplified ch{i}",
              peak_a < 10 * amp, f"peak={peak_a:.2f}")
        check(f"Angular output not wildly amplified ch{i}",
              peak_o < 10 * amp, f"peak={peak_o:.2f}")


def test_step_response_settles():
    """
    A step input should produce a transient response that decays over time
    (the high-pass nature of the otolith model washes out DC).
    """
    vs = make_system()
    # Long simulation to see washout
    T_long = 60.0
    N_long = int(T_long / DT)
    accel_step = np.ones((3, N_long)) * 2.0   # 2 m/s², above threshold
    omega_step = np.zeros((3, N_long))

    accel_out, _ = vs._vestibular_output(accel_step, omega_step)

    for i in range(3):
        early = np.max(np.abs(accel_out[i][:100]))   # first second
        late  = np.max(np.abs(accel_out[i][-100:]))  # last second
        check(f"Step response decays over time ch{i}",
              late < early,
              f"early_peak={early:.4f}, late_peak={late:.4f}")


# ── runner ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_instantiation,
        test_output_structure,
        test_zero_input_gives_zero_output,
        test_threshold_suppresses_small_signals,
        test_above_threshold_produces_nonzero_output,
        test_output_is_finite,
        test_output_bounded_by_input,
        test_step_response_settles,
    ]

    print("\n" + "=" * 60)
    print("  VESTIBULAR SYSTEM TESTS")
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