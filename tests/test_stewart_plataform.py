"""
test_stewart_platform.py

Tests for StewartPlatform and ThreadedSimulator classes.
Run with:  python test_stewart_platform.py
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from stewart_platform import StewartPlatform, ThreadedSimulator

# ── helpers ─────────────────────────────────────────────────────────────────

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"

def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"{status}  {name}{suffix}")
    if not condition:
        raise AssertionError(f"Test failed: {name}")

# ── fixtures ─────────────────────────────────────────────────────────────────

def make_platform(kp=50.0, ki=0.0, kd=1.0):
    """
    Standard 6x6 Stewart platform with evenly spaced anchors.
    Top plate radius = 1.5 m, base plate radius = 1.75 m.
    Anchors are placed at angles 0°, 60°, 120°, 180°, 240°, 300°.
    """
    rp = 1.5
    rb = 1.75
    angles_top  = np.radians([0, 60, 120, 180, 240, 300])
    angles_base = np.radians([0, 60, 120, 180, 240, 300])

    top_anchors  = np.array([[rp * np.cos(a), rp * np.sin(a), 0.0] for a in angles_top])
    base_anchors = np.array([[rb * np.cos(a), rb * np.sin(a), 0.0] for a in angles_base])

    return StewartPlatform(base_anchors, top_anchors, kp=kp, ki=ki, kd=kd)


# ── tests ────────────────────────────────────────────────────────────────────

def test_instantiation():
    """Platform can be instantiated without errors."""
    p = make_platform()
    check("Instantiation", p is not None)
    check("current_pose is 6-DOF", len(p.current_pose) == 6)
    check("target_pose is 6-DOF",  len(p.target_pose) == 6)
    check("current_lengths has 6 elements", len(p.current_lengths) == 6)
    check("actuator_minimum_lengths shape", p.actuator_minimum_lengths.shape == (6,))
    check("actuator_maximum_lengths shape", p.actuator_maximum_lengths.shape == (6,))


def test_ik_zero_pose_positive_lengths():
    """
    At the neutral (zero) pose all leg lengths must be positive.
    """
    p = make_platform()
    lengths = p._calculate_ik(np.zeros(6))
    check("All neutral leg lengths > 0",
          np.all(lengths > 0),
          f"min={np.min(lengths):.4f} m")


def test_ik_symmetry():
    """
    At the neutral pose all 6 leg lengths should be equal
    (symmetric platform geometry).
    """
    p = make_platform()
    lengths = p._calculate_ik(np.zeros(6))
    check("All neutral leg lengths equal",
          np.allclose(lengths, lengths[0], atol=1e-6),
          f"std={np.std(lengths):.2e}")


def test_ik_changes_with_pose():
    """
    Moving the platform (translation) must change the leg lengths.
    """
    p = make_platform()
    l0 = p._calculate_ik(np.zeros(6))
    l1 = p._calculate_ik(np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]))  # 1 cm in x
    check("IK changes when pose changes",
          not np.allclose(l0, l1),
          f"max_diff={np.max(np.abs(l1 - l0)):.4f}")


def test_ik_fk_roundtrip():
    """
    IK → FK round-trip: set a pose, run IK to get lengths, then run FK
    to recover the pose. The recovered pose should match the original.
    """
    p = make_platform()
    target = np.array([0.5, 0.1, 0.02, 0.02, 0.02, 0.01])  # small motion

    # Set leg lengths from IK
    p.current_lengths = p._calculate_ik(target)
    # Provide FK a good starting guess close to the answer
    p.current_pose = target + np.random.randn(6) * 1e-4

    recovered = p.get_forward_kinematics()
    err = np.max(np.abs(recovered - target))
    check("FK recovers IK pose (round-trip)",
          err < 1e-1,
          f"max_error={err:.2e}")


def test_set_target_pose():
    """
    set_target_pose stores the pose and updates target_lengths via IK.
    """
    p = make_platform()
    p.set_target_pose(0.01, 0.0, 0.0, 0.0, 0.0, 0.0)
    check("target_pose updated", np.allclose(p.target_pose[:3], [0.01, 0.0, 0.0]))
    expected_len = p._calculate_ik(p.target_pose)
    check("target_lengths updated by IK",
          np.allclose(p.target_lengths, expected_len, atol=1e-8))


def test_control_step_moves_toward_target():
    """
    After several control steps the current_lengths should move
    closer to the target_lengths.
    """
    p = make_platform(kp=50.0, ki=0.5, kd=0.10)
    p.set_target_pose(0.1, 0.05, 0.03, 0.0, 0.0, 0.0)

    initial_error = np.linalg.norm(p.target_lengths - p.current_lengths)

    dt = 0.001
    for _ in range(200):
        p.control_step(dt)

    final_error = np.linalg.norm(p.target_lengths - p.current_lengths)
    check("Control step reduces length error",
          final_error < initial_error,
          f"initial={initial_error:.4f}, final={final_error:.4f}")


def test_actuator_limits_respected():
    """
    After control steps, leg lengths must stay within [min, max].
    """
    p = make_platform(kp=500.0)  # aggressive gain to stress limits
    # Request an extreme pose that would exceed limits
    p.set_target_pose(10.5, 2.5, 2.5, 2.5, 2.5, 2.5)

    dt = 0.01
    for _ in range(100):
        p.control_step(dt)

    within_min = np.all(p.current_lengths >= p.actuator_minimum_lengths - 1e-9)
    within_max = np.all(p.current_lengths <= p.actuator_maximum_lengths + 1e-9)
    check("Leg lengths >= minimum",
          within_min,
          f"min_len={np.min(p.current_lengths):.4f}")
    check("Leg lengths <= maximum",
          within_max,
          f"max_len={np.max(p.current_lengths):.4f}")


def test_control_step_returns_correct_shapes():
    """
    control_step must return (pose[6], lengths[6]).
    """
    p = make_platform()
    p.set_target_pose(0.005, 0.0, 0.0, 0.0, 0.0, 0.0)
    pose, lengths = p.control_step(0.001)
    check("Returned pose has 6 elements",   len(pose) == 6)
    check("Returned lengths has 6 elements", len(lengths) == 6)


def test_threaded_simulator_start_stop():
    """
    ThreadedSimulator starts and stops cleanly.
    """
    p   = make_platform()
    sim = ThreadedSimulator(p, dt=0.001)
    sim.start()
    check("Thread running after start", sim.running and sim.thread.is_alive())
    time.sleep(0.05)
    sim.stop()
    check("Thread stopped after stop", not sim.running)


def test_threaded_simulator_update_and_read():
    """
    Updating target pose from main thread and reading current pose
    must not raise errors, and the platform must move toward the target.
    """
    p   = make_platform(kp=50.0)
    sim = ThreadedSimulator(p, dt=0.001)
    sim.start()

    sim.update_target_pose(0.01, 0.005, 0.0, 0.0, 0.0, 0.0)
    time.sleep(0.3)  # let the controller run

    pose    = sim.get_current_pose()
    lengths = sim.get_current_lengths()
    sim.stop()

    check("get_current_pose returns 6-DOF array", len(pose) == 6)
    check("get_current_lengths returns 6 values", len(lengths) == 6)
    check("Pose values are finite", np.all(np.isfinite(pose)))
    check("Lengths are positive",   np.all(lengths > 0))


def test_threaded_simulator_thread_safety():
    """
    Rapidly updating the target while the simulation runs must not
    cause data corruption (lengths must remain finite and positive).
    """
    p   = make_platform(kp=50.0)
    sim = ThreadedSimulator(p, dt=0.001)
    sim.start()

    for i in range(20):
        sim.update_target_pose(i * 0.001, 0.0, 0.0, 0.0, 0.0, 0.0)
        time.sleep(0.01)

    lengths = sim.get_current_lengths()
    sim.stop()

    check("Lengths finite after rapid updates", np.all(np.isfinite(lengths)))
    check("Lengths positive after rapid updates", np.all(lengths > 0))


# ── runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_instantiation,
        test_ik_zero_pose_positive_lengths,
        test_ik_symmetry,
        test_ik_changes_with_pose,
        test_ik_fk_roundtrip,
        test_set_target_pose,
        test_control_step_moves_toward_target,
        test_actuator_limits_respected,
        test_control_step_returns_correct_shapes,
        test_threaded_simulator_start_stop,
        test_threaded_simulator_update_and_read,
        test_threaded_simulator_thread_safety,
    ]

    print("\n" + "=" * 60)
    print("  STEWART PLATFORM TESTS")
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