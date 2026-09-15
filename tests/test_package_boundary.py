def test_legacy_solver_import_is_same_adapter():
    from proj_xrt.g1_adapter import G1FullBodySEWSolver
    from projects.g1_full_body_kinematic.g1_full_body_sew_solver import G1FullBodySEWSolver as legacy
    assert legacy is G1FullBodySEWSolver

def test_cli_import_has_no_application_side_effects():
    import subprocess
    import sys
    subprocess.run([sys.executable, '-c',
        'import sys; import proj_xrt.cli; '
        'assert "projects.inspire_hand_teleop.XRT_user_study_manager" not in sys.modules; '
        'assert "unitree_sdk2py" not in sys.modules'], check=True)
