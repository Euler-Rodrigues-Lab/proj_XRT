"""Lazy application entry points; importing this module starts no hardware."""

def teleop():
    from projects.inspire_hand_teleop.demo_g1_inspire_xr_robot_teleop_full_body_hw import main
    return main()

def study():
    from projects.inspire_hand_teleop.XRT_user_study_manager import main
    return main()
