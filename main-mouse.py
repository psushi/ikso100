import time
import mujoco
import mujoco.viewer
import numpy as np

# Simulation parameters
integration_dt: float = 1.0
gravity_compensation: bool = True
damping: float = 1e-4
dt: float = 0.002
max_angvel = 0.0

# Control parameters
MOUSE_SENSITIVITY = 0.001  # How sensitive mouse movement is
SCROLL_SENSITIVITY = 0.05  # How sensitive scroll wheel is for Z movement

def main():
    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)

    # Override the default timestep
    model.opt.timestep = dt

    end_effector = model.site("attachment_site").id
    target_id = model.body("target").mocapid[0]

    body_names = [
        "Rotation_Pitch",
        "Upper_Arm",
        "Lower_Arm",
        "Wrist_Pitch_Roll",
        "Fixed_Jaw",
        "Moving_Jaw",
    ]

    if gravity_compensation:
        body_ids = [model.body(name).id for name in body_names]
        model.body_gravcomp[body_ids] = 1.0

    joint_names = [
        "Rotation",
        "Pitch",
        "Elbow",
        "Wrist_Pitch",
        "Wrist_Roll",
        "Jaw",
    ]

    dof_ids = [model.joint(name).id for name in joint_names]
    actuator_ids = [model.actuator(name).id for name in joint_names]

    key_id = model.key("home").id

    # Pre-allocate numpy arrays
    jac = np.zeros((6, model.nv))
    diag = damping * np.eye(6)
    error = np.zeros(6)
    error_pos = error[:3]
    error_ori = error[3:]
    ee_quat = np.zeros(4)
    ee_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)

    # Mouse control variables
    last_mouse_x = 0
    last_mouse_y = 0
    mouse_button_pressed = False

    def mouse_button_callback(button, action):
        nonlocal mouse_button_pressed
        if button == 0:  # Left mouse button
            mouse_button_pressed = (action == 1)  # 1 = press, 0 = release
            print(f"Mouse button {'pressed' if mouse_button_pressed else 'released'}")
        return True

    def mouse_move_callback(x, y):
        nonlocal last_mouse_x, last_mouse_y
        
        if mouse_button_pressed:
            # Calculate mouse movement
            dx = x - last_mouse_x
            dy = y - last_mouse_y
            
            # Get current target position
            pos = data.mocap_pos[target_id].copy()
            
            # Update position based on mouse movement
            # X movement controls target X position
            # Y movement controls target Y position
            pos[0] += dx * MOUSE_SENSITIVITY
            pos[1] -= dy * MOUSE_SENSITIVITY  # Invert Y for intuitive control
            
            # Update target position
            data.mocap_pos[target_id] = pos
            print(f"Mouse moved target to: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        
        last_mouse_x = x
        last_mouse_y = y
        return True

    def scroll_callback(x_offset, y_offset):
        # Get current target position
        pos = data.mocap_pos[target_id].copy()
        
        # Update Z position based on scroll
        pos[2] += y_offset * SCROLL_SENSITIVITY
        
        # Update target position
        data.mocap_pos[target_id] = pos
        print(f"Scroll moved target to: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        return True

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        # Set up camera
        viewer.cam.distance = 1.0
        viewer.cam.lookat = np.array([0, 0, 0.2])
        viewer.cam.elevation = -30
        viewer.cam.azimuth = 140

        # Set up mouse callbacks
        viewer.mouse_button_callback = mouse_button_callback
        viewer.mouse_move_callback = mouse_move_callback
        viewer.scroll_callback = scroll_callback

        mujoco.mj_resetDataKeyframe(model, data, key_id)

        print("\n" + "="*60)
        print("MOUSE CONTROL INSTRUCTIONS:")
        print("="*60)
        print("1. Click and hold LEFT MOUSE BUTTON on the simulation window")
        print("2. Drag the mouse to move the target in X-Y plane:")
        print("   - Left/Right: Move target left/right (X axis)")
        print("   - Up/Down: Move target forward/backward (Y axis)")
        print("3. Use SCROLL WHEEL to move target up/down (Z axis)")
        print("4. Watch the red target box move!")
        print("5. The arm should follow the target using inverse kinematics")
        print("6. Release mouse button to stop moving")
        print("="*60)
        print(f"Initial target position: {data.mocap_pos[target_id]}")
        print("="*60)

        while viewer.is_running():
            step_start = time.time()

            # Calculate error between target and end-effector
            error_pos[:] = data.mocap_pos[target_id] - data.site(end_effector).xpos

            mujoco.mju_mat2Quat(ee_quat, data.site(end_effector).xmat)
            mujoco.mju_negQuat(ee_quat_conj, ee_quat)
            mujoco.mju_mulQuat(error_quat, data.mocap_quat[target_id], ee_quat_conj)
            mujoco.mju_quat2Vel(error_ori, error_quat, 1.0)

            # Calculate Jacobian
            mujoco.mj_jacSite(model, data, jac[:3], jac[3:], end_effector)

            # Solve system of equations: J @ dq = error
            dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, error)

            # Scale down joint velocities if they exceed maximum
            if max_angvel > 0:
                dq_abs_max = np.abs(dq).max()
                if dq_abs_max > max_angvel:
                    dq *= max_angvel / dq_abs_max

            # Integrate joint velocities to obtain joint positions
            q = data.qpos.copy()
            mujoco.mj_integratePos(model, q, dq, integration_dt)

            # Set the control signal
            np.clip(q, *model.jnt_range.T, out=q)
            data.ctrl[actuator_ids] = q[dof_ids]

            # Step the simulation
            mujoco.mj_step(model, data)

            viewer.sync()

            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()