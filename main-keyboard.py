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
MOVE_SPEED = 0.05  # Increased speed for more noticeable movement
ROT_SPEED = 0.3    # Increased rotation speed

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

    def handle_key(key):
        print(f"Key pressed: {key}")  # Debug output
        
        # Get current target position and orientation
        pos = data.mocap_pos[target_id].copy()
        quat = data.mocap_quat[target_id].copy()
        
        old_pos = pos.copy()  # For debug output
        
        # Position control using ASCII values for number keys
        if key == 49:  # '1' key - Forward
            pos[1] += MOVE_SPEED
            print(f"Moving forward: Y {old_pos[1]:.3f} -> {pos[1]:.3f}")
        elif key == 50:  # '2' key - Backward
            pos[1] -= MOVE_SPEED
            print(f"Moving backward: Y {old_pos[1]:.3f} -> {pos[1]:.3f}")
        elif key == 51:  # '3' key - Left
            pos[0] -= MOVE_SPEED
            print(f"Moving left: X {old_pos[0]:.3f} -> {pos[0]:.3f}")
        elif key == 52:  # '4' key - Right
            pos[0] += MOVE_SPEED
            print(f"Moving right: X {old_pos[0]:.3f} -> {pos[0]:.3f}")
        elif key == 53:  # '5' key - Up
            pos[2] += MOVE_SPEED
            print(f"Moving up: Z {old_pos[2]:.3f} -> {pos[2]:.3f}")
        elif key == 54:  # '6' key - Down
            pos[2] -= MOVE_SPEED
            print(f"Moving down: Z {old_pos[2]:.3f} -> {pos[2]:.3f}")
        
        # Orientation control using arrow keys (GLFW key codes)
        elif key == 265:  # Up arrow - Pitch up
            quat = quat * np.array([np.cos(ROT_SPEED/2), np.sin(ROT_SPEED/2), 0, 0])
            print("Pitching up")
        elif key == 264:  # Down arrow - Pitch down
            quat = quat * np.array([np.cos(-ROT_SPEED/2), np.sin(-ROT_SPEED/2), 0, 0])
            print("Pitching down")
        elif key == 263:  # Left arrow - Yaw left
            quat = quat * np.array([np.cos(ROT_SPEED/2), 0, 0, np.sin(ROT_SPEED/2)])
            print("Yawing left")
        elif key == 262:  # Right arrow - Yaw right
            quat = quat * np.array([np.cos(-ROT_SPEED/2), 0, 0, np.sin(-ROT_SPEED/2)])
            print("Yawing right")
        elif key == 266:  # Page Up - Roll left
            quat = quat * np.array([np.cos(ROT_SPEED/2), 0, np.sin(ROT_SPEED/2), 0])
            print("Rolling left")
        elif key == 267:  # Page Down - Roll right
            quat = quat * np.array([np.cos(-ROT_SPEED/2), 0, np.sin(-ROT_SPEED/2), 0])
            print("Rolling right")
        else:
            print(f"Unhandled key: {key}")
            return True
        
        # Update target position and orientation
        data.mocap_pos[target_id] = pos
        data.mocap_quat[target_id] = quat
        print(f"Target position updated to: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        return True

    def key_callback(key):
        return handle_key(key)

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        # Set up camera
        viewer.cam.distance = 1.0
        viewer.cam.lookat = np.array([0, 0, 0.2])
        viewer.cam.elevation = -30
        viewer.cam.azimuth = 140

        # Set up key callback
        viewer.key_callback = key_callback

        mujoco.mj_resetDataKeyframe(model, data, key_id)

        print("\nKeyboard Controls:")
        print("Position Control (Number Keys):")
        print("  1: Move forward (Y+)")
        print("  2: Move backward (Y-)")
        print("  3: Move left (X-)")
        print("  4: Move right (X+)")
        print("  5: Move up (Z+)")
        print("  6: Move down (Z-)")
        print("\nOrientation Control (Arrow Keys):")
        print("  ↑: Pitch up")
        print("  ↓: Pitch down")
        print("  ←: Yaw left")
        print("  →: Yaw right")
        print("  PgUp: Roll left")
        print("  PgDn: Roll right")
        print("\nPress ESC to exit")
        print(f"\nInitial target position: {data.mocap_pos[target_id]}")

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