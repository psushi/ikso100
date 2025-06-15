import time
from queue import Empty, Queue

import mujoco
import numpy as np

integration_dt: float = 1.0
# Whether to enable gravity compensation.
gravity_compensation: bool = True

# Damping term for the pseudoinverse. This is used to prevent joint velocities from
# becoming too large when the Jacobian is close to singular.
damping: float = 1e-4

# Simulation timestep in seconds.
dt: float = 0.002

# Maximum allowable joint velocity in rad/s. Set to 0 to disable.
max_angvel = 0.0


def sim_loop(queue: Queue | None = None):
    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)

    # Override the default timestep.
    model.opt.timestep = dt

    end_effector = model.site("attachment_site").id

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

    # Get jaw joint ID for manual control
    jaw_dof_id = model.joint("Jaw").id
    jaw_actuator_id = model.actuator("Jaw").id
    key_id = model.key("home-scene").id

    mocap_id = model.body("target").mocapid[0]

    # Gripper control variables
    gripper_target = 0.0

    # Pre-allocate numpy arrays.
    jac = np.zeros((6, model.nv))
    diag = damping * np.eye(6)
    error = np.zeros(6)
    error_pos = error[:3]
    error_ori = error[3:]
    ee_quat = np.zeros(4)
    ee_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)

    # Define a trajectory for the end-effector site to follow.
    def circle(t: float, r: float, h: float, k: float, f: float) -> np.ndarray:
        """Return the (x, y) coordinates of a circle with radius r centered at (h, k)
        as a function of time t and frequency f."""
        x = r * np.cos(2 * np.pi * f * t) + h
        y = r * np.sin(2 * np.pi * f * t) + k
        return np.array([x, y])

    # Keyboard callback function
    def key_callback(key):
        nonlocal gripper_target
        if key == 265:  # UP arrow key
            gripper_target = max(gripper_target - 0.2, -0.2)  # Open gripper
        elif key == 264:  # DOWN arrow key
            gripper_target = min(gripper_target + 0.2, 2.0)  # Close gripper

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
        key_callback=key_callback,
    ) as viewer:
        mujoco.mj_resetDataKeyframe(model, data, key_id)

        # Initialize the camera view to that of the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Position camera behind the robot
        viewer.cam.distance = 0.6  # Even closer to target
        viewer.cam.azimuth = 270  # Behind the robot (270 degrees)
        viewer.cam.elevation = -25  # Higher elevation, looking down more

        # Enable body axes visualization
        # viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY

        while viewer.is_running():
            step_start = time.time()
            # data.mocap_pos[mocap_id, 0:2] = circle(data.time, 0.5, 0.5, 0.1, 0.1)

            # Set gripper control from keyboard callback
            data.ctrl[jaw_actuator_id] = gripper_target

            if queue is not None:
                try:
                    latest_pose = queue.get_nowait()
                    data.mocap_pos[mocap_id] += latest_pose.pos
                    # Skip quaternion updates for now
                    # data.mocap_quat[mocap_id] = latest_pose.quat
                except Empty:
                    pass

            error_pos[:] = data.mocap_pos[mocap_id] - data.site(end_effector).xpos

            mujoco.mju_mat2Quat(ee_quat, data.site(end_effector).xmat)
            mujoco.mju_negQuat(ee_quat_conj, ee_quat)
            mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], ee_quat_conj)
            mujoco.mju_quat2Vel(error_ori, error_quat, 1.0)

            mujoco.mj_jacSite(model, data, jac[:3], jac[3:], end_effector)

            # Solve system of equations: J @ dq = error.
            dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, error)

            # scale down joint velocities if they exceed maximum.
            if max_angvel > 0:
                dq_abs_max = np.abs(dq).max()
                if dq_abs_max > max_angvel:
                    dq *= max_angvel / dq_abs_max

            # Integrate joint velocities to obtain joint positions.
            q = data.qpos.copy()
            mujoco.mj_integratePos(model, q, dq, integration_dt)

            # Set the control signal.
            # Only clip the actuated joints (robot joints), not the free cube joints
            q_robot = q[dof_ids]
            # Get joint ranges for only the actuated joints
            robot_joint_ranges = model.jnt_range[dof_ids]
            np.clip(q_robot, robot_joint_ranges[:, 0], robot_joint_ranges[:, 1], out=q_robot)
            q[dof_ids] = q_robot
            data.ctrl[actuator_ids] = q_robot

            # Step the simulation.
            mujoco.mj_step(model, data)

            viewer.sync()

            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    sim_loop()
