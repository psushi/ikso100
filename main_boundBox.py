import time

import mujoco

import mujoco.viewer 

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

# 3D Workspace boundaries for the end effector (in meters)
# Adjust these values based on your robot's workspace
WORKSPACE_BOUNDS = {
    'x_min': -0.3,   # Left limit
    'x_max': 0.3,    # Right limit
    'y_min': -0.3,   # Back limit (away from base)
    'y_max': 0.3,    # Front limit (towards base)
    'z_min': 0.05,   # Bottom limit (above ground)
    'z_max': 0.6     # Top limit
}

def check_workspace_boundaries(target_pos: np.ndarray) -> np.ndarray:
    """
    Check if target position is within workspace boundaries and clamp if necessary.
    
    Args:
        target_pos: 3D target position [x, y, z]
        
    Returns:
        Clamped target position within workspace boundaries
    """
    clamped_pos = target_pos.copy()
    
    # Clamp X coordinate
    if clamped_pos[0] < WORKSPACE_BOUNDS['x_min']:
        clamped_pos[0] = WORKSPACE_BOUNDS['x_min']
        print(f"Warning: Target X clamped to minimum boundary: {WORKSPACE_BOUNDS['x_min']:.3f}")
    elif clamped_pos[0] > WORKSPACE_BOUNDS['x_max']:
        clamped_pos[0] = WORKSPACE_BOUNDS['x_max']
        print(f"Warning: Target X clamped to maximum boundary: {WORKSPACE_BOUNDS['x_max']:.3f}")
    
    # Clamp Y coordinate
    if clamped_pos[1] < WORKSPACE_BOUNDS['y_min']:
        clamped_pos[1] = WORKSPACE_BOUNDS['y_min']
        print(f"Warning: Target Y clamped to minimum boundary: {WORKSPACE_BOUNDS['y_min']:.3f}")
    elif clamped_pos[1] > WORKSPACE_BOUNDS['y_max']:
        clamped_pos[1] = WORKSPACE_BOUNDS['y_max']
        print(f"Warning: Target Y clamped to maximum boundary: {WORKSPACE_BOUNDS['y_max']:.3f}")
    
    # Clamp Z coordinate
    if clamped_pos[2] < WORKSPACE_BOUNDS['z_min']:
        clamped_pos[2] = WORKSPACE_BOUNDS['z_min']
        print(f"Warning: Target Z clamped to minimum boundary: {WORKSPACE_BOUNDS['z_min']:.3f}")
    elif clamped_pos[2] > WORKSPACE_BOUNDS['z_max']:
        clamped_pos[2] = WORKSPACE_BOUNDS['z_max']
        print(f"Warning: Target Z clamped to maximum boundary: {WORKSPACE_BOUNDS['z_max']:.3f}")
    
    return clamped_pos


def add_boundary_visualization(viewer):
    """
    Add red wireframe visualization of the workspace boundaries.
    
    Args:
        viewer: MuJoCo viewer instance
    """
    # Get boundary coordinates
    x_min, x_max = WORKSPACE_BOUNDS['x_min'], WORKSPACE_BOUNDS['x_max']
    y_min, y_max = WORKSPACE_BOUNDS['y_min'], WORKSPACE_BOUNDS['y_max']
    z_min, z_max = WORKSPACE_BOUNDS['z_min'], WORKSPACE_BOUNDS['z_max']
    
    # Define the 8 corners of the bounding box
    corners = np.array([
        [x_min, y_min, z_min],  # 0: bottom-back-left
        [x_max, y_min, z_min],  # 1: bottom-back-right
        [x_max, y_max, z_min],  # 2: bottom-front-right
        [x_min, y_max, z_min],  # 3: bottom-front-left
        [x_min, y_min, z_max],  # 4: top-back-left
        [x_max, y_min, z_max],  # 5: top-back-right
        [x_max, y_max, z_max],  # 6: top-front-right
        [x_min, y_max, z_max],  # 7: top-front-left
    ])
    
    # Define the 12 edges of the bounding box (pairs of corner indices)
    edges = [
        # Bottom face edges
        (0, 1), (1, 2), (2, 3), (3, 0),
        # Top face edges
        (4, 5), (5, 6), (6, 7), (7, 4),
        # Vertical edges
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    
    # Add each edge as a line visualization using the correct method
    for i, (start_idx, end_idx) in enumerate(edges):
        start_pos = corners[start_idx]
        end_pos = corners[end_idx]
        
        # Create a new geometry object for the line
        if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
            geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
            
            # Initialize the geometry as a line
            mujoco.mjv_initGeom(
                geom,
                type=mujoco.mjtGeom.mjGEOM_LINE,
                size=np.array([0.002, 0, 0]),  # Line width
                pos=(start_pos + end_pos) / 2,  # Midpoint
                mat=np.eye(3).flatten(),
                rgba=np.array([1.0, 0.0, 0.0, 1.0])  # Red color
            )
            
            # Set the line endpoints using connector
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_LINE,
                0.002,  # Line width
                start_pos.astype(np.float64),
                end_pos.astype(np.float64)
            )
            
            viewer.user_scn.ngeom += 1


def main():
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

    key_id = model.key("home").id

    mocap_id = model.body("target").mocapid[0]

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

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mj_resetDataKeyframe(model, data, key_id)

        # Initialize the camera view to that of the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        print("="*60)
        print("WORKSPACE BOUNDARIES:")
        print(f"X: [{WORKSPACE_BOUNDS['x_min']:.2f}, {WORKSPACE_BOUNDS['x_max']:.2f}] meters")
        print(f"Y: [{WORKSPACE_BOUNDS['y_min']:.2f}, {WORKSPACE_BOUNDS['y_max']:.2f}] meters") 
        print(f"Z: [{WORKSPACE_BOUNDS['z_min']:.2f}, {WORKSPACE_BOUNDS['z_max']:.2f}] meters")
        print("Red wireframe box shows the workspace boundaries in the viewer.")
        print("="*60)

        while viewer.is_running():
            step_start = time.time()
            
            # Clear previous visualization elements
            viewer.user_scn.ngeom = 0
            
            # Add boundary visualization (red wireframe box)
            add_boundary_visualization(viewer)
            
            # Example: Uncomment this line to test with circular motion
            data.mocap_pos[mocap_id, 0:2] = circle(data.time, 0.5, 0.5, 0.1, 0.1)
            
            # IMPORTANT: When integrating with OpenCV/ArUco detection,
            # add the boundary check right after updating mocap_pos from OpenCV:
            # 
            # Example integration point:
            # if opencv_detected_new_position:
            #     data.mocap_pos[mocap_id] = opencv_target_position  # From ArUco detection
            #     data.mocap_pos[mocap_id] = check_workspace_boundaries(data.mocap_pos[mocap_id])
            
            # Apply boundary check to current target position
            data.mocap_pos[mocap_id] = check_workspace_boundaries(data.mocap_pos[mocap_id])

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
            np.clip(q, *model.jnt_range.T, out=q)
            data.ctrl[actuator_ids] = q[dof_ids]

            # Step the simulation.
            mujoco.mj_step(model, data)

            viewer.sync()

            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
