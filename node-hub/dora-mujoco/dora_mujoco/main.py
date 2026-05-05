"""MuJoCo simulation node for Dora with robot descriptions support."""

from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import time
from typing import Dict, Any

import mujoco
import mujoco.viewer
import numpy as np
import pyarrow as pa
from dora import Node
from robot_descriptions.loaders.mujoco import load_robot_description


class MuJoCoSimulator:
    """MuJoCo simulator for Dora."""

    def __init__(self, model_path_or_name: str = None):
        """Initialize the MuJoCo simulator."""
        # Check environment variable first, then use parameter, then default
        self.model_path_or_name = (
            os.getenv("MODEL_NAME") or 
            model_path_or_name or 
            "go2_mj_description"
        )
        
        self.model = None
        self.data = None
        self.viewer = None
        self.renderer = None
        self.camera_name = os.getenv("CAMERA_NAME", "")
        self.image_width = int(os.getenv("IMAGE_WIDTH", "640"))
        self.image_height = int(os.getenv("IMAGE_HEIGHT", "480"))
        self.camera_intrinsics = None
        self.sim_kp = float(os.getenv("SIM_KP", "45"))
        self.sim_kd = float(os.getenv("SIM_KD", "4"))
        self.gripper_open_rad = float(os.getenv("GRIPPER_OPEN_RAD", "-1.0472"))
        self.gripper_open_slide = float(os.getenv("GRIPPER_OPEN_SLIDE", "0.044"))
        self.state_data = {}
        self.arm_joint_names = {
            "left": [f"openarm_left_joint{i}" for i in range(1, 8)],
            "right": [f"openarm_right_joint{i}" for i in range(1, 8)],
        }
        self.arm_actuator_names = {
            "left": [f"left_joint{i}_ctrl" for i in range(1, 8)],
            "right": [f"right_joint{i}_ctrl" for i in range(1, 8)],
        }
        self.gripper_actuator_names = {
            "left": ["left_finger1_ctrl", "left_finger2_ctrl"],
            "right": ["right_finger1_ctrl", "right_finger2_ctrl"],
        }
        self.arm_targets = {}
        self.gripper_targets = {}
        self.load_model()

        print(f"MuJoCo Simulator initialized with model: {self.model_path_or_name}")

    def load_model(self) -> bool:
        """Load MuJoCo model from path or robot description name."""
        model_path = Path(self.model_path_or_name)
        if model_path.exists() and model_path.suffix == '.xml':
            print(f"Loading model from direct path: {model_path}")
            self.model = mujoco.MjModel.from_xml_path(str(model_path))
        else:
            self.model = load_robot_description(self.model_path_or_name, variant="scene")

        # Initialize simulation data
        self.data = mujoco.MjData(self.model)
        
        # Set control to neutral position
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        
        # Print model info for debugging
        print("Model loaded successfully:")
        print(f"  DOF (nq): {self.model.nq}")
        print(f"  Velocities (nv): {self.model.nv}")
        print(f"  Actuators (nu): {self.model.nu}")
        print(f"  Control inputs: {len(self.data.ctrl)}")
        
        # Print actuator info
        if self.model.nu > 0:
            print("Actuators:")
            for i in range(self.model.nu):
                actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                joint_id = self.model.actuator_trnid[i, 0]  # First transmission joint
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) if joint_id >= 0 else "N/A"
                ctrl_range = self.model.actuator_ctrlrange[i]
                print(f"  [{i}] {actuator_name or f'actuator_{i}'} -> joint: {joint_name} | range: [{ctrl_range[0]:.2f}, {ctrl_range[1]:.2f}]")
            
        # Initialize state data
        self._setup_control_targets()
        self._setup_camera_renderer()
        self._update_state_data()
        return True

    def _object_id(self, object_type, name: str) -> int:
        return mujoco.mj_name2id(self.model, object_type, name)

    def _joint_qpos_index(self, name: str) -> int | None:
        joint_id = self._object_id(mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            return None
        return int(self.model.jnt_qposadr[joint_id])

    def _joint_qvel_index(self, name: str) -> int | None:
        joint_id = self._object_id(mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            return None
        return int(self.model.jnt_dofadr[joint_id])

    def _actuator_index(self, name: str) -> int | None:
        actuator_id = self._object_id(mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id < 0:
            return None
        return int(actuator_id)

    def _setup_control_targets(self):
        """Initialize joint targets from the current simulated state."""
        self.arm_targets = {}
        for arm, joint_names in self.arm_joint_names.items():
            qpos_indices = [self._joint_qpos_index(name) for name in joint_names]
            if any(index is None for index in qpos_indices):
                continue
            self.arm_targets[arm] = self.data.qpos[qpos_indices].copy()

        self.gripper_targets = {}
        for arm, actuator_names in self.gripper_actuator_names.items():
            actuator_indices = [self._actuator_index(name) for name in actuator_names]
            actuator_indices = [index for index in actuator_indices if index is not None]
            if actuator_indices:
                self.gripper_targets[arm] = 0.0

    def _setup_camera_renderer(self):
        """Create an offscreen renderer for an MJCF camera, if configured."""
        self.renderer = None
        self.camera_intrinsics = None
        if not self.camera_name:
            return

        camera_id = self._object_id(mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        if camera_id < 0:
            print(f"WARNING: CAMERA_NAME={self.camera_name!r} not found; RGBD disabled")
            return

        fovy = float(self.model.cam_fovy[camera_id])
        fy = 0.5 * self.image_height / math.tan(math.radians(fovy) * 0.5)
        fx = fy
        cx = (self.image_width - 1) * 0.5
        cy = (self.image_height - 1) * 0.5
        self.camera_intrinsics = (fx, fy, cx, cy)

        try:
            self.renderer = mujoco.Renderer(
                self.model,
                height=self.image_height,
                width=self.image_width,
            )
        except Exception as e:
            print(f"WARNING: Failed to create MuJoCo RGBD renderer: {e}")
            self.renderer = None
            self.camera_intrinsics = None
            return

        print(
            f"RGBD camera enabled: {self.camera_name} "
            f"{self.image_width}x{self.image_height} "
            f"fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}"
        )

    def apply_control(self, control_input: np.ndarray):
        """Apply control input to the simulation.

        Args:
            control_input: Control values for actuators

        """
        if control_input is None or len(control_input) == 0:
            return
            
        # Ensure we don't exceed the number of actuators
        n_controls = min(len(control_input), self.model.nu)
        
        # Apply control directly to actuators 
        for i in range(n_controls):
            # Apply joint limits if available
            ctrl_range = self.model.actuator_ctrlrange[i]
            if ctrl_range[0] < ctrl_range[1]:  # Valid range
                control_value = np.clip(control_input[i], ctrl_range[0], ctrl_range[1])
            else:
                control_value = control_input[i]
            
            self.data.ctrl[i] = control_value

    def apply_joint_command(self, target_angles: np.ndarray, arm: str = "left"):
        """Store a position target from dora-motion-planner for PD control."""
        if arm not in self.arm_targets:
            print(f"Ignoring joint_command for unavailable arm: {arm}")
            return
        target = np.asarray(target_angles, dtype=np.float64)[:7]
        if target.shape[0] < 7:
            print(f"Ignoring short joint_command for {arm}: {target.shape[0]} values")
            return
        self.arm_targets[arm] = target.copy()

    def apply_gripper_command(self, target_rad: float, arm: str = "left"):
        """Map OpenArm gripper radians to MuJoCo slide-joint position targets."""
        if arm not in self.gripper_targets:
            return
        open_fraction = np.clip(target_rad / self.gripper_open_rad, 0.0, 1.0)
        self.gripper_targets[arm] = float(open_fraction * self.gripper_open_slide)

    def _apply_position_targets(self):
        """Apply PD torques for arm joints and position targets for fingers."""
        for arm, target in self.arm_targets.items():
            for i, joint_name in enumerate(self.arm_joint_names[arm]):
                qpos_index = self._joint_qpos_index(joint_name)
                qvel_index = self._joint_qvel_index(joint_name)
                actuator_index = self._actuator_index(self.arm_actuator_names[arm][i])
                if qpos_index is None or qvel_index is None or actuator_index is None:
                    continue

                err = target[i] - self.data.qpos[qpos_index]
                vel = self.data.qvel[qvel_index]
                control = self.sim_kp * err - self.sim_kd * vel
                ctrl_range = self.model.actuator_ctrlrange[actuator_index]
                if ctrl_range[0] < ctrl_range[1]:
                    control = np.clip(control, ctrl_range[0], ctrl_range[1])
                self.data.ctrl[actuator_index] = control

        for arm, target in self.gripper_targets.items():
            for actuator_name in self.gripper_actuator_names[arm]:
                actuator_index = self._actuator_index(actuator_name)
                if actuator_index is None:
                    continue
                ctrl_range = self.model.actuator_ctrlrange[actuator_index]
                control = target
                if ctrl_range[0] < ctrl_range[1]:
                    control = np.clip(control, ctrl_range[0], ctrl_range[1])
                self.data.ctrl[actuator_index] = control

    def _get_available_models(self) -> dict:
        """Get available models from the mapping file."""
        config_path = Path(__file__).parent / "robot_models.json"
        with open(config_path) as f:
            return json.load(f)

    def step_simulation(self):
        """Step the simulation forward."""
        self._apply_position_targets()
        mujoco.mj_step(self.model, self.data)
        self._update_state_data()

    def get_arm_joint_state(self, arm: str) -> np.ndarray | None:
        """Return the first seven joint positions for a named OpenArm side."""
        joint_names = self.arm_joint_names.get(arm)
        if joint_names is None:
            return None
        qpos_indices = [self._joint_qpos_index(name) for name in joint_names]
        if any(index is None for index in qpos_indices):
            return None
        return self.data.qpos[qpos_indices].astype(np.float32).copy()

    def render_camera(self):
        """Render RGB and mono16 depth images from the configured MJCF camera."""
        if self.renderer is None or self.camera_intrinsics is None:
            return None

        mujoco.mj_forward(self.model, self.data)
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(self.data, camera=self.camera_name)
        rgb = self.renderer.render()

        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera=self.camera_name)
        depth_m = self.renderer.render()
        self.renderer.disable_depth_rendering()

        depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
        depth_mm[depth_mm > 5000] = 0
        return rgb.astype(np.uint8), depth_mm

    def close(self):
        """Release MuJoCo rendering resources."""
        renderer = self.renderer
        self.renderer = None
        if renderer is not None:
            renderer.close()
    
    def _update_state_data(self):
        """Update state data that can be sent via Dora."""
        self.state_data = {
            "time": self.data.time,                    # Current simulation time
            "qpos": self.data.qpos.copy(),            # Joint positions  
            "qvel": self.data.qvel.copy(),            # Joint velocities
            "qacc": self.data.qacc.copy(),            # Joint accelerations
            "ctrl": self.data.ctrl.copy(),            # Control inputs/actuator commands
            "qfrc_applied": self.data.qfrc_applied.copy(),  # External forces applied to joints
            "sensordata": self.data.sensordata.copy() if self.model.nsensor > 0 else np.array([])  # Sensor readings
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get current simulation state."""
        return self.state_data.copy()


def main():
    """Run the main Dora node function."""
    node = Node()
    
    # Initialize simulator
    simulator = MuJoCoSimulator()

    print("MuJoCo simulation node started")

    # Launch viewer
    show_viewer = os.getenv("SHOW_VIEWER", "true").lower() in ("1", "true", "yes")
    viewer_context = (
        mujoco.viewer.launch_passive(simulator.model, simulator.data)
        if show_viewer
        else nullcontext(None)
    )
    try:
        with viewer_context as viewer:
            if viewer is not None:
                print("MuJoCo viewer launched - simulation running")
            else:
                print("MuJoCo viewer disabled - simulation running")

            # Main Dora event loop
            for event in node:
                if event["type"] == "INPUT":
                    # Handle control input
                    event_id = event["id"]
                    metadata = event["metadata"]
                    if event_id == "control_input":
                        control_array = event["value"].to_numpy()
                        simulator.apply_control(control_array)

                    elif event_id == "joint_command":
                        arm = metadata.get("arm", "left")
                        simulator.apply_joint_command(event["value"].to_numpy(), arm=arm)

                    elif event_id == "gripper_command":
                        arm = metadata.get("arm", "left")
                        value = event["value"].to_numpy().astype(np.float32)
                        if len(value) > 0:
                            simulator.apply_gripper_command(float(value[0]), arm=arm)

                    if event_id == "tick":
                        simulator.step_simulation()
                        if viewer is not None:
                            viewer.sync()

                        state = simulator.get_state()
                        current_time = state.get("time", time.time())

                        node.send_output(
                            "joint_positions",
                            pa.array(state["qpos"]),
                            {"timestamp": current_time, "encoding": "jointstate"},
                        )
                        node.send_output(
                            "joint_velocities",
                            pa.array(state["qvel"]),
                            {"timestamp": current_time},
                        )
                        node.send_output(
                            "actuator_controls",
                            pa.array(state["ctrl"]),
                            {"timestamp": current_time},
                        )

                        left_state = simulator.get_arm_joint_state("left")
                        if left_state is not None:
                            joint_metadata = {
                                "timestamp": current_time,
                                "encoding": "jointstate",
                                "arm": "left",
                            }
                            node.send_output(
                                "joint_state",
                                pa.array(left_state, type=pa.float32()),
                                joint_metadata,
                            )
                            node.send_output(
                                "left_joint_state",
                                pa.array(left_state, type=pa.float32()),
                                joint_metadata,
                            )

                        right_state = simulator.get_arm_joint_state("right")
                        if right_state is not None:
                            node.send_output(
                                "right_joint_state",
                                pa.array(right_state, type=pa.float32()),
                                {
                                    "timestamp": current_time,
                                    "encoding": "jointstate",
                                    "arm": "right",
                                },
                            )

                        if len(state["sensordata"]) > 0:
                            node.send_output(
                                "sensor_data",
                                pa.array(state["sensordata"]),
                                {"timestamp": current_time},
                            )

                    elif event_id == "camera_tick":
                        frame = simulator.render_camera()
                        if frame is None:
                            continue
                        rgb, depth = frame
                        fx, fy, cx, cy = simulator.camera_intrinsics
                        image_metadata = metadata.copy()
                        image_metadata["encoding"] = "rgb8"
                        image_metadata["width"] = int(rgb.shape[1])
                        image_metadata["height"] = int(rgb.shape[0])
                        image_metadata["resolution"] = [int(cx), int(cy)]
                        image_metadata["focal_length"] = [float(fx), float(fy)]
                        image_metadata["timestamp"] = time.time_ns()
                        node.send_output("image", pa.array(rgb.ravel()), image_metadata)

                        depth_metadata = image_metadata.copy()
                        depth_metadata["encoding"] = "mono16"
                        node.send_output("depth", pa.array(depth.ravel()), depth_metadata)

                elif event["type"] == "ERROR":
                    raise RuntimeError(event["error"])
    finally:
        simulator.close()


if __name__ == "__main__":
    main()
