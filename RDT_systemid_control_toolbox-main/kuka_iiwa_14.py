import mujoco as mj 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mujoco.glfw import glfw
from mujoco_base import MuJoCoBase

class kuka_iiwa14(MuJoCoBase):
    def __init__(self, xml_path):
        super().__init__(xml_path)
        self.timeframe = 0

    def load_exp_data(self, pathTodata):
        """Load experimental data (CSV file) and perform smoothing"""
        self.exp_data = pd.read_csv(pathTodata)
        for col in self.exp_data.columns:
            if col.startswith("mt"):
                self.exp_data[col] = self.exp_data[col].rolling(window=5, min_periods=1).mean()

    def reset(self):
        """reset simulation and camera angle"""
        self.cam.azimuth = 120.89
        self.cam.elevation = -15.81
        self.cam.distance = 5.0
        self.cam.lookat = np.array([0.0, 0.0, 1.0])
        self.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        self.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        self.timeframe = 0

    def pid_controller(self, target_position):
        """PID Controller based on target position"""
        Kp, Kd, Ki = 150, 20, 5  # PID Gain parameter
        error = target_position - self.data.qpos[:]
        d_error = -self.data.qvel[:]
        self.integral_error = getattr(self, "integral_error", np.zeros_like(error))
        self.integral_error += error * 0.005  # time step
        self.data.ctrl[:] = Kp * error + Kd * d_error + Ki * self.integral_error

    def simulate(self):
        """main loop of simulation"""
        glfw.set_window_title(self.window, "KUKA iiwa 14 PID Simulation")

        joint_positions = []
        joint_torques = []

        while not glfw.window_should_close(self.window):
            simstart = self.data.time

            while (self.data.time - simstart < 1.0 / 200.0):
                if self.timeframe >= len(self.exp_data):
                    print("End of experimental data reached.")
                    glfw.set_window_should_close(self.window, True)
                    break

                mj.mj_step(self.model, self.data)
                target_position = self.exp_data.iloc[self.timeframe][
                    ["mp1", "mp2", "mp3", "mp4", "mp5", "mp6", "mp7"]
                ].to_numpy()
                self.pid_controller(target_position)

                joint_positions.append(self.data.qpos[:].copy())
                joint_torques.append(self.data.ctrl[:].copy())

                if self.timeframe % 10 == 0:
                    viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
                    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
                    mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                                       mj.mjtCatBit.mjCAT_ALL.value, self.scene)
                    mj.mjr_render(viewport, self.scene, self.context)
                    glfw.swap_buffers(self.window)

                glfw.poll_events()
                self.timeframe += 1

            if self.timeframe >= len(self.exp_data):
                break

        glfw.terminate()
        self.save_results(joint_positions, joint_torques)

    def save_results(self, joint_positions, joint_torques):
        """保存仿真结果"""
        # JSON 格式
        df = pd.DataFrame({
            "qpos": [list(pos) for pos in joint_positions],
            "ctrl": [list(ctrl) for ctrl in joint_torques]
        })
        df.to_json("simulation_results_pid.json", orient="records", indent=2)

    def compare_and_plot_subplots(self, experiment_file):
        """comparison"""
        # load data
        simulation_data = pd.read_json("simulation_results_pid.json")
        experiment_data = pd.read_csv(experiment_file)

        # Extract simulation and target positions
        sim_qpos = np.array([row for row in simulation_data["qpos"]])
        target_qpos = experiment_data[["mp1", "mp2", "mp3", "mp4", "mp5", "mp6", "mp7"]].to_numpy()

        # time axis
        time = np.linspace(0, len(sim_qpos) * 0.005, len(sim_qpos))

        fig, axs = plt.subplots(7, 1, figsize=(10, 15), sharex=True)

        for joint in range(sim_qpos.shape[1]):
            axs[joint].plot(time, sim_qpos[:, joint], label=f"Simulated Joint {joint+1}")
            axs[joint].plot(time, target_qpos[:, joint], label=f"Target Joint {joint+1}", linestyle="dashed")
            axs[joint].set_ylabel("Position (rad)")
            axs[joint].set_title(f"Joint {joint+1}")
            axs[joint].legend()

        axs[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.savefig("pid_results.png")  # save images
        plt.show(block=True)


if __name__ == "__main__":
    xml_path = "model/kuka_iiwa_14/scene.xml"
    kukaSim = kuka_iiwa14(xml_path)
    kukaSim.load_exp_data("sysid_data/kuka_iiwa_14/kuka_iiwa14_data.csv")
    kukaSim.simulate()
    kukaSim.compare_and_plot_subplots("sysid_data/kuka_iiwa_14/kuka_iiwa14_data.csv")