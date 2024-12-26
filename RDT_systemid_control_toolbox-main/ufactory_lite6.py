import mujoco as mj
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mujoco.glfw import glfw
from mujoco_base import MuJoCoBase
import os

class ufactory_lite6(MuJoCoBase):
    def __init__(self, xml_path):
        super().__init__(xml_path)
        self.timeframe = 0
        self.qpos_list = []
    def load_exp_data(self,pathTodata):
        self.exp_data = pd.read_csv(pathTodata)
        # print(self.exp_data['mt1'][1])
        # print(self.exp_data.shape[0])
        # print(self.data.ctrl.shape)
    def set_initial_state(self):
        # Set the initial state of the robot
        for i in range(self.data.qpos.shape[0]):
            tpos_id= "mp"+str(i+1)
            self.data.qpos[i] = self.exp_data[tpos_id][0]
            # print(self.data.qpos[i])
        mj.mj_forward(self.model, self.data)
    def save_data(self):
        self.qpos_list.append(self.data.qpos.copy())
        
    def reset(self):
        #Set camera configuration
        self.cam.azimuth = 120.89  # 89.608063
        self.cam.elevation = -15.81  # -11.588379
        self.cam.distance = 5.0  # 5.0
        self.cam.lookat = np.array([0.0, 0.0, 1.0])

        self.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        self.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        
        self.timeframe = 0
    def sysid_actuation(self):
        # Actuate the robot with the sysid data (mt1-mt7)
        for i in range(self.data.ctrl.shape[0]):
            torque_id= "mp"+str(i+1)
            self.data.ctrl[i] = self.exp_data[torque_id][self.timeframe]
            # print(self.data.ctrl[i])
    def sysid_animation(self):
        # Animate the robot with the sysid data (mp1-mp7)
        for i in range(self.data.qpos.shape[0]):
            tpos_id= "mp"+str(i+1)
            self.data.qpos[i] = self.exp_data[tpos_id][self.timeframe]
            # print(self.data.qpos[i])
    def simulate(self):
        glfw.set_window_title(self.window, "Torque Control with damping and friction")
        while not glfw.window_should_close(self.window):
            simstart = self.data.time
            # self.set_initial_state()
            simtimeframe = 0
            while (self.data.time - simstart < 1.0/1000.0):
                # Step the simulation
                mj.mj_step(self.model, self.data)
                if simtimeframe % 1 == 0:
                    # Apply the controller
                    self.sysid_actuation()
                    # self.sysid_animation()
                    # save data
                if simtimeframe % 10 == 0:
                    self.save_data()
                    self.timeframe += 1
                # print(self.timeframe)
            if self.timeframe >= self.exp_data.shape[0]-1:
                break
            
            # get framebuffer viewport
            viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
            viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

            # Show joint frames
            self.opt.flags[mj.mjtVisFlag.mjVIS_JOINT] = 0

            # Update scene and render
            self.cam.lookat[0] = self.data.qpos[0]
            mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                               mj.mjtCatBit.mjCAT_ALL.value, self.scene)
            mj.mjr_render(viewport, self.scene, self.context)

            # swap OpenGL buffers (blocking call due to v-sync)
            glfw.swap_buffers(self.window)

            # process pending GUI events, call GLFW callbacks
            glfw.poll_events()

        glfw.terminate()
        

if __name__=="__main__":
    # Load the model
    xml_path = "model/ufactory_lite6/scene.xml"
    lite6Sim = ufactory_lite6(xml_path)
    lite6Sim.load_exp_data("sysid_data/ufactory_lite6/joint_pos_and_torques_lite_6_100hz.csv")
    lite6Sim.set_initial_state()
    lite6Sim.simulate()
    print(lite6Sim.qpos_list)
    qpos_sim = np.array(lite6Sim.qpos_list)
    # print(qpos_sim[:,0])
    qpos_exp = np.array(lite6Sim.exp_data.iloc[:,1:7])
    print(qpos_exp.shape)

    for i in range(6):
        plt.plot(qpos_sim[:,i],label="simulated")
        plt.plot(qpos_exp[:,i],label="experimental")
        plt.legend()
        plt.show()