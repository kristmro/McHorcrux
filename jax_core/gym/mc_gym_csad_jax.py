#!/usr/bin/env python3
"""
------------------------------------------------------------------------------------------------------------------------------------------------
6DOF Ship Navigation Simulation Gym (MC-GYM-JAX)
------------------------------------------------------------------------------------------------------------------------------------------------
Author: Adapted Example
Date:   2025-03-25

Description:
    This JAX‑based gym environment simulates a 6DOF vessel subject to differentiable wave loads.
    Vessel dynamics are computed functionally using csad_x_dot and integrated via an RK4 integrator.
    Vessel parameters are loaded via load_csad_parameters (init_csad style) and the state is stored as a
    12-element vector (η and ν). Wave loads are computed using init_wave_load and wave_load.
    
    The environment supports user‑defined tasks (or premade ones) and returns a zero reward.
    
    Optional real‑time rendering (pygame) and post‑simulation plotting (matplotlib) are provided.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pygame

# Functional vessel simulation routines:
from jax_core.simulator.vessels.csad_jax import load_csad_parameters, csad_x_dot
from jax_core.utils import rk4_step
# Wave spectrum & load routines:
from jax_core.simulator.waves.wave_spectra_jax import jonswap_spectrum
from jax_core.simulator.waves.wave_load_jax_jit import init_wave_load, wave_load, WaveLoad
# Conversion utilities:
from jax_core.utils import three2sixDOF, six2threeDOF
# Reference trajectory filter (for four-corner test)
from jax_core.ref_gen.reference_filter import ThrdOrderRefFilter

# Default config file (adjust the path as needed)
DEFAULT_CONFIG_FILE = "/home/kmroen/miniconda3/envs/tensor/lib/python3.9/site-packages/mclsimpy/vessel_data/CSAD/vessel_json.json"

class McGymJAX:
    """
    JAX‑based gym environment for 6DOF ship navigation.
    
    Parameters:
      dt          : Simulation timestep.
      grid_width  : Domain width (east).
      grid_height : Domain height (north).
      render_on   : If True, enable real‑time rendering via pygame.
      final_plot  : If True, store trajectory data for post‑simulation plotting.
      config_file : Path to vessel configuration JSON.
      Uc          : Ambient current speed (default 0.0).
      beta_c      : Ambient current direction (radians, default 0.0).
    """
    def __init__(self, dt=0.1, grid_width=15, grid_height=6, render_on=False, final_plot=True,
                 config_file=DEFAULT_CONFIG_FILE, Uc=0.0, beta_c=0.0):
        self.dt = dt
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.render_on = render_on
        self.final_plot = final_plot
        self.config_file = config_file

        # Load vessel parameters (functional style)
        self.params = load_csad_parameters(config_file)
        # The vessel state is now a 12-element array: [η (6DOF); ν (6DOF)]
        self.x = None  # Will be set in reset()
        
        self.Uc = Uc
        self.beta_c = beta_c

        # Wave load (to be initialized in set_wave_conditions)
        self.waveload = None
        self.curr_sim_time = 0.0

        # Trajectory storage for plotting
        self.trajectory = []  
        self.true_vel = []

        # Task-related variables
        self.start_position = None   # (north, east, heading_rad)
        self.goal = None             # (north, east, size)
        self.obstacles = []
        self.wave_conditions = None
        self.goal_func = None
        self.obstacle_func = None

        # Four-corner test settings
        self.four_corner_test = False
        self.set_points = None
        self.simtime = 300
        self.t = None
        self.ref_model = None
        self.store_xd = None
        self.ref_omega = None

        self.position_tolerance = 0.5  
        self.goal_heading_deg = None  
        self.heading_tolerance_deg = 10.0  

        # For JAX random operations
        self.rng_key = jax.random.PRNGKey(42)

        # Pygame setup (if rendering is enabled)
        self.screen = None
        self.clock = None
        self.WINDOW_WIDTH = 750
        self.WINDOW_HEIGHT = 300
        if self.render_on and pygame is not None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
            pygame.display.set_caption("6DOF Ship Navigation Simulation (JAX)")
            self.clock = pygame.time.Clock()
        self.x_scale = self.WINDOW_WIDTH / self.grid_width  
        self.y_scale = self.WINDOW_HEIGHT / self.grid_height

        # For reward shaping (placeholder)
        self.previous_action = jnp.zeros(3)

    def set_task(self, 
                 start_position, 
                 goal=None, 
                 wave_conditions=None, 
                 obstacles=None,
                 goal_func=None,
                 obstacle_func=None,
                 position_tolerance=0.5,
                 goal_heading_deg=None,
                 heading_tolerance_deg=10.0,
                 four_corner_test=False,
                 ref_omega=[0.2, 0.2, 0.2],
                 simtime=300):
        """
        Configure a new task.
          start_position : (north, east, heading_deg)
          goal           : (north, east, size)
          wave_conditions: (hs, tp, wave_dir_deg)
          obstacles      : list of (obs_n, obs_e, obs_size)
        """
        # Convert heading to radians
        self.start_position = jnp.array([start_position[0], start_position[1], jnp.deg2rad(start_position[2])])
        self.goal = goal
        self.wave_conditions = wave_conditions
        self.obstacles = obstacles if obstacles is not None else []
        self.goal_func = goal_func
        self.obstacle_func = obstacle_func

        self.position_tolerance = position_tolerance
        self.goal_heading_deg = goal_heading_deg
        self.heading_tolerance_deg = heading_tolerance_deg

        self.four_corner_test = four_corner_test
        if self.four_corner_test:
            self.set_points = [jnp.array([2.0, 2.0, 0.0]),
                               jnp.array([4.0, 2.0, 0.0]),
                               jnp.array([4.0, 4.0, 0.0]),
                               jnp.array([4.0, 4.0, -jnp.pi/4]),
                               jnp.array([2.0, 4.0, -jnp.pi/4]),
                               jnp.array([2.0, 2.0, 0.0])]
            self.simtime = simtime
            self.t = jnp.arange(0, simtime, self.dt)
            self.ref_omega = jnp.array(ref_omega)
            print("Four-corner test enabled")
            self.ref_model = ThrdOrderRefFilter(self.dt, omega=self.ref_omega, initial_eta=self.start_position)
            self.store_xd = jnp.zeros((len(self.t), 9))
        self.reset()

    def reset(self):
        """Reset the environment and vessel state."""
        self.curr_sim_time = 0.0
        # Convert start_position (3DOF: north, east, heading) to 6DOF using three2sixDOF
        eta_start = three2sixDOF(jnp.array([self.start_position[0],
                                              self.start_position[1],
                                              self.start_position[2]]))
        nu_start = jnp.zeros(6)
        self.x = jnp.concatenate([eta_start, nu_start])
        if self.wave_conditions is not None:
            hs, tp, wave_dir_deg = self.wave_conditions
            self.set_wave_conditions(hs, tp, wave_dir_deg)
        if self.final_plot:
            self.trajectory = []
        self.previous_action = jnp.zeros(3)
        if self.render_on and self.screen is not None:
            self.screen.fill((20, 20, 20))

    def set_wave_conditions(self, hs, tp, wave_dir_deg, N_w=100, gamma=3.3):
        """
        Initialize differentiable wave load.
          hs         : Significant wave height.
          tp         : Peak period.
          wave_dir_deg: Wave incident direction (degrees).
        """
        wp = 2 * jnp.pi / tp
        wmin, wmax = wp / 2, 3.0 * wp
        dw = (wmax - wmin) / N_w
        freqs = jnp.linspace(wmin, wmax, N_w, endpoint=True)
        omega, spec = jonswap_spectrum(freqs, hs, tp, gamma=gamma, freq_hz=False)
        wave_amps = jnp.sqrt(2 * spec * dw)
        self.rng_key, subkey = jax.random.split(self.rng_key)
        eps = jax.random.uniform(subkey, shape=(N_w,), minval=0.0, maxval=2*jnp.pi)
        angles = jnp.ones(N_w) * jnp.deg2rad(wave_dir_deg)
        self.waveload = init_wave_load(
            wave_amps=wave_amps,
            freqs=omega,
            eps=eps,
            angles=angles,
            config_file=self.config_file,
            rho=1025,
            g=9.81,
            dof=6,
            depth=100,
            deep_water=True,
            qtf_method="Newman",
            qtf_interp_angles=True,
            interpolate=True
        )

    def get_four_corner_nd(self, step_count):
        """
        Retrieve desired reference states during a four-corner test.
        """
        current_time = self.t[step_count]
        if self.four_corner_test and jnp.allclose(self.start_position, self.set_points[0], atol=1e-3):
            if current_time < 10.0:
                idx = 0
            else:
                shifted_time = current_time - 10.0
                remaining_time = self.simtime - 10.0
                segment_duration = remaining_time / 5.0
                idx = 1 + min(4, int(shifted_time // segment_duration))
        else:
            if current_time > 5 * self.simtime / 6:
                idx = 5
            elif current_time > 4 * self.simtime / 6:
                idx = 4
            elif current_time > 3 * self.simtime / 6:
                idx = 3
            elif current_time > 2 * self.simtime / 6:
                idx = 2
            elif current_time > self.simtime / 6:
                idx = 1
            else:
                idx = 0
        self.ref_model.set_eta_r(self.set_points[idx])
        self.ref_model.update()
        eta_d      = self.ref_model.get_eta_d()
        eta_d_dot  = self.ref_model.get_eta_d_dot()
        eta_d_ddot = self.ref_model.get_eta_d_ddot()
        nu_d_body  = self.ref_model.get_nu_d()
        self.store_xd = self.store_xd.at[step_count].set(self.ref_model._x)
        return eta_d, eta_d_dot, eta_d_ddot, nu_d_body

    def step(self, action):
        """
        Perform one simulation step.
          action: Control action in 3DOF (e.g., surge, sway, yaw).
        Returns: (state, done, info, reward)
        """
        self.curr_sim_time += self.dt

        # Update dynamic goal/obstacles if callable
        if self.goal_func is not None:
            self.goal = self.goal_func(self.curr_sim_time)
        if self.obstacle_func is not None:
            self.obstacles = self.obstacle_func(self.curr_sim_time)

        # Convert 3DOF action to 6DOF control force
        tau_control = three2sixDOF(action)
        # Compute wave force using differentiable wave load; use vessel's current η = self.x[:6]
        if self.waveload is not None:
            tau_wave = wave_load(self.curr_sim_time, self.x[:6], self.waveload)
        else:
            tau_wave = jnp.zeros(6)
        tau = tau_control + tau_wave

        # Update the vessel state using RK4 integration:
        self.x = rk4_step(self.x, self.dt, csad_x_dot, self.Uc, self.beta_c, tau, self.params)
        boat_pos = six2threeDOF(self.x[:6])
        done, info = self._check_termination(boat_pos)
        reward = self.compute_reward(action, self.previous_action)
        self.previous_action = action

        if self.final_plot:
            self.trajectory.append(jnp.array(boat_pos).copy())
            self.true_vel.append(six2threeDOF(self.x[6:]).copy())
        if self.render_on:
            self.render()
        return self.get_state(), done, info, reward

    def _check_termination(self, boat_pos):
        """
        Check termination conditions: four-corner test completion, goal attainment, or collisions.
        """
        if self.four_corner_test:
            if self.curr_sim_time > self.simtime:
                print("Four-corner test completed.")
                return True, {"reason": "four_corner_test_completed"}
            else:
                return False, {}
        if self.goal is not None:
            boat_yaw = self.x[5]  # assuming yaw is the last element of η
            g_n, g_e, g_size = self.goal
            dx = boat_pos[0] - g_n
            dy = boat_pos[1] - g_e
            distance_to_goal = jnp.sqrt(dx**2 + dy**2)
            heading_ok = True
            if self.goal_heading_deg is not None:
                boat_heading_deg = jnp.rad2deg(boat_yaw)
                heading_diff = (boat_heading_deg - self.goal_heading_deg + 180) % 360 - 180
                heading_ok = jnp.abs(heading_diff) <= self.heading_tolerance_deg
            if distance_to_goal < self.position_tolerance and heading_ok:
                print("Goal reached!")
                return True, {"reason": "goal_reached"}
            # Collision check using a simple circle test on boat hull points.
            hull_local = self._get_boat_hull_local_pts()
            c = jnp.cos(boat_yaw)
            s = jnp.sin(boat_yaw)
            rot = jnp.array([[c, s], [-s, c]])
            hull_global = []
            for (lx, ly) in hull_local:
                pt = rot @ jnp.array([lx, ly])
                gx_global = boat_pos[1] + pt[0]
                gy_global = boat_pos[0] + pt[1]
                hull_global.append(jnp.array([gx_global, gy_global]))
            for obs_n, obs_e, obs_size in self.obstacles:
                obs_radius = obs_size / 2.0
                for pt in hull_global:
                    if jnp.linalg.norm(pt - jnp.array([obs_n, obs_e])) < obs_radius:
                        print("Collision with obstacle!")
                        return True, {"reason": "collision"}
            return False, {}
        else:
            return True, {"reason": "No goal or four_corner_test initiated, terminating environment"}

    def get_state(self):
        """
        Return the current environment state.
        """
        eta = self.x[:6]
        nu = self.x[6:]
        return {
            "eta": eta,
            "nu": nu,
            "goal": self.goal,
            "obstacles": self.obstacles,
            "wave_conditions": self.wave_conditions,
        }

    def render(self):
        """Render the simulation using pygame."""
        if not self.render_on or self.screen is None:
            return
        self.screen.fill((20, 20, 20))
        self._draw_grid()
        if self.goal is not None or self.goal_func is not None:
            self._draw_goal()
        if self.obstacles is not None or self.obstacle_func is not None:
            self._draw_obstacles()
        self._draw_boat()
        pygame.display.flip()
        self.clock.tick(1000)

    def _draw_grid(self):
        grid_color = (50, 50, 50)
        for x in range(self.grid_width + 1):
            start_px = (int(x * self.x_scale), 0)
            end_px = (int(x * self.x_scale), self.WINDOW_HEIGHT)
            pygame.draw.line(self.screen, grid_color, start_px, end_px, 1)
        for y in range(self.grid_height + 1):
            start_px = (0, int(y * self.y_scale))
            end_px = (self.WINDOW_WIDTH, int(y * self.y_scale))
            pygame.draw.line(self.screen, grid_color, start_px, end_px, 1)

    def _draw_goal(self):
        g_n, g_e, g_size = self.goal
        px = g_e * self.x_scale
        py = (self.grid_height - g_n) * self.y_scale
        radius_px = (g_size / 2) * self.x_scale
        pygame.draw.circle(self.screen, (255, 215, 0), (int(px), int(py)), int(radius_px))

    def _draw_obstacles(self):
        for obs_n, obs_e, obs_size in self.obstacles:
            px = obs_e * self.x_scale
            py = (self.grid_height - obs_n) * self.y_scale
            radius_px = (obs_size / 2) * self.x_scale
            pygame.draw.circle(self.screen, (200, 0, 0), (int(px), int(py)), int(radius_px))

    def _get_boat_hull_local_pts(self):
        """
        Compute boat hull points in local coordinates.
        """
        Lpp = 2.5780001
        B   = 0.4440001
        halfL = 0.5 * Lpp
        halfB = 0.5 * B
        bow_start_x = 0.9344
        def bow_curve_points(n=40):
            pts = []
            P0 = (bow_start_x, +halfB)
            P1 = (halfL, 0.0)
            P2 = (bow_start_x, -halfB)
            for i in range(n+1):
                s = i / n
                x = (1 - s)**2 * P0[0] + 2*(1 - s)*s * P1[0] + s**2 * P2[0]
                y = (1 - s)**2 * P0[1] + 2*(1 - s)*s * P1[1] + s**2 * P2[1]
                pts.append((x, y))
            return pts
        x_stern_left  = -halfL
        x_stern_right = bow_start_x
        hull_pts_snippet = []
        hull_pts_snippet.append((x_stern_left, +halfB))
        hull_pts_snippet.append((x_stern_right, +halfB))
        hull_pts_snippet.extend(bow_curve_points(n=40))
        hull_pts_snippet.append((x_stern_left, -halfB))
        hull_pts_snippet.append((x_stern_left, +halfB))
        hull_pts_local = [ (pt[1], pt[0]) for pt in hull_pts_snippet ]
        return jnp.array(hull_pts_local)

    def _draw_boat(self):
        eta = self.x[:6]
        boat_pos = six2threeDOF(eta)
        boat_yaw = eta[5]  # assuming yaw is the 6th element
        hull_local = self._get_boat_hull_local_pts()
        c = jnp.cos(boat_yaw)
        s = jnp.sin(boat_yaw)
        rot = jnp.array([[c, s], [-s, c]])
        pixel_pts = []
        for (lx, ly) in hull_local:
            pt = rot @ jnp.array([lx, ly])
            gx = boat_pos[1] + pt[0]
            gy = boat_pos[0] + pt[1]
            sx = int(gx * self.x_scale)
            sy = int((self.grid_height - gy) * self.y_scale)
            pixel_pts.append((sx, sy))
        pygame.draw.polygon(self.screen, (0, 100, 255), pixel_pts)

    def compute_reward(self, action, prev_action):
        """Placeholder reward (always returns 0.0)."""
        return 0.0

    def close(self):
        if self.render_on and self.screen is not None and pygame is not None:
            pygame.quit()

    def __del__(self):
        self.close()

    def plot_trajectory(self):
        """
        Post-simulation plotting using matplotlib.
        """
        import numpy as np
        if not self.final_plot:
            return
        if self.goal is not None or self.goal_func is not None:
            traj = jnp.array(self.trajectory)
            traj_np = np.array(traj)
            plt.figure(figsize=(8, 4))
            plt.plot(traj_np[:, 1], traj_np[:, 0], 'g-', label="Boat Trajectory")
            g_n, g_e, g_s = self.goal
            plt.scatter(g_e, g_n, c='yellow', s=(g_s * self.x_scale)**2,
                        edgecolor='black', label="Goal")
            for obs_n, obs_e, obs_size in self.obstacles:
                plt.scatter(obs_e, obs_n, c='red', s=(obs_size * self.x_scale)**2,
                            edgecolor='black', label="Obstacle")
            plt.xlim([0, self.grid_width])
            plt.ylim([0, self.grid_height])
            plt.xlabel("East [m]")
            plt.ylabel("North [m]")
            plt.title("Boat Trajectory ({}×{} Domain)".format(self.grid_width, self.grid_height))
            plt.legend(loc='upper right', fontsize='small', scatterpoints=1, markerscale=0.1)
            plt.grid(True)
            plt.show()
        if self.four_corner_test:
            traj = jnp.array(self.trajectory)
            true_vel = jnp.array(self.true_vel)
            traj_np = np.array(traj)
            true_vel_np = np.array(true_vel)
            store_xd_np = np.array(self.store_xd)
            t_np = np.array(self.t)
            plt.figure(figsize=(8, 4))
            plt.plot(traj_np[:, 1], traj_np[:, 0], 'b-', label="Boat Trajectory")
            plt.plot(store_xd_np[:, 1], store_xd_np[:, 0], 'g-', label="Desired Trajectory")
            plt.xlim([0, self.grid_width])
            plt.ylim([0, self.grid_height])
            plt.xlabel("East [m]")
            plt.ylabel("North [m]")
            plt.title("Desired vs. Real Trajectory ({}×{} Domain)".format(self.grid_width, self.grid_height))
            plt.legend(loc='upper right', fontsize='small', scatterpoints=1, markerscale=0.1)
            plt.grid(True)
            plt.show()
            plt.figure(figsize=(8, 4))
            plt.plot(t_np, store_xd_np[:, 0], 'r-', label="North Desired")
            plt.plot(t_np, store_xd_np[:, 1], 'g-', label="East Desired")
            plt.plot(t_np, traj_np[:, 0], 'b-', label="North Actual")
            plt.plot(t_np, traj_np[:, 1], 'c-', label="East Actual")
            plt.xlabel("Time [s]")
            plt.ylabel("Position [m]")
            plt.title("Desired Trajectory Over Time")
            plt.legend(loc='upper right', fontsize='small', scatterpoints=1, markerscale=0.1)
            plt.show()
            plt.figure(figsize=(8, 4))
            plt.plot(t_np, store_xd_np[:, 3], 'r-', label="Desired Yaw")
            plt.plot(t_np, traj_np[:, 2], 'b-', label="Actual Yaw")
            plt.xlabel("Time [s]")
            plt.ylabel("Yaw [rad]")
            plt.title("Desired vs. Actual Yaw")
            plt.legend(loc='upper right', fontsize='small', scatterpoints=1, markerscale=0.1)
            plt.show()
            plt.figure(figsize=(8, 4))
            plt.plot(t_np, store_xd_np[:, 6], 'r-', label="North Desired Velocity")
            plt.plot(t_np, true_vel_np[:, 0], 'b-', label="North Actual Velocity")
            plt.plot(t_np, store_xd_np[:, 7], 'g-', label="East Desired Velocity")
            plt.plot(t_np, true_vel_np[:, 1], 'c-', label="East Actual Velocity")
            plt.xlabel("Time [s]")
            plt.ylabel("Velocity [m/s]")
            plt.title("Desired vs. Actual Velocities")
            plt.legend(loc='upper right', fontsize='small', scatterpoints=1, markerscale=0.1)
            plt.show()
            plt.figure(figsize=(8, 4))
            plt.plot(t_np, store_xd_np[:, 8], 'r-', label="Desired Yaw Rate")
            plt.plot(t_np, true_vel_np[:, 2], 'b-', label="Actual Yaw Rate")
            plt.xlabel("Time [s]")
            plt.ylabel("Yaw Rate [rad/s]")
            plt.title("Desired vs. Actual Yaw Rate")
            plt.legend(loc='upper right', fontsize='small', scatterpoints=1, markerscale=0.1)
            plt.show()
