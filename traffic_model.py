import numpy as np

class IntelligentDriverModel:
    """
    Microscopic Intelligent Driver Model (IDM) for vehicle car-following dynamics.
    Parameters:
      v0: Desired speed (m/s)
      tau: Safe time headway (s)
      a_max: Maximum acceleration (m/s^2)
      b_comf: Comfortable deceleration (m/s^2)
      s0: Minimum jam distance (m)
    """
    def __init__(self, v0=28.5, tau=1.35, a_max=1.8, b_comf=2.0, s0=2.0):
        self.v0 = v0
        self.tau = tau
        self.a_max = a_max
        self.b_comf = b_comf
        self.s0 = s0

    def calc_acceleration(self, v, lead_v, gap):
        gap = max(0.1, gap)
        dv = v - lead_v
        s_star = self.s0 + max(0.0, v * self.tau + (v * dv) / (2.0 * np.sqrt(self.a_max * self.b_comf)))
        accel = self.a_max * (1.0 - (v / max(0.1, self.v0))**4 - (s_star / gap)**2)
        return float(np.clip(accel, -7.0, self.a_max))


class HighwayCorridor:
    """
    Simulates a multi-lane highway corridor with 2 induction loop detector stations.
    Station A: Upstream entry (km 0.5)
    Station B: Downstream bottleneck (km 1.5)
    """
    def __init__(self, length_m=2000.0, num_lanes=3):
        self.length = length_m
        self.num_lanes = num_lanes
        self.detector_a_pos = 500.0
        self.detector_b_pos = 1500.0

    def simulate_counts(self, idm_params, duration_s=180.0, dt=0.5, seed=42):
        """
        Executes a microscopic simulation run and returns hourly equivalent counts
        for Detector Station A and Detector Station B.
        """
        np.random.seed(seed)
        v0, tau, s0 = idm_params[0], idm_params[1], idm_params[2]
        driver_model = IntelligentDriverModel(v0=v0, tau=tau, s0=s0)

        num_vehicles = 45
        positions = np.sort(np.random.uniform(0, self.length, num_vehicles))[::-1]
        speeds = np.random.uniform(20.0, 28.0, num_vehicles)
        lanes = np.random.randint(0, self.num_lanes, num_vehicles)

        det_a_crossings = 0
        det_b_crossings = 0

        steps = int(duration_s / dt)
        for _ in range(steps):
            for i in range(num_vehicles):
                # Identify vehicle ahead in the same lane
                lane = lanes[i]
                same_lane_indices = [j for j in range(num_vehicles) if lanes[j] == lane and j != i]
                
                lead_gap = 999.0
                lead_speed = speeds[i]
                for j in same_lane_indices:
                    dx = positions[j] - positions[i]
                    if dx < 0:
                        dx += self.length
                    if dx < lead_gap:
                        lead_gap = dx
                        lead_speed = speeds[j]

                # Update acceleration and kinematics
                acc = driver_model.calc_acceleration(speeds[i], lead_speed, lead_gap - 4.5)
                speeds[i] = max(0.0, speeds[i] + acc * dt)
                old_x = positions[i]
                new_x = (old_x + speeds[i] * dt) % self.length
                positions[i] = new_x

                # Check detector station crossings
                if old_x < self.detector_a_pos <= old_x + speeds[i] * dt:
                    det_a_crossings += 1
                if old_x < self.detector_b_pos <= old_x + speeds[i] * dt:
                    det_b_crossings += 1

        # Scale to hourly flow rate (veh/h)
        scale = 3600.0 / duration_s
        return np.array([det_a_crossings * scale, det_b_crossings * scale])


def compute_rmse(sim_counts, obs_counts):
    """Computes Root-Mean-Square Error between simulated and observed detector counts."""
    return float(np.sqrt(np.mean((sim_counts - obs_counts) ** 2)))
