from src.simulation import Simulation

class SimulationRL(Simulation):
    
    def sim_step_rl(self, external_action):
        # Update sensors before action
        self.agent.sensors.update_sensors(self.vehicle.center_point, self.vehicle.heading)

        # Get current state
        state = self.get_state()

        # Unpack external action (steering, acceleration)
        steering, acceleration = external_action[0], external_action[1]

        # Update vehicle position based on action and timestep
        self.vehicle.update_position(steering, acceleration, self.dt)

        # Update sensors after movement
        self.agent.sensors.update_sensors(self.vehicle.center_point, self.vehicle.heading)

        # Update simulation status (in lane, in motion, etc.)
        self.update_sim_status()

        # Check for collision between vehicle and obstacles
        collision = self.check_collision_with_obstacles()
        self._collision = collision  # store collision flag for episode termination

        # Compute reward considering lane and motion status
        reward = self.agent.compute_reward(state, in_lane=self.vehicle_in_lane, in_motion=self.vehicle_in_motion)

        if collision:
            reward = -10.0  

        return reward

    def check_collision_with_obstacles(self):
        vehicle_pos = [self.vehicle.center_point.x, self.vehicle.center_point.y]
        vehicle_radius = getattr(self.vehicle, 'radius', 2.0)  # approximate vehicle radius if not defined

        for obs in self.environment.obstacles:
            obs_pos = obs.position
            dist = ((vehicle_pos[0] - obs_pos[0]) ** 2 + (vehicle_pos[1] - obs_pos[1]) ** 2) ** 0.5
            
            if dist < (vehicle_radius + obs.radius):
                return True
        return False

    
    def is_done(self):
        _, in_lane, in_motion = self.get_sim_status()
        if hasattr(self, '_collision') and self._collision:
            return True
        return not in_lane or not in_motion

