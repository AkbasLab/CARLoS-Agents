from src.simulation import Simulation

class SimulationRL(Simulation):
    def sim_step_rl(self, external_action):
        self.agent.sensors.update_sensors(self.vehicle.center_point, self.vehicle.heading)
        state = self.get_state()

        action = external_action 
        
        steering, acceleration = action[0], action[1]
        self.vehicle.update_position(steering, acceleration, self.dt)

        self.agent.sensors.update_sensors(self.vehicle.center_point, self.vehicle.heading)
        self.update_sim_status()

        reward = self.agent.compute_reward(state, in_lane=self.vehicle_in_lane, in_motion=self.vehicle_in_motion)
        return reward
    
    def is_done(self):
        _, in_lane, in_motion = self.get_sim_status()
        return not in_lane or not in_motion
