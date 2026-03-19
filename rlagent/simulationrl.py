from src.simulation import Simulation


# ---------------------------------------------------------------------------
# Default reward weights — single source of truth.
#
# These are used by sim_step_rl() when no weights are passed explicitly.
# CarlosGymEnv._DEFAULT_REWARD_WEIGHTS must match these values exactly.
# The gym env always passes its own resolved weights to sim_step_rl(), so
# this fallback is only reached if sim_step_rl() is called directly
# (e.g. from a test script) without specifying weights.
#
# History of the mismatch that was here:
#   Original simulationrl.py had collision=-50.0, out_lane=-10.0.
#   CarlosGymEnv._DEFAULT_REWARD_WEIGHTS had collision=-10.0, out_lane=-5.0.
#   These were never the same value, making the "true" default ambiguous.
#   Fixed: both files now use the same baseline values below.
# ---------------------------------------------------------------------------
_DEFAULT_REWARD_WEIGHTS = {
    'in_lane'  :  1.0,
    'in_motion':  0.5,
    'collision': -10.0,
    'out_lane' :  -5.0,
}


class SimulationRL(Simulation):

    def sim_step_rl(self, external_action, reward_weights=None):
        """
        Advance the simulation by one step given an external (steering,
        acceleration) action and return the scalar reward.

        Parameters
        ----------
        external_action : tuple (steering_rad, acceleration_mph2)
        reward_weights  : dict or None.  If None, falls back to
                          _DEFAULT_REWARD_WEIGHTS defined above.
                          In normal gym training/evaluation the gym env
                          always passes resolved weights so this fallback
                          is only used in direct test scripts.

        Returns
        -------
        reward : float
        """
        # 1. Update sensors to reflect current vehicle pose BEFORE the action
        self.agent.sensors.update_sensors(
            self.vehicle.center_point, self.vehicle.heading
        )

        # 2. Read state BEFORE the action (used by compute_reward for context)
        state = self.get_state()

        # 3. Unpack and apply the action
        steering, acceleration = external_action[0], external_action[1]
        self.vehicle.update_position(steering, acceleration, self.dt)

        # 4. Update sensors to reflect new vehicle pose AFTER the action
        self.agent.sensors.update_sensors(
            self.vehicle.center_point, self.vehicle.heading
        )

        # 5. Update in-lane / in-motion status flags
        self.update_sim_status()

        # 6. Collision check — result stored as instance attribute so
        #    CarlosGymEnv.step() can read it via getattr(self.sim, '_collision')
        #    without calling is_done().
        #    Assigned every step so a collision from a previous episode
        #    never bleeds into the next one (no stale True value).
        self._collision = self.check_collision_with_obstacles()

        # 7. Resolve reward weights
        active_weights = reward_weights if reward_weights is not None \
                         else _DEFAULT_REWARD_WEIGHTS

        # 8. Compute and return reward
        reward = self.agent.compute_reward(
            state,
            in_lane   = self.vehicle_in_lane,
            in_motion = self.vehicle_in_motion,
            collision = self._collision,
            weights   = active_weights,
        )
        return reward

    # -----------------------------------------------------------------------
    def check_collision_with_obstacles(self):
        """
        Return True if the vehicle bounding circle overlaps any obstacle.
        Uses circular approximation for both vehicle and obstacles.
        """
        vehicle_pos    = [self.vehicle.center_point.x,
                          self.vehicle.center_point.y]
        vehicle_radius = getattr(self.vehicle, 'radius', 2.0)

        for obs in self.environment.obstacles:
            obs_pos = obs.position
            dist = (
                (vehicle_pos[0] - obs_pos[0]) ** 2 +
                (vehicle_pos[1] - obs_pos[1]) ** 2
            ) ** 0.5
            if dist < (vehicle_radius + obs.radius):
                return True
        return False

    # -----------------------------------------------------------------------
    def is_done(self):
        """
        Episode termination predicate.

        NOTE: CarlosGymEnv.step() does NOT call this method.  Termination
        in the gym environment is determined directly from in_lane, in_motion,
        and self._collision flags to avoid the SEGMENT=0 early-termination
        bug caused by the parent Simulation class on a closed-loop track.

        This method is kept for compatibility with non-gym code that calls
        SimulationRL directly (e.g. standalone test scripts).  Its logic is
        intentionally identical to what CarlosGymEnv.step() implements:
          terminated when: collision OR out-of-lane OR stopped.
        """
        if getattr(self, '_collision', False):
            return True
        _, in_lane, in_motion = self.get_sim_status()
        return not in_lane or not in_motion