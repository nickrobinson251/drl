import numpy as np
from pprint import pformat

from valkyrie.envs.filter import ButterworthFilter, KalmanFilter


class PDController:
    """PD Contoller class.

    Parameters
    ----------
    sample_period : int
        Sample period of Butterworth filter
    cutoff : list of length 3 of floats
        Cutoff frequency in hertz for [position, velocity, torque]
    filter_order : int
        Order of filter (1, 2, 3, or 4)
    is_filter : list of length 3 of bools (default is [False, False, False])
        Whether or not to filter each of [position, velocity, torque]
    """
    def __repr__(self):
        return "PDController({})".format(pformat(dict(
                 cutoff=self.cutoff,
                 filter_order=self.filter_order,
                 Kp=self.Kp,
                 Kd=self.Kd,
                 is_filter=self.is_filter,
                 name=self.name,
                 sample_period=self.sample_period,
                 u_max=self.u_max,
                 v_max=self.v_max)))

    def __init__(self,
                 sample_period,
                 cutoff,
                 filter_order,
                 Kp=0,
                 Kd=0,
                 is_filter=[False, False, False],
                 name=None,
                 u_max=0,
                 v_max=0):
        self.Kp = Kp
        self.Kd = Kd
        self.name = name
        self.u_max = u_max
        self.v_max = v_max
        self.is_filter = is_filter
        self.sample_period = sample_period
        self.cutoff = cutoff
        self.filter_order = filter_order
        # initialise filters
        self.position_filter = ButterworthFilter(
            sample_period,
            cutoff[0],
            filter_order)
        self.velocity_filter = ButterworthFilter(
            sample_period,
            cutoff[1],
            filter_order)
        self.torque_filter = ButterworthFilter(
            sample_period,
            cutoff[2],
            filter_order)
        self.kalman_position_filter = KalmanFilter(1e-6, 1e-4, 0.0, 0.0)
        self.kalman_velocity_filter = KalmanFilter(1e-5, 1e-3, 0.0, 0.0)
        # initialise measuremnt values to zero
        self.measured_position = 0.0
        self.measured_velocity = 0.0
        self.filtered_position = 0.0
        self.filtered_velocity = 0.0
        self.adjusted_position = 0.0
        self.adjusted_velocity = 0.0
        self.kalman_filtered_velocity = 0.0
        self.kalman_filtered_position = 0.0
        self.u = 0.0  # torque
        self.u_e = 0.0
        self.u_de = 0.0
        self.u_raw = 0.0
        self.u_adj = 0.0
        self.u_kal = 0.0

    def _calculate_torque(self, target_position, target_velocity, method):
        """Compute torque using measured and target position and velocity.

        Parameters
        ----------
        method : str ("raw", "adjusted", "filtered")
            method used to meaure current position and velocity
        """
        use_torque_filter = self.is_filter[2]
        if method == "adjusted":
            current_position = self.adjusted_position
            current_velocity = self.adjusted_velocity
        elif method == "filtered":
            current_position = self.kalman_filtered_position
            current_velocity = self.kalman_filtered_position
        else:
            current_position = self.measured_position
            current_velocity = self.measured_velocity
        e = float(target_position - current_position)
        de = float(target_velocity - current_velocity)
        self.u_e = self.Kp * e
        self.u_de = self.Kd * de
        torque = self.u_e + self.u_de
        if use_torque_filter:
            torque = self.torque_filter(torque)
        torque = np.clip(torque, -self.u_max, self.u_max)
        return torque

    def control(self, target_position, target_velocity):
        # torque calculated using raw inputs
        self.u_raw = self._calculate_torque(
            target_position,
            target_velocity,
            method="raw")
        # torque calculated using adjusted inputs
        self.u_adj = self._calculate_torque(
            target_position,
            target_velocity,
            method="adjusted")
        # torque calculated using kalman filter
        self.u_kal = self._calculate_torque(
            target_position,
            target_velocity,
            method="filtered")
        return self.u_kal

    def update_measurements(self, measured_position, measured_velocity):
        """Update measurements of current position and velocity."""
        past_position = self.filtered_position
        if self.is_filter[0]:
            self.filtered_position = self.position_filter(
                measured_position)
        else:
            self.filtered_position = measured_position
        self.measured_position = measured_position
        self.adjusted_position = (self.filtered_position
                                  + (self.filtered_position - past_position))
        self.kalman_filtered_position = (
            self.kalman_position_filter.input_latest_noisy_measurement(
                measured_position))

        past_velocity = self.filtered_velocity
        if self.is_filter[1]:
            self.filtered_velocity = self.velocity_filter(
                measured_velocity)
        else:
            self.filtered_velocity = measured_velocity
        self.measured_velocity = measured_velocity
        self.adjusted_velocity = (self.filtered_velocity
                                  + (self.filtered_velocity - past_velocity))
        self.kalman_filtered_velocity = (
            self.kalman_velocity_filter.input_latest_noisy_measurement(
                measured_velocity))

    def reset(self, position, velocity, torque):
        self.position_filter.initialise_xy(position)
        self.velocity_filter.initialise_xy(velocity)
        self.torque_filter.initialise_xy(torque)
