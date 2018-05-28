import numpy as np
from pprint import pformat

from valkyrie.envs.filter import FilterClass, KalmanFilter


class PDController:
    """PD Contoller class.

    Parameters
    ----------
    T : int
        Sample period of Butterworth filter
    cutoff : list of length 3
        Cutoff frequency in hertz for position velocity torque
    N : int
        Order of filter (1, 2, 3, or 4)
    """
    def __repr__(self):
        return "PDController({})".format(pformat(dict(
                 gains=[self.Kp, self.Kd],
                 u_max=self.u_max,
                 v_max=self.v_max,
                 name=self.name,
                 is_filter=self.is_filter,
                 T=self.T,
                 cutoff=self.cutoff,
                 N=self.N)))

    def __init__(self,
                 T,
                 cutoff,
                 N,
                 gains=[0, 0],
                 u_max=0,
                 v_max=0,
                 name=None,
                 is_filter=[False, False, False]):
        self.Kp = gains[0]
        self.Kd = gains[1]
        self.name = name
        self.u_max = u_max
        self.v_max = v_max
        self.is_filter = is_filter  # filter for position velocity torque
        self.T = T
        self.cutoff = cutoff  # cutoff frequency for position velocity torque
        self.N = N
        self.position_filter = FilterClass()
        self.position_filter.butterworth(T, cutoff[0], N)
        self.velocity_filter = FilterClass()
        self.velocity_filter.butterworth(T, cutoff[1], N)
        self.torque_filter = FilterClass()
        self.torque_filter.butterworth(T, cutoff[2], N)
        # self.position_filter = EMAFilter(0.1)#0.9 of new signal
        # self.velocity_filter = EMAFilter(0.1)#0.1 of new signal
        self.kalman_position_filter = KalmanFilter(1e-6, 1e-4, 0.0, 0.0)
        self.kalman_velocity_filter = KalmanFilter(1e-5, 1e-3, 0.0, 0.0)
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
        self.u_kal = 0.0  # torque calculated using kalman filter

    def control(self, target_position, target_velocity):
        # torque calculated using raw inputs
        e = float(target_position-self.measured_position)
        de = float(target_velocity-self.measured_velocity)
        self.u_e = self.Kp*e
        self.u_de = self.Kd*de
        u = self.u_e + self.u_de
        u = np.clip(u, -self.u_max, self.u_max)
        self.u_raw = u

        # torque calculated using adjusted inputs
        e = float(target_position-self.adjusted_position)
        de = float(target_velocity-self.adjusted_velocity)
        self.u_e = self.Kp*e
        self.u_de = self.Kd*de
        u = self.u_e + self.u_de
        u = np.clip(u, -self.u_max, self.u_max)
        self.u_adj = u

        # torque calculated using kalman filter
        e = float(target_position-self.kalman_filtered_position)
        de = float(target_velocity-self.kalman_filtered_velocity)
        self.u_e = self.Kp*e
        self.u_de = self.Kd*de
        u = self.u_e + self.u_de
        u = np.clip(u, -self.u_max, self.u_max)
        self.u_kal = u

        # torque calculated using filtered inputs
        e = float(target_position-self.kalman_filtered_position)
        de = float(target_velocity-self.kalman_filtered_velocity)
        # e = float(target_position-self.filtered_position)
        # de = float(target_velocity-self.filtered_velocity)
        self.u_e = self.Kp*e
        self.u_de = self.Kd*de
        u = self.u_e + self.u_de

        if self.is_filter[2]:
            u = self.torque_filter.applyFilter(u)
        u = np.clip(u, -self.u_max, self.u_max)  # clipping torque
        self.u = u
        return u

    def updateMeasurements(self, measured_position, measured_velocity):
        if self.is_filter[0]:
            past_position = self.filtered_position
            self.filtered_position = self.position_filter.applyFilter(
                measured_position)
            self.adjusted_position = self.filtered_position + (
                self.filtered_position - past_position)
            self.measured_position = measured_position
            self.kalman_filtered_position = (
                self.kalman_position_filter.input_latest_noisy_measurement(
                    measured_position))
            self.kalman_filtered_velocity = (
                self.kalman_velocity_filter.input_latest_noisy_measurement(
                    measured_velocity))
        else:
            past_position = self.filtered_position
            self.filtered_position = measured_position
            self.adjusted_position = self.filtered_position + (
                self.filtered_position - past_position)
            self.measured_position = measured_position
            self.kalman_filtered_position = (
                self.kalman_position_filter.input_latest_noisy_measurement(
                    measured_position))
            self.kalman_filtered_velocity = (
                self.kalman_velocity_filter.input_latest_noisy_measurement(
                    measured_velocity))
        if self.is_filter[1]:
            past_velocity = self.filtered_velocity
            self.filtered_velocity = self.velocity_filter.applyFilter(
                measured_velocity)
            self.adjusted_velocity = self.filtered_velocity + (
                self.filtered_velocity - past_velocity)
            self.measured_velocity = measured_velocity
            self.kalman_filtered_position = (
                self.kalman_position_filter.input_latest_noisy_measurement(
                    measured_position))
            self.kalman_filtered_velocity = (
                self.kalman_velocity_filter.input_latest_noisy_measurement(
                    measured_velocity))
        else:
            past_velocity = self.filtered_velocity
            self.filtered_velocity = measured_velocity
            self.adjusted_velocity = self.filtered_velocity + (
                self.filtered_velocity - past_velocity)
            self.measured_velocity = measured_velocity
            self.kalman_filtered_position = (
                self.kalman_position_filter.input_latest_noisy_measurement(
                    measured_position))
            self.kalman_filtered_velocity = (
                self.kalman_velocity_filter.input_latest_noisy_measurement(
                    measured_velocity))

    def reset(self, position, velocity, torque):
        # self.position_filter = FilterClass()
        # self.position_filter.butterworth(self.T, self.cutoff, self.N)
        # self.velocity_filter = FilterClass()
        # self.velocity_filter.butterworth(self.T, self.cutoff, self.N)
        self.position_filter.initializeFilter(position)
        self.velocity_filter.initializeFilter(velocity)
        self.torque_filter.initializeFilter(torque)
