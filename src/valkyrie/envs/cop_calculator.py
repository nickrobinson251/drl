import numpy as np
from valkyrie.envs.filter import ButterworthFilter


class COPCalculator():
    """Calculator for filtered and unfiliter centre of pressure."""
    def __repr__(self):
        return str(self.__class__)

    def __init__(self,
                 force_cutoff,
                 position_cutoff,
                 sample_period,
                 filter_order):
        self.force_filter = [
            ButterworthFilter(sample_period, force_cutoff, filter_order),
            ButterworthFilter(sample_period, force_cutoff, filter_order),
            ButterworthFilter(sample_period, force_cutoff, filter_order)]
        self.left_foot_force_filter = [
            ButterworthFilter(sample_period, force_cutoff, filter_order),
            ButterworthFilter(sample_period, force_cutoff, filter_order),
            ButterworthFilter(sample_period, force_cutoff, filter_order)]
        self.right_foot_force_filter = [
            ButterworthFilter(sample_period, force_cutoff, filter_order),
            ButterworthFilter(sample_period, force_cutoff, filter_order),
            ButterworthFilter(sample_period, force_cutoff, filter_order)]
        self.COP_filter = [
            ButterworthFilter(sample_period, position_cutoff, filter_order),
            ButterworthFilter(sample_period, position_cutoff, filter_order),
            ButterworthFilter(sample_period, position_cutoff, filter_order)]
        self.left_foot_COP_filter = [
            ButterworthFilter(sample_period, position_cutoff, filter_order),
            ButterworthFilter(sample_period, position_cutoff, filter_order),
            ButterworthFilter(sample_period, position_cutoff, filter_order)]
        self.right_foot_COP_filter = [
            ButterworthFilter(sample_period, position_cutoff, filter_order),
            ButterworthFilter(sample_period, position_cutoff, filter_order),
            ButterworthFilter(sample_period, position_cutoff, filter_order)]

    def __call__(self, right_contact_info, left_contact_info):
        """Calculate centre of pressure."""
        self.left_foot_COP, self.left_force, self.left_foot_COP_flag = \
            self.calculate_foot_COP(left_contact_info)
        self.right_foot_COP, self.right_force, self.right_foot_COP_flag = \
            self.calculate_foot_COP(right_contact_info)

        if self.left_foot_COP_flag and not self.right_foot_COP_flag:
            self.force = self.left_force
            self.COP = self.left_foot_COP
            self.COP_flag = True
        elif self.right_foot_COP_flag and not self.left_foot_COP_flag:
            self.force = self.right_force
            self.COP = self.right_foot_COP
            self.COP_flag = True
        elif self.left_foot_COP_flag and self.right_foot_COP_flag:
            self.force = self.left_force + self.right_force
            self.COP = ((self.right_foot_COP * self.right_force[2]
                         + self.left_foot_COP * self.left_force[2])
                        / self.force[2])
            self.COP_flag = True
        else:
            self.force = np.array([0, 0, 0])
            self.COP = np.array([0, 0, 0])
            self.COP_flag = False
        self.filter()

    def calculate_foot_COP(self, contact_info):
        """Calculate position of foot Center Of Pressure (COP).

        Parameters
        ----------
        contact_info : list
            contact points returned by pybullet.getContactPoints

        Returns
        -------
        COP : numpy.array of length 3
            Foot centre of pressure
        force : numpy.array of length 3
            Force along the x, y, and z axes of world frame
        flag : bool
            True iff foot is grounded
        """
        if len(contact_info) < 1:  # no contact
            return np.zeros(3), np.zeros(3), False

        for i in range(len(contact_info)):
            # contact normal of foot pointing towards plane
            contact_normal = np.array(contact_info[i][7])
            # contact normal of plane pointing towards foot
            contact_normal = -contact_normal
            contact_normal_force = np.array(contact_info[i][9])
            contact_position = np.array(contact_info[i][5])  # position on plane
            force = np.array(contact_normal) * contact_normal_force
            # integration of contact point times vertical force
            COP = contact_position * force[2]
        if force[2] != 0:
            COP = COP / force[2]
        if force[2] < 10:  # threshold for force is z direction
            COP = np.array([0, 0, 0])
            return COP, force, False
        return COP, force, True

    def filter(self):
        for i in range(3):
            self.COP_filter[i](self.COP[i])
            self.force_filter[i](self.force[i])
            self.left_foot_COP_filter[i](self.left_foot_COP[i])
            self.left_foot_force_filter[i](self.left_force[i])
            self.right_foot_COP_filter[i](self.right_foot_COP[i])
            self.right_foot_force_filter[i](self.right_force[i])

    def clear_filter(self):
        for i in range(3):
            self.COP_filter[i].clear()
            self.force_filter[i].clear()
            self.left_foot_COP_filter[i].clear()
            self.left_foot_force_filter[i].clear()
            self.right_foot_COP_filter[i].clear()
            self.right_foot_force_filter[i].clear()

    def get_COP(self):
        """Returns COP assuming calculator has run."""
        return (
            self.COP,
            self.force,
            self.COP_flag,
            self.right_foot_COP,
            self.right_force,
            self.right_foot_COP_flag,
            self.left_foot_COP,
            self.left_force,
            self.left_foot_COP_flag)

    def get_filtered_COP(self):
        """Returns COP from Butterworth filter, assuming calculator has run."""
        self.COP_filtered = np.array(
            [self.COP_filter[0].y[0],
             self.COP_filter[1].y[0],
             self.COP_filter[2].y[0]])
        self.force_filtered = np.array(
            [self.force_filter[0].y[0],
             self.force_filter[1].y[0],
             self.force_filter[2].y[0]])

        self.left_foot_COP_filtered = np.array(
            [self.left_foot_COP_filter[0].y[0],
             self.left_foot_COP_filter[1].y[0],
             self.left_foot_COP_filter[2].y[0]])
        self.left_force_filtered = np.array(
            [self.left_foot_force_filter[0].y[0],
             self.left_foot_force_filter[1].y[0],
             self.left_foot_force_filter[2].y[0]])

        self.right_foot_COP_filtered = np.array(
            [self.right_foot_COP_filter[0].y[0],
             self.right_foot_COP_filter[1].y[0],
             self.right_foot_COP_filter[2].y[0]])
        self.right_force_filtered = np.array(
            [self.right_foot_force_filter[0].y[0],
             self.right_foot_force_filter[1].y[0],
             self.right_foot_force_filter[2].y[0]])

        return (
            self.COP_filtered,
            self.force_filtered,
            self.COP_flag,
            self.right_foot_COP_filtered,
            self.right_force_filtered,
            self.right_foot_COP_flag,
            self.left_foot_COP_filtered,
            self.left_force_filtered,
            self.left_foot_COP_flag)
