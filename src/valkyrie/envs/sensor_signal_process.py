import numpy as np
from valkyrie.envs.filter import FilterClass


class calculateCOP():
    def __repr__(self):
        return str(self.__class__)

    def __init__(self, force_cuttoff_freq, pos_cutoff_freq, dt, order):
        self.left_foot_force_filter = [
            FilterClass(), FilterClass(), FilterClass()]
        self.right_foot_force_filter = [
            FilterClass(), FilterClass(), FilterClass()]
        self.left_foot_COP_filter = [
            FilterClass(), FilterClass(), FilterClass()]
        self.right_foot_COP_filter = [
            FilterClass(), FilterClass(), FilterClass()]
        self.COP_filter = [
            FilterClass(), FilterClass(), FilterClass()]
        self.force_filter = [
            FilterClass(), FilterClass(), FilterClass()]

        for i in range(3):
            self.left_foot_force_filter[i].butterworth(
                dt, force_cuttoff_freq, order)
            self.right_foot_force_filter[i].butterworth(
                dt, force_cuttoff_freq, order)
            self.force_filter[i].butterworth(
                dt, force_cuttoff_freq, order)
            self.left_foot_COP_filter[i].butterworth(
                dt, pos_cutoff_freq, order)
            self.right_foot_COP_filter[i].butterworth(
                dt, pos_cutoff_freq, order)
            self.COP_filter[i].butterworth(
                dt, pos_cutoff_freq, order)

        self.COP_info = []
        self.left_COP_info = []
        self.right_COP_info = []
        self.COP_filtered_info = []
        self.left_COP_filtered_info = []
        self.right_COP_filtered_info = []

    def __call__(self, right_contact_info, left_contact_info):
        """Calculate centre of pressure."""
        leftFootCOP, leftF, leftFootCOPFlag = self.calculateFootCOP(
                                                                    left_contact_info)
        self.left_COP_info = [leftFootCOP, leftF, leftFootCOPFlag]
        rightFootCOP, rightF, rightFootCOPFlag = self.calculateFootCOP(
            right_contact_info)
        self.right_COP_info = [rightFootCOP, rightF, rightFootCOPFlag]

        if leftFootCOPFlag and (rightFootCOPFlag is False):
            F = leftF
            COP = leftFootCOP
            COPFlag = True
        elif (leftFootCOPFlag is False) and rightFootCOPFlag:
            F = rightF
            COP = rightFootCOP
            COPFlag = True
        elif leftFootCOPFlag and rightFootCOPFlag:
            F = leftF + rightF
            COP = (rightFootCOP*rightF[2]+leftFootCOP*leftF[2])/F[2]
            COPFlag = True
        else:
            F = np.array([0, 0, 0])
            COP = np.array([0, 0, 0])
            COPFlag = False

        self.COP_info = [COP, F, COPFlag]
        self.performFiltering()
        return (COP, F, COPFlag, rightFootCOP, rightF, rightFootCOPFlag,
                leftFootCOP, leftF, leftFootCOPFlag)

    def calculateFootCOP(self, contact_info):
        """Calculate position of foot Center Of Pressure (COP).

        Returns
        -------
        COP : numpy.array
            Foot centre of pressure
        F : numpy.array
            force along the x, y, and z axes of world frame
        flag : bool
        """
        COP = np.array([0, 0, 0])
        F = np.array([0, 0, 0])
        if len(contact_info) < 1:  # no contact
            return COP, F, False

        for i in range(len(contact_info)):
            # contact normal of foot pointing towards plane
            contactNormal = np.array(contact_info[i][7])
            # contact normal of plane pointing towards foot
            contactNormal = -contactNormal
            contactNormalForce = np.array(contact_info[i][9])
            contactPosition = np.array(contact_info[i][5])  # position on plane
            F_contact = np.array(contactNormal) * contactNormalForce
            F = F + F_contact
            # integration of contact point times vertical force
            COP = COP + contactPosition * F_contact[2]
        COP = COP / F[2]

        if F[2] < 10:  # threshold
            COP = np.array([0, 0, 0])
            return COP, F, False

        return COP, F, True

    def performFiltering(self):
        leftF = self.left_COP_info[1]
        rightF = self.right_COP_info[1]
        F = self.COP_info[1]
        leftCOP = self.left_COP_info[0]
        rightCOP = self.right_COP_info[0]
        COP = self.COP_info[0]
        for i in range(3):
            self.left_foot_force_filter[i].applyFilter(leftF[i])
            self.right_foot_force_filter[i].applyFilter(rightF[i])
            self.left_foot_COP_filter[i].applyFilter(leftCOP[i])
            self.right_foot_COP_filter[i].applyFilter(rightCOP[i])
            self.COP_filter[i].applyFilter(COP[i])
            self.force_filter[i].applyFilter(F[i])
        return

    def clearFilter(self):
        for i in range(3):
            self.left_foot_force_filter[i].clear_filter()
            self.right_foot_force_filter[i].clear_filter()
            self.left_foot_COP_filter[i].clear_filter()
            self.right_foot_COP_filter[i].clear_filter()
            self.COP_filter[i].clear_filter()
            self.force_filter[i].clear_filter()
        return

    def getFilteredCOP(self):
        COP = np.array(
            [self.COP_filter[0].y[0],
             self.COP_filter[1].y[0],
             self.COP_filter[2].y[0]])
        COPFlag = self.COP_info[1]
        F = np.array(
            [self.force_filter[0].y[0],
             self.force_filter[1].y[0],
             self.force_filter[2].y[0]])

        leftFootCOP = np.array(
            [self.left_foot_COP_filter[0].y[0],
             self.left_foot_COP_filter[1].y[0],
             self.left_foot_COP_filter[2].y[0]])
        leftFootCOPFlag = self.left_COP_info[1]
        leftF = np.array([self.left_foot_force_filter[0].y[0],
                          self.left_foot_force_filter[1].y[0],
                          self.left_foot_force_filter[2].y[0]])

        rightFootCOP = np.array(
            [self.right_foot_COP_filter[0].y[0],
             self.right_foot_COP_filter[1].y[0],
             self.right_foot_COP_filter[2].y[0]])
        rightFootCOPFlag = self.right_COP_info[2]
        rightF = np.array(
            [self.right_foot_force_filter[0].y[0],
             self.right_foot_force_filter[1].y[0],
             self.right_foot_force_filter[2].y[0]])

        return (COP, F, COPFlag, rightFootCOP, rightF, rightFootCOPFlag,
                leftFootCOP, leftF, leftFootCOPFlag)
