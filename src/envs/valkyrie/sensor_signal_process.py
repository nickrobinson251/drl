import numpy as np
from envs.valkyrie.filter import FilterClass


class calCOP():
    def __init__(self, force_cuttoff_freq, pos_cutoff_freq, dt, order):
        self.left_foot_force_filter = [
            FilterClass(), FilterClass(), FilterClass()]
        self.right_foot_force_filter = [
            FilterClass(), FilterClass(), FilterClass()]
        self.left_foot_COP_filter = [
            FilterClass(), FilterClass(), FilterClass()]
        self.right_foot_COP_filter = [
            FilterClass(), FilterClass(), FilterClass()]
        self.COP_filter = [FilterClass(), FilterClass(), FilterClass()]
        self.force_filter = [FilterClass(), FilterClass(), FilterClass()]

        for i in range(3):
            self.left_foot_force_filter[i].butterworth(
                dt, force_cuttoff_freq, order)
            self.right_foot_force_filter[i].butterworth(
                dt, force_cuttoff_freq, order)
            self.force_filter[i].butterworth(dt, force_cuttoff_freq, order)
            self.left_foot_COP_filter[i].butterworth(dt, pos_cutoff_freq, order)
            self.right_foot_COP_filter[i].butterworth(
                dt, pos_cutoff_freq, order)
            self.COP_filter[i].butterworth(dt, pos_cutoff_freq, order)

        self.COP_info = []
        self.left_COP_info = []
        self.right_COP_info = []

        self.COP_filtered_info = []
        self.left_COP_filtered_info = []
        self.right_COP_filtered_info = []

    def __call__(self, right_contact_info, left_contact_info):
        leftFootCOP, leftF, leftFootCOPFlag = self.calFootCOP(left_contact_info)
        self.left_COP_info = [leftFootCOP, leftF, leftFootCOPFlag]
        rightFootCOP, rightF, rightFootCOPFlag = self.calFootCOP(
            right_contact_info)
        self.right_COP_info = [rightFootCOP, rightF, rightFootCOPFlag]

        if leftFootCOPFlag and (rightFootCOPFlag is False):
            COP = leftFootCOP
            F = leftF
            COPFlag = True
        elif (leftFootCOPFlag is False) and rightFootCOPFlag:
            COP = rightFootCOP
            F = rightF
            COPFlag = True
        elif leftFootCOPFlag and rightFootCOPFlag:
            F = leftF + rightF

            COP = (rightFootCOP*rightF[2]+leftFootCOP*leftF[2])/F[2]

            COPFlag = True
        else:
            COP = np.array([0, 0, 0])

            F = np.array([0, 0, 0])
            COPFlag = False

        self.COP_info = [COP, F, COPFlag]

        self.performFiltering()
        return (COP, F, COPFlag, rightFootCOP, rightF, rightFootCOPFlag,
                leftFootCOP, leftF, leftFootCOPFlag)

    def calFootCOP(self, contact_info):

        if len(contact_info) < 1:  # no contact
            COP = np.array([0, 0, 0])
            F = np.array([0, 0, 0])
            return COP, F, False

        COP = np.array([0, 0, 0])  # position of Center of pressure
        F = np.array([0, 0, 0])  # force among the x,y,z axis of world frame
        # print(len(footGroundContact))
        for i in range(len(contact_info)):
            # contact normal of foot pointing towards plane
            contactNormal = np.array(contact_info[i][7])
            # print(contactNormal)
            # contact normal of plane pointing towards foot
            contactNormal = -contactNormal
            contactNormalForce = np.array(contact_info[i][9])
            # print(contactNormalForce)
            contactPosition = np.array(contact_info[i][5])  # position on plane
            F_contact = np.array(contactNormal)*contactNormalForce
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
        # print(COP)
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
