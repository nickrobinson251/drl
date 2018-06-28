import math
import numpy as np


class FilterClass():
    def __repr__(self):
        return str(self.__class__)

    def __init__(self):
        """Initialize all variables to 0."""
        self.MAX_FILTER_LENGTH = 8
        self.num_coefs = 6  # number of coefficients
        self.i = 0
        self.x = np.zeros(self.MAX_FILTER_LENGTH)
        self.y = np.zeros(self.MAX_FILTER_LENGTH)
        self.a = np.zeros(self.MAX_FILTER_LENGTH)
        self.b = np.zeros(self.MAX_FILTER_LENGTH)

    def clear(self):
        """Rest filter a, b, x, y values all to zeros."""
        self.x = np.zeros(self.MAX_FILTER_LENGTH)
        self.y = np.zeros(self.MAX_FILTER_LENGTH)
        self.a = np.zeros(self.MAX_FILTER_LENGTH)
        self.b = np.zeros(self.MAX_FILTER_LENGTH)

    def initialise_xy(self, x, y=None):
        for i in range(self.MAX_FILTER_LENGTH):
            self.x[i] = x
            self.y[i] = y if y else x

    def __call__(self, x):
        """Return filter signal from new input.

        Assumes the filter was run so that the coefficients are available
        """
        self.x[4] = self.x[3]
        self.x[3] = self.x[2]
        self.x[2] = self.x[1]
        self.x[1] = self.x[0]
        self.x[0] = x
        self.y[4] = self.y[3]
        self.y[3] = self.y[2]
        self.y[2] = self.y[1]
        self.y[1] = self.y[0]
        self.y[0] = self.a[0]*self.x[0]+self.x[1]*self.a[1] +\
            self.x[2]*self.a[2]+self.x[3]*self.a[3] +\
            self.x[4]*self.a[4]-self.y[1]*self.b[1]-self.y[2]*self.b[2] -\
            self.y[3]*self.b[3]-self.y[4]*self.b[4]
        return self.y[0]

    def differentiator(self, x):
        """Return filter signal from new input.

        Assumes the filter was run so that the coefficients are available
        """
        self.x[4] = self.x[3]
        self.x[3] = self.x[2]
        self.x[2] = self.x[1]
        self.x[1] = self.x[0]
        self.x[0] = x
        self.y[4] = self.y[3]
        self.y[3] = self.y[2]
        self.y[2] = self.y[1]
        self.y[1] = self.y[0]
        self.y[0] = (self.a[0]*self.x[0] + self.x[1]*self.a[1] +
                     self.x[2]*self.a[2] + self.x[3]*self.a[3] +
                     self.x[4]*self.a[4] - self.y[1]*self.b[1] -
                     self.y[2]*self.b[2] - self.y[3]*self.b[3] -
                     self.y[4]*self.b[4]) / self.b[0]
        return self.y[0]

    def init_differential_filter(self, initial_value):
        """Iinitialize the differential filter with the initial value."""
        for i in range(self.MAX_FILTER_LENGTH):
            self.x[i] = initial_value


class LeastSquaresFilter(FilterClass):
    """
    Parameters
    ----------
    sample_period : int
        sample time in seconds
    filter_order : int (4 or 8)
        order for the filter.
    """

    def __init__(self, sample_period, filter_order):
        super().__init__()
        self.clear()  # prepare the filter
        freq = 1.0/float(sample_period)  # Define the frequency

        if filter_order == 4:
            self.num_coefs = 4
            self.b[0] = -0.3*freq
            self.b[1] = -0.1*freq
            self.b[2] = 0.1*freq
            self.b[3] = 0.3*freq
        elif filter_order == 8:
            self.num_coefs = 8
            self.b[0] = -0.0833*freq
            self.b[1] = -0.0595*freq
            self.b[2] = -0.0357*freq
            self.b[3] = -0.0119*freq
            self.b[4] = 0.0119*freq
            self.b[5] = 0.0357*freq
            self.b[6] = 0.0595*freq
            self.b[7] = 0.0833*freq
        else:  # Fail gracefully
            self.num_coefs = 2
            self.b[0] = -freq
            self.b[1] = freq


class MovingAverageFilter(FilterClass):
    def __init__(self, filter_order):
        super().__init__()
        self.clear()
        if filter_order > self.MAX_FILTER_LENGTH:
            filter_order = self.MAX_FILTER_LENGTH
        self.num_coefs = filter_order
        C = 1.0/float(filter_order)
        for cnt in range(filter_order):
            self.a[cnt] = C


class ButterworthFilter(FilterClass):
    """Build a Butterwoth filter.

    A Butterworth filter is a maximally flat magnitude filter.

    Parameters
    ----------
    sample_period : int
        Sample period
    cutoff : list
        Cutoff frequency in hertz
    filter_order : int
        Order (1, 2, 3, or 4)
    """

    def __init__(self, sample_period, cutoff, filter_order):
        super().__init__()
        self.clear()
        C = 1.0/math.tan(math.pi * cutoff * sample_period)
        if filter_order == 1:
            A = 1.0/(1.0+C)
            self.a[0] = A
            self.a[1] = A
            self.b[0] = 1.0
            self.b[1] = (1.0-C)*A

        elif filter_order == 2:
            A = 1.0/(1.0+1.4142135623730950488016887242097*C+math.pow(C, 2))
            self.a[0] = A
            self.a[1] = 2*A
            self.a[2] = A

            self.b[0] = 1.0
            self.b[1] = (2.0-2*math.pow(C, 2))*A
            self.b[2] = (
                1.0-1.4142135623730950488016887242097*C+math.pow(C, 2))*A

        elif filter_order == 3:
            A = 1.0/(1.0+2.0*C+2.0*math.pow(C, 2)+math.pow(C, 3))
            self.a[0] = A
            self.a[1] = 3*A
            self.a[2] = 3*A
            self.a[3] = A

            self.b[0] = 1.0
            self.b[1] = (3.0+2.0*C-2.0*math.pow(C, 2)-3.0*math.pow(C, 3))*A
            self.b[2] = (3.0-2.0*C-2.0*math.pow(C, 2)+3.0*math.pow(C, 3))*A
            self.b[3] = (1.0-2.0*C+2.0*math.pow(C, 2)-math.pow(C, 3))*A

        elif filter_order == 4:
            A = 1.0 / (1 + 2.6131259*C + 3.4142136*math.pow(C, 2)
                       + 2.6131259*math.pow(C, 3) + math.pow(C, 4))
            self.a[0] = A
            self.a[1] = 4.0*A
            self.a[2] = 6.0*A
            self.a[3] = 4.0*A
            self.a[4] = A

            self.b[0] = 1.0
            self.b[1] = (
                4.0 + 2.0 * 2.6131259 * C - 2.0 * 2.6131259 * math.pow(C, 3)
                - 4.0 * math.pow(C, 4)) * A
            self.b[2] = (6.0*math.pow(C, 4)-2.0*3.4142136*math.pow(C, 2)+6.0)*A
            self.b[3] = (4.0 - 2.0 * 2.6131259 * C
                         + 2.0 * 2.6131259 * math.pow(C, 3)
                         - 4.0 * math.pow(C, 4)) * A
            self.b[4] = (
                1.0 - 2.6131259 * C + 3.4142136 * math.pow(C, 2) - 2.6131259 *
                math.pow(C, 3) + math.pow(C, 4)) * A
        else:
            raise ValueError('filter_order must be 1, 2, 3, or 4. '
                             'Got filter_order={}'.format(filter_order))


class ButterworthDifferentiator(FilterClass):
    """Build a butterwoth differentiator.

    Generate the filter coefficients where a[0] is the zero order factor
    in the numerator

    Parameters
    ----------
    sample_period : int
        sample time in seconds
    cutoff : float
        cutoff frequency in hertz
    filter_order : int (1, 2, 3, or 4)
        order of the filter
    """

    def __init__(self, sample_period, cutoff, filter_order):
        super().__init__()
        self.clear()
        C = 1.0/math.tan(math.pi*cutoff*sample_period)

        if (filter_order == 1):
            self.a[0] = 1.0
            self.a[1] = -1.0
            self.b[0] = sample_period*(1.0+C)/2.0
            self.b[1] = sample_period*(1.0-C)/2.0

        elif (filter_order == 2):
            self.a[0] = 1.0
            self.a[1] = 0.0
            self.a[2] = -1.0
            self.b[0] = sample_period/2.0*(1+1.414213562373095*C+math.pow(C, 2))
            self.b[1] = sample_period/2.0*(2-2*math.pow(C, 2))
            self.b[2] = sample_period/2.0*(1-1.414213562373095*C+math.pow(C, 2))

        elif (filter_order == 3):
            self.a[0] = 1.0  # 2*(sample_period/2) multiplicative factor
            self.a[1] = 1.0
            self.a[2] = -1.0
            self.a[3] = -1.0
            self.b[0] = sample_period / (
                2.0 * (1.0 + 2*C + 2.0*math.pow(C, 2) + math.pow(C, 3)))
            self.b[1] = sample_period / (
                2.0 * (3.0 + 2*C - 2.0*math.pow(C, 2) - 3.0*math.pow(C, 3)))
            self.b[2] = sample_period / (
                2.0 * (3.0 - 2*C - 2.0*math.pow(C, 2) + 3.0*math.pow(C, 3)))
            self.b[3] = sample_period / (
                2.0 * (1.0 - 2*C + 2.0*math.pow(C, 2) - math.pow(C, 3)))

        elif (filter_order == 4):
            self.a[0] = 1.5  # 3*(sample_period/2) multiplicative factor
            self.a[1] = 1.0  # 2*(sample_period/2) multiplicative factor
            self.a[2] = 0.0
            self.a[3] = -1.0
            self.a[4] = -1.5
            self.b[0] = sample_period / (
                2.0 * (1.0 + 2.6131259*C + 3.4142136*math.pow(C, 2)
                       + 2.6131259*math.pow(C, 3) + math.pow(C, 4)))
            self.b[1] = sample_period / (
                2.0 * (4.0 + 2*2.6131259*C - 2.0*2.6131259*math.pow(C, 3)
                       - 4.0*math.pow(C, 4)))
            self.b[2] = sample_period / (
                2.0 * (6.0*math.pow(C, 4) - 2.0*3.4142136*math.pow(C, 2) + 6.0))
            self.b[3] = sample_period / (
                2.0 * (4.0 - 2*2.6131259*C + 2.0*2.6131259*math.pow(C, 3)
                       - 4.0*math.pow(C, 4)))
            self.b[4] = sample_period / (
                2.0 * (1.0 - 2.6131259*C + 3.4142136*math.pow(C, 2)
                       - 2.6131259*math.pow(C, 3) + math.pow(C, 4)))
        else:
            raise ValueError('filter_order must be 1, 2, 3, or 4. '
                             'Got filter_order={}'.format(filter_order))


class KalmanFilter(object):
    def __repr__(self):
        return str(self.__class__)

    def __init__(self,
                 process_variance,
                 estimated_measurement_variance,
                 posteri_estimate=0.0,
                 posteri_error_estimate=1.0):
        self.process_variance = process_variance
        self.estimated_measurement_variance = estimated_measurement_variance
        self.posteri_estimate = posteri_estimate
        self.posteri_error_estimate = posteri_error_estimate

    def input_latest_noisy_measurement(self, measurement):
        priori_estimate = self.posteri_estimate
        priori_error_estimate = (self.posteri_error_estimate
                                 + self.process_variance)

        blending_factor = priori_error_estimate / (
            priori_error_estimate + self.estimated_measurement_variance)
        self.posteri_estimate = priori_estimate + \
            blending_factor * (measurement - priori_estimate)
        self.posteri_error_estimate = ((1 - blending_factor)
                                       * priori_error_estimate)
        return self.posteri_estimate

    def get_latest_estimated_measurement(self):
        return self.posteri_estimate


class BinaryFilter(object):  # TODO need some changes
    def __repr__(self):
        return str(self.__class__)

    # trigger based on boolean judgement
    def __init__(self, initState, counter):
        self.counterActivation = 0
        self.isActivation = initState
        self.max_count_activation = counter
        self.y = initState
        self.initState = initState

    def __call__(self, contactTrigger):
        if (contactTrigger == 1) and (not self.isActivation):
            self.counterActivation += 1
        elif (self.counterActivation >= 0) and (not contactTrigger):
            self.counterActivation -= 1

        if(self.counterActivation > self.max_count_activation):
            self.isActivation = 1
        elif self.counterActivation < 0:
            self.isActivation = 0
        self.y = self.isActivation
        return self.isActivation

    def clear(self):
        self.counterActivation = 0
        self.isActivation = self.initState
        self.y = self.initState
        return


class EMAFilter():
    def __repr__(self):
        return str(self.__class__)

    def __init__(self, gamma):
        self.y = 0.0
        self.gamma = gamma

    def __call__(self, X):
        self.y = self.y*(1.0-self.gamma) + self.gamma*X
        return self.y

    def initialise_y(self, y):
        self.y = y
        return self.y
