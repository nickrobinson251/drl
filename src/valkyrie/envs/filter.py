import math
import numpy as np

MAX_FILTER_LENGTH = 8


class FilterClass():
    def __repr__(self):
        return str(self.__class__)

    def __init__(self):
        """Initialize all variables to 0."""
        self.Ncoeff = 6  # number of coefficients
        self.i = 0
        self.x = np.zeros(MAX_FILTER_LENGTH)
        self.y = np.zeros(MAX_FILTER_LENGTH)
        self.a = np.zeros(MAX_FILTER_LENGTH)
        self.b = np.zeros(MAX_FILTER_LENGTH)
        self.InitF = False

    def init_diff_filter(self, X):
        """Iinitialize the differential filter with the initial value."""
        for i in range(MAX_FILTER_LENGTH):
            self.x[i] = X

    def clear_filter(self):
        """Clear the filter as constructor does."""
        self.x = np.zeros(MAX_FILTER_LENGTH)
        self.y = np.zeros(MAX_FILTER_LENGTH)
        self.a = np.zeros(MAX_FILTER_LENGTH)
        self.b = np.zeros(MAX_FILTER_LENGTH)

    def least_squares_filter(self, T, N):
        """T ias sample time in seconds, N is the order for the filter."""
        self.clear_filter()  # prepare the filter
        freq = 1.0/float(T)  # Define the frequency

        # according to the order N do
        if N == 4:
            self.Ncoeff = 4
            self.b[0] = -0.3*freq
            self.b[1] = -0.1*freq
            self.b[2] = 0.1*freq
            self.b[3] = 0.3*freq
        elif N == 8:
            self.Ncoeff = 8
            self.b[0] = -0.0833*freq
            self.b[1] = -0.0595*freq
            self.b[2] = -0.0357*freq
            self.b[3] = -0.0119*freq
            self.b[4] = 0.0119*freq
            self.b[5] = 0.0357*freq
            self.b[6] = 0.0595*freq
            self.b[7] = 0.0833*freq
        else:  # Fail gracefully
            self.Ncoeff = 2
            self.b[0] = -freq
            self.b[1] = freq

    def moving_average_filter(self, N):
        self.clear_filter()
        if N > MAX_FILTER_LENGTH:
            N = MAX_FILTER_LENGTH
        self.clear_filter()
        self.Ncoeff = N
        C = 1.0/float(N)
        for cnt in range(N):
            self.a[cnt] = C

    def butterworth(self, T, cutoff, N):
        """Build a butterwoth filter.

        Parameters
        ----------
        T : int
            Sample period
        cutoff : list
            Cutoff frequency in hertz
        N : int
            Order (1, 2, 3, or 4)
        """
        self.clear_filter()
        C = 1.0/math.tan(math.pi * cutoff * T)

        if N == 1:
            A = 1.0/(1.0+C)
            self.a[0] = A
            self.a[1] = A
            self.b[0] = 1.0
            self.b[1] = (1.0-C)*A

        elif N == 2:
            A = 1.0/(1.0+1.4142135623730950488016887242097*C+math.pow(C, 2))
            self.a[0] = A
            self.a[1] = 2*A
            self.a[2] = A

            self.b[0] = 1.0
            self.b[1] = (2.0-2*math.pow(C, 2))*A
            self.b[2] = (
                1.0-1.4142135623730950488016887242097*C+math.pow(C, 2))*A

        elif N == 3:
            A = 1.0/(1.0+2.0*C+2.0*math.pow(C, 2)+math.pow(C, 3))
            self.a[0] = A
            self.a[1] = 3*A
            self.a[2] = 3*A
            self.a[3] = A

            self.b[0] = 1.0
            self.b[1] = (3.0+2.0*C-2.0*math.pow(C, 2)-3.0*math.pow(C, 3))*A
            self.b[2] = (3.0-2.0*C-2.0*math.pow(C, 2)+3.0*math.pow(C, 3))*A
            self.b[3] = (1.0-2.0*C+2.0*math.pow(C, 2)-math.pow(C, 3))*A

        elif N == 4:
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
            raise ValueError('N must be 1, 2, 3, or 4. Got N={}'.format(N))

    def initializeFilter(self, X):
        for i in range(MAX_FILTER_LENGTH):
            self.x[i] = X
            self.y[i] = X

    def applyFilter(self, X):
        """Return filter signal from new input.

        Assumes the filter was run so that the coefficients are available
        """
        self.x[4] = self.x[3]
        self.x[3] = self.x[2]
        self.x[2] = self.x[1]
        self.x[1] = self.x[0]
        self.x[0] = X
        self.y[4] = self.y[3]
        self.y[3] = self.y[2]
        self.y[2] = self.y[1]
        self.y[1] = self.y[0]
        self.y[0] = self.a[0]*self.x[0]+self.x[1]*self.a[1] +\
            self.x[2]*self.a[2]+self.x[3]*self.a[3] +\
            self.x[4]*self.a[4]-self.y[1]*self.b[1]-self.y[2]*self.b[2] -\
            self.y[3]*self.b[3]-self.y[4]*self.b[4]
        return self.y[0]

    def butterDifferentiator(self, T, cutoff, N):
        """Build a butterwoth differentiator.

        T is the sample period
        cutoff is the cutoff frequency in hertz
        N is the order of the filter (1, 2, 3, or 4)
        Generate the filter coefficients where a[0] is the zero order factor
        in the numerator
        """
        self.clear_filter()
        C = 1.0/math.tan(math.pi*cutoff*T)

        if (N == 1):
            self.a[0] = 1.0
            self.a[1] = -1.0
            self.b[0] = T*(1.0+C)/2.0
            self.b[1] = T*(1.0-C)/2.0

        elif (N == 2):
            self.a[0] = 1.0
            self.a[1] = 0.0
            self.a[2] = -1.0
            self.b[0] = T/2.0*(1+1.414213562373095*C+math.pow(C, 2))
            self.b[1] = T/2.0*(2-2*math.pow(C, 2))
            self.b[2] = T/2.0*(1-1.414213562373095*C+math.pow(C, 2))

        elif (N == 3):
            self.a[0] = 1.0  # 2*(T/2) multiplicative factor
            self.a[1] = 1.0
            self.a[2] = -1.0
            self.a[3] = -1.0
            self.b[0] = T/2.0*(1.0+2*C+2.0*math.pow(C, 2)+math.pow(C, 3))
            self.b[1] = T/2.0*(3.0+2*C-2.0*math.pow(C, 2)-3.0*math.pow(C, 3))
            self.b[2] = T/2.0*(3.0-2*C-2.0*math.pow(C, 2)+3.0*math.pow(C, 3))
            self.b[3] = T/2.0*(1.0-2*C+2.0*math.pow(C, 2)-math.pow(C, 3))

        elif (N == 4):
            self.a[0] = 1.5  # 3*(T/2) multiplicative factor
            self.a[1] = 1.0  # 2*(T/2) multiplicative factor
            self.a[2] = 0.0
            self.a[3] = -1.0
            self.a[4] = -1.5
            self.b[0] = T/2.0*(1.0 + 2.6131259*C + 3.4142136*math.pow(C, 2)
                               + 2.6131259*math.pow(C, 3) + math.pow(C, 4))
            self.b[1] = T/2.0*(4.0+2*2.6131259*C-2.0*2.6131259*math.pow(C, 3)
                               - 4.0*math.pow(C, 4))
            self.b[2] = T/2.0*(6.0*math.pow(C, 4)
                               - 2.0*3.4142136*math.pow(C, 2)+6.0)
            self.b[3] = T/2.0*(4.0-2*2.6131259*C+2.0*2.6131259*math.pow(C, 3)
                               - 4.0*math.pow(C, 4))
            self.b[4] = T/2.0*(1.0-2.6131259*C+3.4142136*math.pow(C, 2)
                               - 2.6131259*math.pow(C, 3)+math.pow(C, 4))
        else:
            raise ValueError('N must be 1, 2, 3, or 4. Got N={}'.format(N))

    def differentiator(self, X):
        """Return filter signal from new input.

        Assumes the filter was run so that the coefficients are available
        """
        self.x[4] = self.x[3]
        self.x[3] = self.x[2]
        self.x[2] = self.x[1]
        self.x[1] = self.x[0]
        self.x[0] = X
        self.y[4] = self.y[3]
        self.y[3] = self.y[2]
        self.y[2] = self.y[1]
        self.y[1] = self.y[0]
        self.y[0] = (self.a[0]*self.x[0]+self.x[1]*self.a[1] +
                     self.x[2]*self.a[2]+self.x[3]*self.a[3] +
                     self.x[4]*self.a[4]-self.y[1]*self.b[1] -
                     self.y[2]*self.b[2]-self.y[3]*self.b[3] -
                     self.y[4]*self.b[4])/self.b[0]
        return self.y[0]


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
        self.posteri_error_estimate = (
            1 - blending_factor) * priori_error_estimate
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

    def applyFilter(self, contactTrigger):
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

    def clearFilter(self):
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

    def applyFilter(self, X):
        self.y = self.y*(1.0-self.gamma) + self.gamma*X
        return self.y

    def initializeFilter(self, init):
        self.y = init
        return self.y
