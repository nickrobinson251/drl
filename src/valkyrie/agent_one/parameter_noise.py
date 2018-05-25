class AdaptiveParamNoise(object):
    def __init__(self,
                 initial_stddev=0.1,
                 desired_action_stddev=0.1,
                 adoption_coefficient=1.01):
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adoption_coefficient = adoption_coefficient
        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance > self.desired_action_stddev:  # decrease stddev
            self.current_stddev /= self.adoption_coefficient
        else:  # increase stddev
            self.current_stddev *= self.adoption_coefficient
        return self.current_stddev

    def get_stats(self):
        stats = {'param_noise_stddev': self.current_stddev}
        return stats

    def __repr__(self):
        string = "AdaptiveParamNoiseSpec(initial_stddev={}, "
        "desired_action_stddev={}, adoption_coefficient={})"
        return string.format(
            self.initial_stddev,
            self.desired_action_stddev,
            self.adoption_coefficient)
