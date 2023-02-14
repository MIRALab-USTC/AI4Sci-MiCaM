import math

from model.mydataclass import TrainingParams

class scheduler(object):
    def __init__(self, params: TrainingParams, init_beta: int, init_step: int) -> None:
        
        self.beta = init_beta
        self.t = init_step
        self.warmup = params.beta_warmup
        self.beta_min = params.beta_min
        self.beta_max = params.beta_max
        self.beta_anneal_period = params.beta_anneal_period
    
    def step(self):
        pass

class sigmoid_schedule(scheduler):
    def __init__(self, params: TrainingParams, init_beta: int, init_step: int) -> None:
        super(sigmoid_schedule, self).__init__(params, init_beta, init_step)
        self.diff = self.beta_max - self.beta_min
        self.anneal_rate = math.pow(0.01, 1 / self.beta_anneal_period)
        self.weight = 1
        
    def step(self):
        if self.t < self.warmup:
            self.beta = self.beta_min
        else:
            self.weight = math.pow(self.anneal_rate, self.t - self.warmup)
            self.beta = self.beta_min + self.diff * (1 - self.weight)
        self.t += 1
        return self.beta

class beta_annealing_schedule(object):
    def __init__(self, params: TrainingParams, init_beta: int=0, init_step: int=0) -> None:
        self.schedule = sigmoid_schedule(params, init_beta, init_step)
    
    def step(self):
        return self.schedule.step()