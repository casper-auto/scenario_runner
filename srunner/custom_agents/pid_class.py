#!/usr/bin/env python

import numpy as np
# import matplotlib.pyplot as plt
from collections import deque

np.random.seed(3)


class PID:

    def __init__(self, dt):
        self.integral_buffer = deque()
        self.max_buffer = 100 # must be large 10000
        self.dt = dt 

        self.max_error = 2.0

        # # Ziegler Nichols Tuning
        # # https://en.wikipedia.org/wiki/Ziegler%E2%80%93Nichols_method
        # self.Kp = 0.07 # I don't really get oscillations, but it goes unstable above 0.02
        # Tu = 35*self.dt#35*self.dt # no oscillations from Kp, but small Tu's give osc, but this is a single number to tune
        # self.Ki = self.Kp/(Tu/2) #0.001 # stable I only = 0.001  # Kp/(Tu/2) = 0.03
        # self.Kd = self.Kp*(Tu/8) #.3 #1 #Kp*(Tu/8)  
        # print("numbers:", self.Kp/(Tu/2), self.Kp*(Tu/8) ) 0.4 0.4

        # Good, but I want it faster
        self.Kp = 0.4 #0.3 #0.2 #0.07 # I don't really get oscillations, but it goes unstable above 0.02
        self.Ki = 0.005/self.dt #self.Kp/(Tu/2) #0.001 # stable I only = 0.001  # Kp/(Tu/2) = 0.03
        self.Kd = 0.4*self.dt # self.Kp*(Tu/8) #.3 #1 #Kp*(Tu/8)

        # self.Kp = 0.8 #0.3 #0.2 #0.07 # I don't really get oscillations, but it goes unstable above 0.02
        # self.Ki = 0.03/self.dt #self.Kp/(Tu/2) #0.001 # stable I only = 0.001  # Kp/(Tu/2) = 0.03
        # self.Kd = 0# 0.1*self.dt # self.Kp*(Tu/8) #.3 #1 #Kp*(Tu/8)
        

    def control(self, x, target):
        error = target - x

        if len(self.integral_buffer)>0:
            derivative = (error - self.integral_buffer[-1])/self.dt
        else: 
            derivative = 0

        if np.abs(error)>self.max_error:
            error=np.sign(error)*self.max_error

        self.integral_buffer.append(error)
        if len(self.integral_buffer)>self.max_buffer:
            self.integral_buffer.popleft()
        integral = np.sum(np.asarray(self.integral_buffer))*self.dt

        # print("integral", integral)
        # # wind up
        # self.max_integral = 50
        # if np.abs(integral)>self.max_integral:
        #   integral = np.sign(integral)*self.max_integral

        pid = self.Kp*error + self.Ki*integral + self.Kd*derivative 
        return pid

    def tune(self):
        # TODO
        pass
