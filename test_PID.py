#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 13:05:06 2022

@author: antonio
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

#%%
x = np.arange(1000)
ar = np.array([0, 0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,0,0,0,0,0,0,0,0,0,0,0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
#%%
prev_act = 0
for a in ar:
    y = a-0.25*(a-prev_act)
    prev_act = y
    print(y)
    
#%%
kp = 1#-0.6*ku
kd = -0.05#kp*12/8
ki = -0.01#2*kp/1250
q_err_ste = deque(np.zeros(10), maxlen=10)
prev_err_ste = 0
prev_act_ste = 0

actions = []
for a in ar:
    et_ste = a - prev_act_ste
    q_err_ste.append(et_ste)
    
    action_ste = kp*et_ste + kd*(et_ste - prev_err_ste)# + ki*np.array(q_err_ste).sum() #v1
    actions.append(action_ste)
    prev_err_ste = et_ste
    prev_act_ste = action_ste
    print(a, action_ste)
    
# plt.plot(ar)
plt.plot(actions)
#%%
