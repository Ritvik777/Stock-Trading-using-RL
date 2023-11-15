#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Akash Yadav-301426295 -akashyadav@mail.fresnostate.edu

#Raghav -301390675- raghav8900@mail.fresnostate.edu

#Ritvik Gaur- 301426477- ritvik777@mail.fresnostate.edu


"""
Created on Wed Nov 16 15:13:02 2022

@author: ritvik
"""

import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', type=str, required=True,
                    help='either "train" or "test"')
args = parser.parse_args()

a = np.load(f'linear_rl_trader_rewards/{args.mode}.npy')

print(f"average reward: {a.mean():.2f}, min: {a.min():.2f}, max: {a.max():.2f}")

if args.mode == 'train':
  # show the training progress
  plt.plot(a)
else:
  # test - show a histogram of rewards
  plt.hist(a, bins=20)

plt.title(args.mode)
plt.show()