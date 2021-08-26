# -*- coding: utf-8 -*-
"""
Created on Fri May 21 13:29:26 2021

@author: jgarcesalmon
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

#%% Plots

stages = [
    "No SB",
    "Bud",
    "Flower",
    "Green",
    "White",
    "Red (picked)",
    "Already picked"
]

def plot_single_SB(SB, stages=stages):
    # data and plot
    
    x = np.arange(SB.size)
    y = SB
    fig, ax = plt.subplots()
    ax.step(x, y)
    
    # details, design
    
    ax.set_title("Strawberry Stage per day")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Strawberry Stages")
    yticks = list(range(SB[-1] + 1))
    ax.set_yticks(yticks)
    ax.set_yticklabels(stages)
    
def plot_SBs(SBs, stages=stages):
    # data and plot
    
    x = np.arange(SBs.shape[0])
    y = SBs
    fig, ax = plt.subplots()
    for row in y.T:
        ax.step(x, row)
    
    # details, design
    
    ax.set_title("Strawberry Stage per day")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Strawberry Stages")
    yticks = list(range(7))
    ax.set_yticks(yticks)
    ax.set_yticklabels(stages)
    
def_title = "Number of Strawberries per day by Stage"    

def plot_SBs_area(counts, x_from=0, perc=False, dataline_info=False,
                  stages=stages[1:], xticks=False, xlim=False,
                  title=def_title):
    # data and plot
    
    x = np.arange(x_from, x_from + counts.shape[0])
    y = counts if not perc else countperc(counts)
    color = sns.color_palette("Set2")
    fig, ax = plt.subplots()
    ax.stackplot(x,y.T, labels=stages, colors=color)
    
    # datalines?
    if dataline_info is not False:
        T = counts.sum(axis=1).max()*1.3
        input_indexes, output_index, forward = dataline_info
        for n, input_index in enumerate(input_indexes):
            fii = forward + input_index
            if n == 0:
                ax.plot([fii, fii], [0,T], 'r-', label="Model Input")
            else:
                ax.plot([fii, fii], [0,T], 'r-')
        foi = forward + output_index
        ax.plot([foi, foi], [0,T], 'b-', label="Model Output")
    
    # details, design
    
    ax.legend(loc='upper right')
    ax.set_xlabel("Time (days)")
    if not perc:
        ax.set_title(title)
        ax.set_ylabel("Number of Strawberries")
    else:
        ax.set_title("% of Strawberries per day by Stage")
        ax.set_ylabel("% of Strawberries")
    ax.set_xlim((0,x_from + counts.shape[0]))
    if xticks is not False:
        ax.set_xticks(xticks)
    if xlim is not False:
        ax.set_xlim(xlim)
    ax.set_ylim((0, counts.sum(axis=1).max()*1.3))

def plot_SBs_area_datalines(counts, days_design, forward = 15, stages=stages):
    # data and plot
    
    x = np.arange(counts.shape[0])
    y = counts
    color = sns.color_palette("Set2")
    fig, ax = plt.subplots()
    ax.stackplot(x, y.T, labels=stages, colors=color)
    
    # logic for datalines:
    
    T = counts[0].sum()
        
    index_skip = days_design["past_skip"] + 1
    index_start = index_skip * (days_design["input"] - 1)
    
    input_indexes = np.arange(forward, forward + index_start+1, index_skip)
    output_index = forward + index_start + days_design["future_pred"]
    
    for n, input_index in enumerate(input_indexes):
        if n == 0:
            ax.plot([input_index,input_index], [0,T], 'r-', label="Model Input")
        else:
            ax.plot([input_index,input_index], [0,T], 'r-')
    ax.plot([output_index,output_index], [0,T], 'b-', label="Model Output")
    
    # details, design
    
    ax.legend(loc='upper right')
    ax.set_title("Number of Strawberries per day by Stage")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Number of Strawberries")
    