# -*- coding: utf-8 -*-
"""
Modified on Sun Aug 22 2021

@author: Javier Garces

Strawberry prediction model
Data source: Simulated data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import SYP_module as SB

# DAYS_DESIGN
# Defines design of prediction models
# Model takes <input> days, each of them <past_skip> days apart,
# to predict <future_pred> days into the future.

DAYS_DESIGN = {
    "input": 3,
    "past_skip": 7,
    "future_pred": 7
}

# BARPLOT_EXAMPLE_INDEX
# Sets day to use as example for Barplot
# showcasing days used as model input (from validation data),
# goal of prediction, and prediction.

BARPLOT_EXAMPLE_INDEX = 20

# SAVEFIG
# Flag to generate PNG files with graphical results
SAVEFIG = True

#%% Data source: Simulated data

# Strawberries growing through 7 phases.
# Each phase length computed randomly from normal distributions.

# For each strawberry:
#     - Stage 0 until some day t_0 (nothing)
#     - Stage 1 until day t_1 (bud)
#     - Stage 2 until day t_2 (flower)
#     - Stage 3 until day t_3 (green)
#     - Stage 4 until day t_4 (white)
#     - Stage 5 until day t_5 (red)
#     - Stage 6 from t_5 (picked)
# Each stage i lasts s_i days.

N_phases = 7

# Mean duration (days) of each of the (middle) 5 stages
s_i_mean = np.array([ 
    9.8,
    3.5,
    19.8,
    5.8,
    1
    ])

# 68.27% of values will be between mean-std and mean+std
# 95.45% of values will be between mean-2*std and mean+2*std
s_i_std = np.array([ 
    1,
    0.5,
    3,
    0.8,
    0.1,
    ])

# To consider: obtained value could be negative. This is ignored (unlikely).

# T: number of days
# N: number of strawberries

T = 120
N = 500

starting_date_mean = 30
starting_date_std = 12

# Matrix SB_days
# columns: strawberries, number of phase each day
# rows: days

SB_days = np.zeros((T, N), dtype=int)

# populate SB_days randomly simulating strawberries growing through phases

phases_lengths = s_i_std * np.random.randn(N,5) + s_i_mean
starting_dates = starting_date_std * np.random.randn(N,1) + starting_date_mean

# days where each strawberry goes from one phase to the other
ts = np.hstack((starting_dates, np.cumsum(phases_lengths, axis=1) + starting_dates))

# filling SB_days matrix
for n in range(N):
    for t in ts[n]:
        SB_days[int(np.floor(t)):,n] += 1
        
# PLOT 1
# Single strawberry. Ladder plot, going through each phase

SB.plot_single_SB(SB_days[:,0])

if SAVEFIG: plt.savefig("01_single_SB.png")

# PLOT 2
# SB_n strawberries. Ladder plots, going through each phase

SB_n = 5
SB.plot_SBs(SB_days[:,0:SB_n])

if SAVEFIG: plt.savefig("02_SBs.png")
        
#%% Counts: Transform each row (day) into a count.
# How many strawberries are in each phase each day

counts = np.zeros(shape=(0,5), dtype=int)

for row in SB_days: # each row is one day
    # take out 0's and 7's
    # we dont want to count how many SB have not grown
    # nor how many SB where plucked already
    row = np.array([i for i in row if i not in [0, 6]], dtype=int)
    # minus one to use bincount
    row = row - 1
    # counts per number 
    bincount = np.bincount(row)
    zeros = (N_phases - 2) - bincount.size
    count = np.hstack((bincount, np.zeros(shape=(zeros,), dtype=int)))
    counts = np.vstack((counts, count))
    
# PLOT 3
# Counts. Area plot, amount of strawberries in each phase for each day.
    
SB.plot_SBs_area(counts)

if SAVEFIG: plt.savefig("03_SB_area.png")
    
#%% Model inputs: Train and test data

# logic:
index_skip = DAYS_DESIGN["past_skip"] + 1
index_start = index_skip * (DAYS_DESIGN["input"] - 1)

N_new = T - index_start - DAYS_DESIGN["future_pred"]
input_indexes = np.arange(0, index_start+1, index_skip)
output_index = index_start + DAYS_DESIGN["future_pred"]

# input and output matrices
 
input_matrix = np.zeros(shape=(N_new, 0))
for i in input_indexes[::-1]:
    input_matrix = np.hstack((input_matrix, counts[i:i+N_new,:]))

output_matrix = counts[output_index:output_index+N_new,:]

# PLOT 4
# Same as PLOT 3,
# with added vertical lines:
# Red lines representing input data
# Blue line representing output data (target)

vertical_lines_from = 15
dataline_info = (input_indexes, output_index, vertical_lines_from)

SB.plot_SBs_area(counts, dataline_info=dataline_info)

if SAVEFIG: plt.savefig("04_SB_area_model_input.png")

#%% Prediction model: Neural Network

from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(input_matrix,
                                                  output_matrix,
                                                  random_state = 0,
                                                  test_size=0.4)

early_stopping = callbacks.EarlyStopping(
    min_delta=0.001, # minimum amount of change to count as an improvement
    patience=20, # how many epochs to wait before stopping
    restore_best_weights=True,
)

model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=[train_X.shape[1]]),
    # layers.Dropout(0.3),
    # layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    # layers.Dropout(0.3),
    # layers.BatchNormalization(),
    layers.Dense(train_y.shape[1]),
])

model.compile(
    optimizer='adam',
    loss='mae',
)

history = model.fit(
    train_X, train_y,
    # validation_data=(val_X, val_y),
    validation_split=0.2,
    batch_size=256,
    epochs=100,
    callbacks=[early_stopping],
    verbose=0,
)

# PLOT 5
# Loss at each loop of the model training procedure

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.figure()

plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.title('Training and validation loss')
plt.yscale('log')
plt.legend()

if SAVEFIG: plt.savefig("05_loss.png")


# PLOT 6 and 7
# Comparisons of real output data vs output data from prediction model

stages = [
    "Bud",
    "Flower",
    "Green",
    "White",
    "Red - Picked"
    ]

# plot for all data

all_prediction = model.predict(input_matrix)
    
SB.plot_SBs_area(output_matrix, stages=stages)

if SAVEFIG: plt.savefig("06_model_goal_data.png")

SB.plot_SBs_area(all_prediction, stages=stages,
                 title = 'Predicted Number of SB per day by Stage')

if SAVEFIG: plt.savefig("07_model_predicted_data.png")


#%% PLOT 8: Bar Plot

# Data for bar plot

sN = len(stages)

vi = BARPLOT_EXAMPLE_INDEX

output_prediction = model.predict(val_X)

# Populate data

day_bars = list()
names = list()

# days from model input (validation data)
for i in reversed(range(DAYS_DESIGN['input'])):
    day_bars.append(val_X[vi][sN*i : sN*(i+1)].astype(int))
    names.append("Day {}".format(i))

# day from model output (validation data)
day_bars.append(val_y[vi])
names.append("Goal")

# day from model prediction
day_bars.append(output_prediction[vi])
names.append("Pred.")

# Draw bar plot

values = np.array(day_bars).T

width = 0.8
figlength = 12.0

fig, ax1 = plt.subplots(1, 1, figsize=(figlength,figlength/2.5))
ax1.set_xlabel('Example of Input and Output data + Prediction')
ax1.set_ylabel('Counts per Phase')
ax1.grid(True)
ax1.set_axisbelow(True)
total_data = len(values)
classes_num = np.arange(len(names))
for i in range(total_data):
    bars = plt.bar(classes_num - width / 2. + i / total_data * width,
                   values[i], 
                   width = width / total_data,
                   align="edge", animated=0.4)
    for rect in bars:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0,
                 height, '{:.0f}'.format(height), ha='center', va='bottom',
                 size='large')
plt.xticks(classes_num, names, rotation=0)
plt.legend(stages)    
ax1.set_ylim((0, ax1.get_ylim()[1] + 2))

if SAVEFIG: plt.savefig("08_model_day_prediction_barplot.png")

plt.pause(0.001)
