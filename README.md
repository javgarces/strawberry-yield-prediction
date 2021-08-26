# :strawberry: strawberry-yield-prediction
Python project simulating strawberries growing through stages. Goal: building a prediction model to estimate future yield.

## Instructions

Execute *Strawberry_Yield_Prediction.py* script.
The script generates a set of graphs showcasing simulation data and prediction model results.

## Example Plots

### Strawberry growth Area plot

In this simulation, a number of strawberries grow through various stages at different days. This plot shows the number of strawberries at each stage per day.

![SB_area](../assets/SB_area.png?raw=true)

### Bar plot: Model objective and result

This graph presents an example of the predictive model results by showcasing the amount of strawberries per stage at the following days:

- Day X to Day 0: a set of days used as input data for the trained model.
- Goal: the corresponding day used as output data for the trained model.
- Pred.: the prediction result by evaluating the input data (validation data) using the trained model.

![barplot](../assets/barplot.png?raw=true)

## Sources

https://www.youtube.com/watch?v=6mL_p2CXVVw (to obtain stage length estimations)
