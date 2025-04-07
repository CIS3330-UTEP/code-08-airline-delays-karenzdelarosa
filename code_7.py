import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np 

#If any of this libraries is missing from your computer. Please install them using pip.

filename = 'Flight_Delays_2018.csv'
df = pd.read_csv(filename)

df = df[['ARR_DELAY', 'CARRIER_DELAY']]

print(df.describe())

Y = df['ARR_DELAY']
X = df[['CARRIER_DELAY']]

X = sm.add_constant(X)
model = sm.OLS(Y,X).fit()
print(model.summary())

plt.figure(figsize=(10,6))
plt.scatter(df["CARRIER_DELAY"], df["ARR_DELAY"], color='purple', alpha=0.5)
plt.plot(df["CARRIER_DELAY"], model.params[0] + model.params[1]*df["CARRIER_DELAY"], color='red')

plt.text(0, 10, f"y = {model.params[0]:.3f} + {model.params[1]:.3f}x", size=12)

plt.xlabel("CARRIER_DELAY")
plt.ylabel("ARR_DELAY")
plt.title("ARRIVAL DELAY VS CARRIES DELAY")

plt.show()

#ARR_DELAY is the column name that should be used as dependent variable (Y)
