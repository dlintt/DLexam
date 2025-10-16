import numpy as np
import matplotlib.pyplot as plt

ENABLE_PLOTTING = True

print("--- Z-Score Method ---")
threshold_z = 3
sample_z = [15, 101, 125, 18, 7, 13, 16, 11, 21, 5, 15, 10, 9]
mean = np.mean(sample_z)
std = np.std(sample_z)
outlier_z = []
for i in sample_z:
    zscore = (i - mean) / std
    if abs(zscore) > threshold_z:
        outlier_z.append(i)
print(f"Outliers: {outlier_z}")

if ENABLE_PLOTTING:
    plt.figure(figsize=(4, 2))
    plt.boxplot(sample_z, vert=False)
    plt.title("Z-Score Method Sample")
    plt.xlabel("Values")
    plt.grid(True)
    plt.show()

print("\n" + "="*25 + "\n")

print("--- IQR Method ---")
sample_iqr = [15, 101, 125, 18, 7, 13, 16, 11, 21, 5, 15, 10, 9]
asc = sorted(sample_iqr)
q1 = np.percentile(asc, 25)
q3 = np.percentile(asc, 75)
IQR = q3 - q1
threshold_iqr = 1.5 * IQR
low = q1 - threshold_iqr
up = q3 + threshold_iqr
outlier_iqr = []
for i in asc:
    if i < low or i > up:
        outlier_iqr.append(i)
print(f"Outliers: {outlier_iqr}")

if ENABLE_PLOTTING:
    plt.figure(figsize=(4, 2))
    plt.boxplot(asc, vert=False)
    plt.title("IQR Method Sample")
    plt.xlabel("Values")
    plt.grid(True)
    plt.show()