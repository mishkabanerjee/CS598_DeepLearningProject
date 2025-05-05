import numpy as np

matrix = np.load(r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\npy\patient_30000.npy")
print(matrix.shape)

print(matrix[:5])  # First 5 time steps

print(np.isnan(matrix).mean())

import matplotlib.pyplot as plt

plt.plot(matrix[:, 0])  # Plot first variable over time
plt.title("vm1 over time for patient 30000")
plt.xlabel("Time steps")
plt.ylabel("Value")
plt.show()

