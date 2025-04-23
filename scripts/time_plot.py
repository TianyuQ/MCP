from cv2 import line
import matplotlib.pyplot as plt

# Data points
y_values = [2.068037986755371, 5.638650894165039, 45.33264899253845, 65.00687885284424,
            124.78684997558594, 204.49898099899292, 323.0447630882263,
            536.2500178813934, 658.0406320095062]

y_values = [yvalue / 9 for yvalue in y_values]

# Generate x values based on the index of y values
x_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]
# x_values = list(range(len(y_values)))
y2_values = [xvalue ** 3 / 15 for xvalue in x_values]

# Create the line plot
plt.figure(figsize=(10, 4))
plt.plot(x_values, y_values, marker='o', linestyle='-', linewidth=3.5)
plt.plot(x_values, y2_values, marker='x', linestyle='--', linewidth=3)
# plt.title("Computation Time vs Number of Players")
plt.legend(["Computation Time", "O(N^3)"], fontsize=14)
plt.xlabel("Number of Players", fontsize=14)
plt.ylabel("Computation Time [s]", fontsize=14)
plt.xticks(x_values, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig("time_plot.pdf", dpi=1000)
# plt.show()
