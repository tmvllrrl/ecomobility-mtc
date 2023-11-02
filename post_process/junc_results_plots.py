import csv

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid", {"grid.linestyle": "--"})

# Initialize a graph
fig, ax = plt.subplots(figsize=(8,8), dpi=300)

x = [10,20,30,40,50,60,70,80,90,100]
fs = 24

y = [59.36, 50.50, 38.62, 27.91, 22.04, 18.91, 20.77, 18.62, 14.45, 12.33]
ax.hlines(26.08, 10, 100, color="black", label="HVs w/ TS", linestyles="--", linewidth=2)
ax.plot(x, y, color="forestgreen", label="RV Pen. Rates", linewidth=2)
ax.set_title("Wait Time Across Intersections", fontsize=fs)
ax.set_ylabel("Wait Time (s)", fontsize=fs)

# y = [0.7138, 0.7022, 0.6162, 0.5821, 0.5796, 0.5451, 0.5321, 0.5686, 0.5468, 0.5427]
# ax.hlines(0.7332, 10, 100, color="black", label="HVs w/ TS", linestyles="--", linewidth=2)
# ax.plot(x, y, color="slateblue", label="RV Pen. Rates", linewidth=2)
# ax.set_title("Fuel Consumption Across Intersections", fontsize=fs)
# ax.set_ylabel("Fuel Consumption (ml/s)", fontsize=fs)

# y = [1754.74, 1628.30, 1456.71, 1356.16, 1315.61, 1250.54, 1254.18, 1290.50, 1271.13, 1284.51]
# ax.hlines(1705.53, 10, 100, color="black", label="HVs w/ TS", linestyles="--", linewidth=2)
# ax.plot(x, y, color="lightcoral", label="RV Pen. Rates", linewidth=2)
# ax.set_title("CO$_2$ Emissions Across Intersections", fontsize=fs)
# ax.set_ylabel("CO$_2$ Emissions (mg/s)", fontsize=fs)

# y = [90.45, 86.33, 70.83, 63.00, 60.72, 53.65, 51.18, 56.72, 48.89, 47.60]
# ax.hlines(83.76, 10, 100, color="black", label="HVs w/ TS", linestyles="--", linewidth=2)
# ax.plot(x, y, color="lightcoral", label="RV Pen. Rates", linewidth=2)
# ax.set_title("CO Emissions Across Intersections", fontsize=fs)
# ax.set_ylabel("CO Emissions (mg/s)", fontsize=fs)

# y = [0.4507, 0.4313, 0.3560, 0.3185, 0.3080, 0.2739, 0.2619, 0.2891, 0.2523, 0.2462]
# ax.hlines(0.4212, 10, 100, color="black", label="HVs w/ TS", linestyles="--", linewidth=2)
# ax.plot(x, y, color="darkorchid", label="RV Pen. Rates", linewidth=2)
# ax.set_title("HC Emissions Across Intersections", fontsize=fs)
# ax.set_ylabel("HC Emissions (mg/s)", fontsize=fs)

# y = [0.7464, 0.7313, 0.6364, 0.5970, 0.5919, 0.5534, 0.5391, 0.5779, 0.5494, 0.5442]
# ax.hlines(0.7535, 10, 100, color="black", label="HVs w/ TS", linestyles="--", linewidth=2)
# ax.plot(x, y, color="darkorange", label="RV Pen. Rates", linewidth=2)
# ax.set_title("NOx Emissions Across Intersections", fontsize=fs)
# ax.set_ylabel("NOx Emissions (mg/s)", fontsize=fs)

ax.set_xlabel("RV Penetration Rates", fontsize=fs)
ax.set_xlim(10, 100)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)

ax.legend(fontsize=fs-4, loc="lower left")
plt.savefig('./imgs/intersection_graphs/fuel_consumption.png', bbox_inches="tight")