import csv

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid", {"grid.linestyle": "--"})

junc_id = "334"

def moving_average(data, window_size=10):
    smoothed_data = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i + window_size]
        average = sum(window) / window_size
        smoothed_data.append(average)
    return smoothed_data

# Initialize a graph
fig, ax = plt.subplots(figsize=(8,8), dpi=300)
x = list(range(500, 991))

input_file_path1 = f"eval_results/onlyHVs_TL/{junc_id}_speed_profiles.csv"
input_file_path2 = f"eval_results/results_RV1.0/{junc_id}_speed_profiles.csv"
input_file_path3 = f"eval_results/results_RV0.2/{junc_id}_speed_profiles.csv"
input_file_path4 = f"eval_results/results_RV0.4/{junc_id}_speed_profiles.csv"
input_file_path5 = f"eval_results/results_RV0.6/{junc_id}_speed_profiles.csv"
input_file_path6 = f"eval_results/results_RV0.8/{junc_id}_speed_profiles.csv"
input_file_path7 = f"eval_results/results_RV0.1/{junc_id}_speed_profiles.csv"


last_row = None
with open(input_file_path1, "r", newline='') as file1:
    csv_reader = csv.reader(file1)
    
    # Iterate through the rows
    for row in csv_reader:
        last_row = row

last_row = [float(value) for value in last_row]
y1 = last_row
y1 = moving_average(y1)

ax.plot(x, y1, label="HVs w/ TS", color="black", linewidth=2)


# last_row = None
# with open(input_file_path3, "r", newline='') as file3:
#     csv_reader = csv.reader(file3)

#     for row in csv_reader:
#         last_row = row

# last_row = [float(value) for value in last_row]
# y3 = last_row
# y3 = moving_average(y3)

# ax.plot(x, y3, label="20% RVs", color="royalblue", linewidth=2)

# last_row = None
# with open(input_file_path4, "r", newline='') as file4:
#     csv_reader = csv.reader(file4)

#     for row in csv_reader:
#         last_row = row

# last_row = [float(value) for value in last_row]
# y4 = last_row
# y4 = moving_average(y4)

# ax.plot(x, y4, label="40% RVs", color="orange", linewidth=2)

# last_row = None
# with open(input_file_path5, "r", newline='') as file5:
#     csv_reader = csv.reader(file5)

#     for row in csv_reader:
#         last_row = row

# last_row = [float(value) for value in last_row]
# y5 = last_row
# y5 = moving_average(y5)

# ax.plot(x, y5, label="60% RVs", color="lightcoral", linewidth=2)

# last_row = None
# with open(input_file_path6, "r", newline='') as file6:
#     csv_reader = csv.reader(file6)

#     for row in csv_reader:
#         last_row = row

# last_row = [float(value) for value in last_row]
# y6 = last_row
# y6 = moving_average(y6)

# ax.plot(x, y6, label="80% RVs", color="teal")

last_row = None
with open(input_file_path2, "r", newline='') as file2:
    csv_reader = csv.reader(file2)

    for row in csv_reader:
        last_row = row

last_row = [float(value) for value in last_row]
y2 = last_row
y2 = moving_average(y2)

ax.plot(x, y2, label="100% RVs", color="darkorchid", linewidth=2)

# last_row = None
# with open(input_file_path7, "r", newline='') as file7:
#     csv_reader = csv.reader(file7)

#     for row in csv_reader:
#         last_row = row

# last_row = [float(value) for value in last_row]
# y7 = last_row
# y7 = moving_average(y7)

# ax.plot(x, y7, label="10% RVs", color="forestgreen", linewidth=2)

fs=24
ax.set_title("Velocity Profile", fontsize=fs)
ax.set_ylabel("Velocity (m/s)", fontsize=fs)
ax.set_xlabel("Time (s)", fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)

ax.legend(fontsize=fs-8)
plt.savefig(f"{junc_id}_speed_profile_100.png", bbox_inches="tight")


