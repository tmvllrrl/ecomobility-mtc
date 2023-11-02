import json
import csv

junc_id = "334"

input_file_path = "eval_results/onlyHVs_TL/eval_trajectory.json"
output_file_path = f"eval_results/onlyHVs_TL/{junc_id}_speed_profiles.csv"

with open(input_file_path, "r") as file1:
    data = json.load(file1)

    for key in data[junc_id]: # run for the number of vehicles essentially
        x = data[junc_id][key]['time']
        y = data[junc_id][key]['speed']

        all_speeds = []

        for t in range(500, 1000):
            if t in x:
                index = x.index(t)
                
                all_speeds.extend([y[index]])
            else:
                all_speeds.extend([-1])

        with open(output_file_path, "a") as file2:
            writer = csv.writer(file2)
            writer.writerow(all_speeds)

# Function to calculate the average of a column while excluding -1 values
def calculate_average(column):
    column_values = [float(val) for val in column if val != '-1']
    if column_values:
        return sum(column_values) / len(column_values)
    else:
        return -1  # Special value to indicate no valid data

# Create a list to hold the data from the input file
data = []

with open(output_file_path, 'r') as csv_file:
    reader = csv.reader(csv_file)
    
    for row in reader:
        data.append(row)

# Calculate the averages for each column and store them in a list
averages = [calculate_average([row[i] for row in data]) for i in range(len(data[0]))]

# Append the calculated averages as a final row to the data
data.append(averages)

# Write the updated data (including averages) to the output CSV file
with open(output_file_path, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    
    for row in data:
        writer.writerow(row)

print("Averages have been calculated and written to the output file.")