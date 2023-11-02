import json

junc_id = "334"
input_file_path = "eval_results/results_RV1.0/eval_trajectory.json"

with open(input_file_path, "r") as file1:
    data = json.load(file1)

    counts = dict()

    for key in data[junc_id]:
        x = data[junc_id][key]['time']

        # Checking the count of each profile length
        length_x = len(x)
        if length_x not in counts.keys():
            counts[length_x] = 0

        counts[length_x] = counts[length_x] + 1
    
    sorted_counts = sorted(counts.items(), key=lambda x: x[1])
    print(sorted_counts)
    print(len(sorted_counts))




