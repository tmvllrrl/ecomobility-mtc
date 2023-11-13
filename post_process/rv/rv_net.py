import csv
import argparse

def process_net(args):

    rv_rate = args.rv_rate

    input_file_paths = f"../../eval_results/results_RV{str(rv_rate)}/avg_results.csv"

    for input_file_path in input_file_paths:

        # Read the input CSV file and calculate the average of each column
        with open(input_file_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)  # Read all rows into a list

            if not rows:
                print("The CSV file is empty.")
            else:
                # Transpose the rows to work with columns
                columns = list(map(list, zip(*rows)))

                # Initialize a list to store the column averages
                column_averages = []

                for column in columns:
                    # Convert column values to floats
                    column_values = [float(value) for value in column]

                    # Calculate the average of the column
                    column_average = sum(column_values) / len(column_values)
                    column_averages.append(column_average)

                # Append the column averages as a new row to the input CSV file
                rows.append(column_averages)

        # Write the updated rows, including the column averages, back to the same CSV file
        with open(input_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(rows)

        print(f"Column Averages: {column_averages}")
        print(f"Appended to {input_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--rv-rate", default=1.0, type=float)
    args = parser.parse_args()

    process_net(args)

    