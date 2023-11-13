import re
import csv
import argparse

def process_emission(args):

    rv_rate = args.rv_rate
    emission = args.emission

    # Input and output file paths
    input_file_path = f"../../eval_results/results_RV{str(rv_rate)}/all_results.txt"
    output_file_path = f"../../eval_results/results_RV{str(rv_rate)}/int_{emission.lower()}_results.csv"

    all_patterns = {
        "CO": [r'OVERALL AVG CO EMISSIONS AT JUNCTION 229: (\d+\.\d+)', r'OVERALL AVG CO EMISSIONS AT JUNCTION 499: (\d+\.\d+)', r'OVERALL AVG CO EMISSIONS AT JUNCTION 332: (\d+\.\d+)', r'OVERALL AVG CO EMISSIONS AT JUNCTION 334: (\d+\.\d+)'],
        "CO2": [r'OVERALL AVG CO2 EMISSIONS AT JUNCTION 229: (\d+\.\d+)', r'OVERALL AVG CO2 EMISSIONS AT JUNCTION 499: (\d+\.\d+)', r'OVERALL AVG CO2 EMISSIONS AT JUNCTION 332: (\d+\.\d+)', r'OVERALL AVG CO2 EMISSIONS AT JUNCTION 334: (\d+\.\d+)'],
        "FUEL": [r'OVERALL AVG FUEL CONSUMPTION AT JUNCTION 229: (\d+\.\d+)', r'OVERALL AVG FUEL CONSUMPTION AT JUNCTION 499: (\d+\.\d+)', r'OVERALL AVG FUEL CONSUMPTION AT JUNCTION 332: (\d+\.\d+)', r'OVERALL AVG FUEL CONSUMPTION AT JUNCTION 334: (\d+\.\d+)'],
        "HC": [r'OVERALL AVG HC EMISSIONS AT JUNCTION 229: (\d+\.\d+)', r'OVERALL AVG HC EMISSIONS AT JUNCTION 499: (\d+\.\d+)', r'OVERALL AVG HC EMISSIONS AT JUNCTION 332: (\d+\.\d+)', r'OVERALL AVG HC EMISSIONS AT JUNCTION 334: (\d+\.\d+)'],
        "NOX": [r'OVERALL AVG NOX EMISSIONS AT JUNCTION 229: (\d+\.\d+)', r'OVERALL AVG NOX EMISSIONS AT JUNCTION 499: (\d+\.\d+)', r'OVERALL AVG NOX EMISSIONS AT JUNCTION 332: (\d+\.\d+)', r'OVERALL AVG NOX EMISSIONS AT JUNCTION 334: (\d+\.\d+)'],
        "PMX": [r'OVERALL AVG PMX EMISSIONS AT JUNCTION 229: (\d+\.\d+)', r'OVERALL AVG PMX EMISSIONS AT JUNCTION 499: (\d+\.\d+)', r'OVERALL AVG PMX EMISSIONS AT JUNCTION 332: (\d+\.\d+)', r'OVERALL AVG PMX EMISSIONS AT JUNCTION 334: (\d+\.\d+)'],
    }

    patterns = all_patterns[emission]
    
    # Regular expression pattern to match lines with floating-point numbers
    for pattern in patterns: 

        # List to store the extracted numbers
        numbers = []

        # Read the input text file and extract the numbers
        with open(input_file_path, 'r') as file:
            for line in file:
                match = re.search(pattern, line)
                if match:
                    number = match.group(1)
                    numbers.append(number)

        # Write the extracted numbers to the output CSV file
        with open(output_file_path, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(numbers)

        print(f"Extracted numbers: {numbers}")
        print(f"Saved to {output_file_path}")


    ############ CALCULATE AVERAGES ##############

    # List to store the averages
    averages = []

    # Read the input CSV file and calculate the average of each line
    with open(output_file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        
        # Read and process each row
        for row in reader:
            # Convert the row values to floats
            row_values = [float(value) for value in row]
            
            # Calculate the average of the row and append it to the averages list
            average = sum(row_values) / len(row_values)
            averages.append(average)

    # Append the averages as a new row to the output CSV file
    with open(output_file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write the averages as a new row in the CSV file
        writer.writerow(averages)

    print(f"Averages: {averages}")
    print(f"Appended to {output_file_path}")

    # Read the input CSV file and calculate the average of the final row
    with open(output_file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)  # Read all rows into a list

        if not rows:
            print("The CSV file is empty.")
        else:
            # Extract the last row and convert its values to floats
            final_row = [float(value) for value in rows[-1]]

            # Calculate the average of the final row
            average = sum(final_row) / len(final_row)

            # Append the average as a new row to the input CSV file
            rows.append([average])

    # Write the updated rows, including the average, back to the same CSV file
    with open(output_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)

    print(f"Average of the final row: {average}")
    print(f"Appended to {output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--rv-rate", type=float, default=1.0)
    parser.add_argument("--emission", default="CO", choices=["CO", "CO2", "FUEL", "HC", "NOX", "PMX"])
    args = parser.parse_args()

    process_emission(args)