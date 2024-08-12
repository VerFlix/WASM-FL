import csv

def fann_to_csv(fann_file, csv_file):
    with open(fann_file, 'r') as f:
        lines = f.readlines()

    num_samples, num_input, num_output = map(int, lines[0].strip().split())

    with open(csv_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        
        # Write header
        header = [f'input_{i+1}' for i in range(num_input)] + [f'output_{i+1}' for i in range(num_output)]
        csvwriter.writerow(header)

        # Write data
        for i in range(num_samples):
            input_data = lines[1 + i * 2].strip().split()
            output_data = lines[2 + i * 2].strip().split()
            row = input_data + output_data
            csvwriter.writerow(row)

if __name__ == "__main__":
    fann_file = 'mnist_train_fann.data'
    csv_file = 'mnist_train_fann.csv'
    fann_to_csv(fann_file, csv_file)
    print(f"Converted {fann_file} to {csv_file}")