import json

# Read dataset_3.json
with open('dataset_3.json', 'r') as f:
    dataset_3 = json.load(f)
    
print("Dataset 3 (first 10 elements):", dataset_3[:10])
print("Dataset 3 length:", len(dataset_3))

# Read a small portion of Vacuum_decay_dataset.json
with open('Vacuum_decay_dataset.json', 'r') as f:
    # Read only the first 1000 characters to get an idea of the structure
    vacuum_data_sample = f.read(1000)
    
print("\nVacuum decay dataset (sample):")
print(vacuum_data_sample) 