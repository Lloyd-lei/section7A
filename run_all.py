import os
import subprocess
import time
import shutil

# Create output directories
os.makedirs('output/Task1/bayesian', exist_ok=True)
os.makedirs('output/Task1/stirling', exist_ok=True)
os.makedirs('output/Task1/bootstrapping', exist_ok=True)
os.makedirs('output/Task2/vacuum', exist_ok=True)
os.makedirs('output/Task2/cavity', exist_ok=True)

# Function to run a script and print its output
def run_script(script_path, description, working_dir=None):
    print(f"\n{'=' * 80}")
    print(f"Running {description}...")
    print(f"{'=' * 80}")
    
    start_time = time.time()
    
    # Set the working directory if provided
    cwd = os.getcwd() if working_dir is None else os.path.join(os.getcwd(), working_dir)
    
    process = subprocess.Popen(['python', os.path.basename(script_path)], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE,
                              universal_newlines=True,
                              cwd=cwd)
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end='')
    
    # Wait for the process to complete
    process.wait()
    
    # Print any errors
    for line in process.stderr:
        print(line, end='')
    
    end_time = time.time()
    print(f"\nCompleted in {end_time - start_time:.2f} seconds")
    
    return process.returncode

# Copy dataset files to the appropriate directories
print("Copying dataset files to task directories...")
for task_dir in ['Task1/bayesian', 'Task1/bootstrapping', 'Task2/vacuum', 'Task2/cavity']:
    if os.path.exists('dataset_3.json'):
        shutil.copy('dataset_3.json', task_dir)
    if os.path.exists('Vacuum_decay_dataset.json'):
        shutil.copy('Vacuum_decay_dataset.json', task_dir)
    if os.path.exists('Cavity_decay_dataset.json'):
        shutil.copy('Cavity_decay_dataset.json', task_dir)

# List of scripts to run with their working directories
scripts = [
    ('bayesian_inference.py', 'Task 1a: Bayesian Inference', 'Task1/bayesian'),
    ('stirling_approximation.py', 'Task 1b: Stirling\'s Approximation', 'Task1/stirling'),
    ('bootstrapping.py', 'Task 1c: Bootstrapping', 'Task1/bootstrapping'),
    ('vacuum_decay_analysis.py', 'Task 2a: Vacuum Decay Analysis', 'Task2/vacuum'),
    ('cavity_decay_analysis.py', 'Task 2b: Cavity Decay Analysis', 'Task2/cavity')
]

# Run each script
for script_path, description, working_dir in scripts:
    return_code = run_script(script_path, description, working_dir)
    if return_code != 0:
        print(f"Error running {script_path}. Return code: {return_code}")

print("\n" + "=" * 80)
print("All tasks completed!")
print("=" * 80)
print("\nResults are saved in the following directories:")
print("- Task 1a (Bayesian Inference): output/Task1/bayesian/")
print("- Task 1b (Stirling's Approximation): output/Task1/stirling/")
print("- Task 1c (Bootstrapping): output/Task1/bootstrapping/")
print("- Task 2a (Vacuum Decay Analysis): output/Task2/vacuum/")
print("- Task 2b (Cavity Decay Analysis): output/Task2/cavity/") 