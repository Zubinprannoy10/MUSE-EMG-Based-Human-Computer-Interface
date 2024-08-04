import subprocess

def run_python_file(file_number):
    # Dictionary mapping numbers to file names
    files = {
        '1': 'record_emg.py',
        '2': 'predict_word.py',
        '3': 'search_engine.py'
        # Add more numbers and corresponding file names as needed
    }

    # Get the file name corresponding to the entered number
    file_name = files.get(file_number)
    
    if file_name:
        try:
            # Run the selected Python file using subprocess
            subprocess.run(['python', file_name], check=True)
        except subprocess.CalledProcessError:
            print(f"Error: Failed to run {file_name}")
    else:
        print("Invalid input. Please enter a number corresponding to a file.")

def main():
    while True:
        print("Enter the number corresponding to the file you want to run:")
        print("1.Record silent speech")
        print("2.Predict the word")
        print("3.Search engine")
        # Add more options as needed
        
        choice = input("Enter your choice (or 'q' to quit): ")

        if choice.lower() == 'q':
            break
        else:
            run_python_file(choice)

if __name__ == "__main__":
    main()

