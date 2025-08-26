import pandas as pd
import os

def preprocess_data(file_path):
    """
    Loads a CSV file, selects the middle 10 seconds of data, and
    downsamples it from 200 Hz to 50 Hz by taking every 4th sample,
    then adds a new time column based on the 50Hz data.
    """
    try:
        df = pd.read_csv(file_path)
        
        # The data is sampled at 200 Hz
        #trimming the first and last 5 seconds
        start_sample = 5 * 200
        end_sample = 15 * 200
        middle_10_sec_df = df.iloc[start_sample:end_sample].copy()
        
        # Downsample from 200 Hz to 50 Hz by taking every 4th sample (200/50 = 4)
        processed_df = middle_10_sec_df.iloc[::4].reset_index(drop=True)
        # so as instructed in the assignment , we have 50hz sampled data of 10 seconds = 500 samples.
        print(f"File: {os.path.basename(file_path)} - Samples before decimation: {len(middle_10_sec_df)}, Samples after decimation: {len(processed_df)}")

        # time column based on the new 50 Hz sampling rate
        processed_df['time'] = [i / 50 for i in range(len(processed_df))]

        return processed_df
        
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def process_all_data(input_folder, output_folder):
    """
    Iterates through a directory, preprocesses all CSV files, and
    saves them to a new output directory
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: {output_folder}")


    for root, dirs, files in os.walk(input_folder):
        # Create the corresponding subdirectory in the output folder
        relative_path = os.path.relpath(root, input_folder)
        output_dir = os.path.join(output_folder, relative_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Process each CSV file in the current directory
        for file in files:
            if file.endswith('.csv'):
                input_file_path = os.path.join(root, file)
                output_file_path = os.path.join(output_dir, f'processed_{file}')
                
                print(f"Processing {input_file_path}...")
                processed_df = preprocess_data(input_file_path)

                if processed_df is not None:
                    # Save the processed DataFrame to the new location
                    processed_df.to_csv(output_file_path, index=False)
                    print(f"Saved processed data to {output_file_path}")

main_folder = 'My_Combined'
output_folder = 'my_combined_processed'

if os.path.exists(main_folder):
    process_all_data(main_folder, output_folder)
    print("All files processed successfully!")
else:
    print(f"The folder '{main_folder}' was not found. Please make sure the script is in the same directory as your '{main_folder}' folder.")