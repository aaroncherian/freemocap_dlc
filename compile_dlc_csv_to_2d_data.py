from pathlib import Path

import pandas as pd


def compile_dlc_csvs(path_to_recording_folder):
    path_to_folder_with_csvs = path_to_recording_folder / 'dlc_data' / 'iteration_2'

    # Filtered csv list
    filtered_csv_list = sorted(list(path_to_folder_with_csvs.glob('*filtered*.csv')))

    # Initialize an empty list to hold dataframes
    dfs = []

    for csv in filtered_csv_list:
        # Read each csv into a dataframe with a multi-index header
        df = pd.read_csv(csv, header=[1, 2])

        # Drop the first column (which just has the headers )
        df = df.iloc[:, 1:]

        # Check if data shape is as expected
        if df.shape[1] % 3 != 0:
            print(f"Unexpected number of columns in {csv}: {df.shape[1]}")
            continue

        try:
            # Convert the df into a 4D numpy array of shape (1, num_frames, num_markers, 3) and append to dfs
            dfs.append(df.values.reshape(1, df.shape[0], df.shape[1] // 3, 3))
        except ValueError as e:
            print(f"Reshape failed for {csv} with shape {df.shape}: {e}")

    # Concatenate all the arrays along the first axis (camera axis)
    dlc_2d_array_with_confidence = np.concatenate(dfs, axis=0)

    confidence_thresholded_dlc_2d_array_XYC = apply_confidence_threshold(array=dlc_2d_array_with_confidence,
                                                                         threshold=0.5)
    # final_thresholded_array = apply_confidence_threshold(final_array, 0.6)

    confidence_thresholded_dlc_2d_array_XY = dlc_2d_array_with_confidence[:, :, :, :2]

    return confidence_thresholded_dlc_2d_array_XY


import numpy as np


def apply_confidence_threshold(array, threshold):
    """
    Set X,Y values to zero where the corresponding confidence value is below threshold.

    Parameters:
    - array: 4D numpy array with shape (num_cameras, num_frames, num_markers, 3)
      The last dimension should have the structure (x, y, confidence).
    - threshold: Confidence threshold. All X,Y values with a confidence below this threshold will be set to zero.
    """
    confidences = array[:, :, :, 2]  # Get the confidence values
    mask = confidences < threshold  # Create a boolean mask where the confidence values are below threshold
    array[mask, 0:2] = np.nan  # Set the X,Y values to NaN where the mask is True
    return array


if __name__ == '__main__':
    path_to_recording_folder = Path(r'D:\sfn\michael_wobble\recording_12_07_09_gmt-5__MDN_wobble_3')
    dlc_2d_array = compile_dlc_csvs(path_to_recording_folder)

    f = 2
