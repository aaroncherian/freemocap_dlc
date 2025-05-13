
from pathlib import Path
import logging
import numpy as np
from typing import List, Optional, Union

from reconstruction_to_3d.reconstruction import reconstruct_3d
from reconstruction_to_3d.visualization import plot_3d_scatter
from reconstruction_to_3d.postprocessing import process_and_filter_data
from reconstruction_to_3d.compile_dlc_csv_to_2d_data import compile_dlc_csvs
from skellymodels.experimental.model_redo.managers.human import Human
from skellymodels.experimental.model_redo.tracker_info.model_info import ModelInfo


# Configure logger
logger = logging.getLogger(__name__)


def process_recording_session(
    path_to_recording_folder: Union[str, Path],
    path_to_calibration_toml: Optional[Path] = None,
    path_to_folder_of_dlc_csvs: Optional[Path] = None,
    use_skellyforge: bool = True,
    filter_order: int = 4,
    cutoff_frequency: float = 6.0,
    sampling_rate: float = 90.0,
    dlc_confidence_threshold: float = 0.6,
    landmark_names: Optional[List[str]] = None,
    create_visualization: bool = True,
) -> np.ndarray:
    """
    Process a recording session from 2D DLC data to 3D reconstruction with optional filtering.
    
    Args:
        path_to_recording_folder: Path to the recording folder
        path_to_calibration_toml: Path to the calibration TOML file (if None, will search in recording folder)
        path_to_folder_of_dlc_csvs: Path to the folder containing DLC CSV files (if None, will use 'dlc_data' subfolder)
        use_skellyforge: Whether to use SkellyForge for filtering the data
        filter_order: Order of the Butterworth filter
        cutoff_frequency: Cutoff frequency for the Butterworth filter
        sampling_rate: Sampling rate of the data in Hz
        confidence_threshold: Confidence threshold for DLC data
        landmark_names: Names of the landmarks (if None, will use default landmarks)
        create_visualization: Whether to create a 3D visualization
        
    Returns:
        Processed 3D data array
    """

    # Convert path to Path object if it's a string
    path_to_recording_folder = Path(path_to_recording_folder)
    
    # Set default paths if not provided
    if path_to_calibration_toml is None:
        path_to_calibration_toml = list(path_to_recording_folder.glob('*calibration.toml'))[0]
    
    if path_to_folder_of_dlc_csvs is None:
        path_to_folder_of_dlc_csvs = path_to_recording_folder / 'dlc_data'
    
    # Set default landmark names if not provided
    if landmark_names is None:
        landmark_names = [
            'nose',
            'right_eye',
            'right_ear',
            'left_eye',
            'left_ear',
            'toy'
        ]

    # Output directory setup
    path_to_output_folder = path_to_recording_folder / 'output_data'
    path_to_output_folder.mkdir(parents=True, exist_ok=True)

    path_to_raw_data_folder = path_to_output_folder / 'raw_data'
    path_to_raw_data_folder.mkdir(parents=True, exist_ok=True)  

    # Load and process 2D data
    logger.info("Compiling DLC CSV files")
    dlc_2d_array = compile_dlc_csvs(
        path_to_folder_of_dlc_csvs,
        confidence_threshold=dlc_confidence_threshold
    )
    
    # Reconstruct 3D data
    logger.info("Reconstructing 3D data")
    dlc_3d_array = reconstruct_3d(dlc_2d_array, path_to_calibration_toml)
    
    # Save raw 3D data
    logger.info("Saving raw 3D data")
    np.save(
        path_to_raw_data_folder / 'dlc_3dData_numFrames_numTrackedPoints_spatialXYZ.npy', 
        dlc_3d_array
    )

    # Apply filtering if requested
    if use_skellyforge:
        logger.info("Applying SkellyForge filters to 3D data")
        dlc_3d_array = process_and_filter_data(
            dlc_3d_array,
            landmark_names,
            cutoff_frequency,
            sampling_rate,
            filter_order
        )

    logger.info("Processing with skeleton model")
    path_to_ferret_yaml = Path(__file__).parents[0] / 'reconstruction_to_3d' / 'tracker_info' / 'dlc_ferret.yaml'
    ferret_model_info = ModelInfo(config_path=path_to_ferret_yaml)
    
    skeleton = Human.from_landmarks_numpy_array( #not a human but it still does the job
        name="ferret",
        model_info=ferret_model_info,
        landmarks_numpy_array=dlc_3d_array
    )
    skeleton.calculate()
    skeleton.save_out_numpy_data(path_to_output_folder=path_to_output_folder)

    if create_visualization:
        logger.info("Creating 3D visualization")
        data_dict = {'dlc_data': dlc_3d_array}
        plot_3d_scatter(data_dict)
    
    return dlc_3d_array


def main():

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    process_recording_session(
        path_to_recording_folder=r'D:\ferret_em_talk\ferret_04_28_25',
        use_skellyforge=False,
        filter_order=4,
        cutoff_frequency=6.0,
        sampling_rate=90.0,
        dlc_confidence_threshold=0.6,
        create_visualization=True
    )


if __name__ == '__main__':
    main()