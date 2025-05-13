from pathlib import Path
import numpy as np
import logging
import multiprocessing
from typing import Union

from anipose_utils.anipose_object_loader import load_anipose_calibration_toml_from_path
import plotly.graph_objects as go

def plot_3d_scatter(data_3d_dict: dict):
    # Determine axis limits based on the data
    all_data = np.concatenate(list(data_3d_dict.values()), axis=1)
    
    mean_x = np.nanmean(all_data[:, :, 0])
    mean_y = np.nanmean(all_data[:, :, 1])
    mean_z = np.nanmean(all_data[:, :, 2])

    ax_range = 2000


    # Create a Plotly figure
    fig = go.Figure()

    # Generate a frame for each time step
    frames = []
    for frame in range(all_data.shape[0]):
        frame_data = []
        for label, data in data_3d_dict.items():
            frame_data.append(go.Scatter3d(
                x=data[frame, :, 0],
                y=data[frame, :, 1],
                z=data[frame, :, 2],
                mode='markers',
                name=label,
                marker=dict(size=4, opacity=0.8)
            ))
        frames.append(go.Frame(data=frame_data, name=str(frame)))

    # Add the first frame's data
    for label, data in data_3d_dict.items():
        fig.add_trace(go.Scatter3d(
            x=data[0, :, 0],
            y=data[0, :, 1],
            z=data[0, :, 2],
            mode='markers',
            name=label,
            marker=dict(size=4, opacity=0.8),
            opacity=.5
        ))

    # Update the layout with sliders and other settings
    fig.update_layout(
        scene=dict(
            aspectmode='cube',
            xaxis=dict(range=[mean_x - ax_range, mean_x + ax_range], title='X'),
            yaxis=dict(range=[mean_y - ax_range, mean_y + ax_range], title='Y'),
            zaxis=dict(range=[mean_z - ax_range, mean_z + ax_range], title='Z')
        ),
        title="3D Scatter Plot",
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}]),
                     dict(label="Pause",
                          method="animate",
                          args=[[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}])]
        )],
        sliders=[{
            "steps": [{"args": [[str(frame)], {"frame": {"duration": 100, "redraw": True}, "mode": "immediate"}],
                       "label": str(frame), "method": "animate"} for frame in range(all_data.shape[0])],
            "currentvalue": {"prefix": "Frame: "}
        }]
    )

    # Add the frames to the figure
    fig.frames = frames

    # Show the plot
    fig.show()


logger = logging.getLogger(__name__)

def reconstruct_3d(freemocap_data_2d: np.ndarray, calibration_toml_path:  Union[str, Path]):

    anipose_calibration_object = load_anipose_calibration_toml_from_path(calibration_toml_path)

    freemocap_data_3d, reprojection_error_data3d, not_sure_what_this_repro_error_is_for = triangulate_3d_data(
        anipose_calibration_object=anipose_calibration_object,
        mediapipe_2d_data=freemocap_data_2d,
        use_triangulate_ransac=False,
        kill_event=None,
    )

    return freemocap_data_3d



def triangulate_3d_data(
    #this triangulate function is taken directly from FreeMoCap, hence why 'mediapipe' is thrown around a lot 
    anipose_calibration_object,
    mediapipe_2d_data: np.ndarray,
    use_triangulate_ransac: bool = False,
    kill_event: multiprocessing.Event = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    number_of_cameras = mediapipe_2d_data.shape[0]
    number_of_frames = mediapipe_2d_data.shape[1]
    number_of_tracked_points = mediapipe_2d_data.shape[2]
    number_of_spatial_dimensions = mediapipe_2d_data.shape[3]

    if not number_of_spatial_dimensions == 2:
        logger.error(
            f"This is supposed to be 2D data but, number_of_spatial_dimensions: {number_of_spatial_dimensions}"
        )
        raise ValueError

    data2d_flat = mediapipe_2d_data.reshape(number_of_cameras, -1, 2)

    logger.info(
        f"Reconstructing 3d points from 2d points with shape: \n"
        f"number_of_cameras: {number_of_cameras},\n"
        f"number_of_frames: {number_of_frames}, \n"
        f"number_of_tracked_points: {number_of_tracked_points},\n"
        f"number_of_spatial_dimensions: {number_of_spatial_dimensions}"
    )

    if use_triangulate_ransac:
        logger.info("Using `triangulate_ransac` method")
        data3d_flat = anipose_calibration_object.triangulate_ransac(data2d_flat, progress=True, kill_event=kill_event)
    else:
        logger.info("Using simple `triangulate` method ")
        data3d_flat = anipose_calibration_object.triangulate(data2d_flat, progress=True, kill_event=kill_event)

    spatial_data3d_numFrames_numTrackedPoints_XYZ = data3d_flat.reshape(number_of_frames, number_of_tracked_points, 3)

    data3d_reprojectionError_flat = anipose_calibration_object.reprojection_error(data3d_flat, data2d_flat, mean=True)
    data3d_reprojectionError_full = anipose_calibration_object.reprojection_error(data3d_flat, data2d_flat, mean=False)
    reprojectionError_cam_frame_marker = np.linalg.norm(data3d_reprojectionError_full, axis=2).reshape(
        number_of_cameras, number_of_frames, number_of_tracked_points
    )

    reprojection_error_data3d_numFrames_numTrackedPoints = data3d_reprojectionError_flat.reshape(
        number_of_frames, number_of_tracked_points
    )

    return (
        spatial_data3d_numFrames_numTrackedPoints_XYZ,
        reprojection_error_data3d_numFrames_numTrackedPoints,
        reprojectionError_cam_frame_marker,
    )

class ResultClass:
    def __init__(self, result=None):
        self.result = result
    
def handle_thread_finished(results, result_class:ResultClass):
    result_class.result = results[TASK_FILTERING]['result']

if __name__ == '__main__':
    from compile_dlc_csv_to_2d_data import compile_dlc_csvs
    from skellyforge.freemocap_utils.postprocessing_widgets.task_worker_thread import TaskWorkerThread
    from skellyforge.freemocap_utils.config import default_settings
    from skellyforge.freemocap_utils.constants import (
    TASK_FILTERING,
    PARAM_CUTOFF_FREQUENCY,
    PARAM_SAMPLING_RATE,
    PARAM_ORDER,
    PARAM_ROTATE_DATA,
    TASK_SKELETON_ROTATION,
    TASK_INTERPOLATION,
)
    from skellymodels.experimental.model_redo.managers.human import Human
    from skellymodels.experimental.model_redo.tracker_info.model_info import ModelInfo

    path_to_recording_folder = Path(r'D:\ferret_em_talk\ferret_04_28_25')
    path_to_calibration_toml = list(path_to_recording_folder.glob('*calibration.toml'))[0]
    path_to_folder_of_dlc_csvs = path_to_recording_folder / 'dlc_data'
    use_skellyforge = True

    #BW filter settings
    order = 4
    cutoff_frequency = 6.0
    sampling_rate = 90.0

    #landmark names
    landmark_names = [
        'nose',
        'right_eye',
        'right_ear',
        'left_eye',
        'left_ear',
        'toy']


    path_to_ouput_folder = path_to_recording_folder / 'output_data'
    path_to_ouput_folder.mkdir(parents=True, exist_ok=True)

    path_to_raw_data_folder = path_to_ouput_folder / 'raw_data'
    path_to_raw_data_folder.mkdir(parents=True, exist_ok=True)  

    dlc_2d_array = compile_dlc_csvs(path_to_folder_of_dlc_csvs,
                                    confidence_threshold=.6)
    
    dlc_3d_array = reconstruct_3d(dlc_2d_array, path_to_calibration_toml)
    np.save(path_to_raw_data_folder/'dlc_3dData_numFrames_numTrackedPoints_spatialXYZ.npy', dlc_3d_array)


    if use_skellyforge:
        adjusted_settings = default_settings.copy()
        adjusted_settings[TASK_FILTERING][PARAM_CUTOFF_FREQUENCY] = cutoff_frequency
        adjusted_settings[TASK_FILTERING][PARAM_SAMPLING_RATE] = sampling_rate
        adjusted_settings[TASK_FILTERING][PARAM_ORDER] = order
        adjusted_settings[TASK_SKELETON_ROTATION][PARAM_ROTATE_DATA] = False
        
        task_list = [TASK_INTERPOLATION, TASK_FILTERING]
        result_handler = ResultClass()
        worker_thread = TaskWorkerThread(
            raw_skeleton_data= dlc_3d_array,
            task_list=task_list,
            landmark_names=landmark_names,
            settings=adjusted_settings,
            all_tasks_finished_callback=lambda results: handle_thread_finished(results, result_handler),
        )

        worker_thread.start()
        worker_thread.join()

        dlc_3d_array = result_handler.result
    

    path_to_ferret_yaml = Path(__file__).parents[0]/'tracker_info'/'dlc_ferret.yaml'
    ferret_model_info = ModelInfo(config_path=path_to_ferret_yaml)
    skeleton = Human.from_landmarks_numpy_array(name="ferret",
                model_info=ferret_model_info,
                landmarks_numpy_array=dlc_3d_array)
    skeleton.calculate()

    skeleton.save_out_numpy_data(path_to_output_folder=path_to_ouput_folder)

    data_dict = {'dlc_data': dlc_3d_array}
    plot_3d_scatter(data_dict)


