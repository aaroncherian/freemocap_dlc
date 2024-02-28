import logging
import multiprocessing
from pathlib import Path
from typing import Union

import numpy as np

from anipose_utils.anipose_object_loader import load_anipose_calibration_toml_from_path

logger = logging.getLogger(__name__)


def reconstruct_3d(freemocap_data_2d: np.ndarray, calibration_toml_path: Union[str, Path]):
    anipose_calibration_object = load_anipose_calibration_toml_from_path(calibration_toml_path)

    freemocap_data_3d, reprojection_error_data3d, not_sure_what_this_repro_error_is_for = triangulate_3d_data(
        anipose_calibration_object=anipose_calibration_object,
        mediapipe_2d_data=freemocap_data_2d,
        use_triangulate_ransac=False,
        kill_event=None,
    )

    return freemocap_data_3d


def triangulate_3d_data(
        # this triangulate function is taken directly from FreeMoCap, hence why 'mediapipe' is thrown around a lot
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


if __name__ == '__main__':
    from compile_dlc_csv_to_2d_data import compile_dlc_csvs
    from visualization.scatter_plot_of_3d_data import MainWindow
    from PyQt6.QtWidgets import QApplication

    path_to_recording_folder = Path(r'D:\sfn\michael_wobble\recording_13_56_45_gmt-5')
    calibration_toml_path = Path(
        r"D:\sfn\michael_wobble\recording_13_56_45_gmt-5\recording_13_56_45_gmt-5_camera_calibration.toml")

    # path_to_recording_folder = Path(r'D:\sfn\michael_wobble\recording_12_07_09_gmt-5__MDN_wobble_3')
    # calibration_toml_path = Path(r"D:\sfn\michael_wobble\recording_12_07_09_gmt-5__MDN_wobble_3\recording_12_07_09_gmt-5__MDN_wobble_3_camera_calibration.toml")

    dlc_2d_array = compile_dlc_csvs(path_to_recording_folder)

    dlc_3d_array = reconstruct_3d(dlc_2d_array, calibration_toml_path)

    # quick scatter plot of the 3d data

    path_to_3d_data = path_to_recording_folder / 'output_data' / 'raw_data' / 'mediapipe3dData_numFrames_numTrackedPoints_spatialXYZ.npy'
    mediapipe_3d_data = np.load(path_to_3d_data)

    freemocap_3d_data = np.concatenate((mediapipe_3d_data, dlc_3d_array), axis=1)

    app = QApplication([])
    win = MainWindow(mediapipe_data_3d=mediapipe_3d_data, dlc_data_3d=dlc_3d_array)
    win.show()
    app.exec()
    f = 2
