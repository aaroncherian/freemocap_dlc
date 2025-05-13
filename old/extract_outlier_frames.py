import deeplabcut
from pathlib import Path

config_path = r"D:\sfn\michael_wobble\wobble_board_project_4-Aaron-2024-02-19\config.yaml"
path_to_recording = Path(r'D:\sfn\michael_wobble\recording_13_56_45_gmt-5')


path_to_videos = path_to_recording/'dlc_data'/'videos_to_analyze'
path_to_dlc_data = path_to_recording/'dlc_data'


deeplabcut.extract_outlier_frames(config=config_path, videos=[str(path_to_videos)], destfolder=str(path_to_dlc_data), automatic=True)

# deeplabcut.refine_labels(config_path)