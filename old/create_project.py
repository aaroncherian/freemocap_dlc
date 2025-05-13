import deeplabcut
from typing import List
from models.project_config import ProjectConfig
from pathlib import Path
import yaml

import yaml


def create_dlc_project(config: ProjectConfig) -> str:
    # Validate directories
    if not config.video_directory.exists():
        raise ValueError(f"Video directory {config.video_directory} does not exist.")
    if not config.working_directory.exists():
        config.working_directory.mkdir(parents=True, exist_ok=True)

    # Get a list of video paths from the specified directory
    video_paths: List[Path] = list(config.video_directory.glob('*.mp4'))  # Adjust the glob pattern for different video formats

    # Ensure there are videos to process
    if not video_paths:
        raise ValueError("No video files found in the specified directory.")

    # Create a new DeepLabCut project
    config_path = deeplabcut.create_new_project(config.project_name, config.experimenter_name, [str(path) for path in video_paths], working_directory=str(config.working_directory), copy_videos=True)


    return config_path

if __name__ == "__main__":
    # Define the project configuration

    video_directory_path = Path(r'D:\sfn\michael_wobble\recording_12_07_09_gmt-5__MDN_wobble_3\synchronized_videos')
    working_directory_path = Path(r'D:\sfn\michael_wobble')

    config = ProjectConfig(
        project_name="wobble_board_project_4",
        experimenter_name="Aaron",
        video_directory= video_directory_path,
        working_directory= working_directory_path,
        copy_videos_to_dlc_folder=False,
    )

    # Create the DeepLabCut project
    create_dlc_project(config)