import deeplabcut
from pathlib import Path
from typing import List

def add_videos_to_dlc_project(config_path: Path, video_directory: Path, video_extensions: List[str] = ['.mp4']) -> None:
    # Find new video files in the specified directory
    new_videos = [str(file) for file in video_directory.glob('*') if file.suffix in video_extensions]

    # Check if there are new videos to add
    if not new_videos:
        print("No new videos found in the specified directory.")
        return

    # Add new videos to the DeepLabCut project
    deeplabcut.add_new_videos(config=config_path, videos=new_videos, copy_videos=False)

    print(f"Added {len(new_videos)} new videos to the project.")


if __name__ == "__main__":

    config_path = Path(r"D:\sfn\michael_wobble\wobble_board_project_4-Aaron-2024-02-19\config.yaml")
    video_directory = Path(r"D:\sfn\michael_wobble\recording_16_53_56_gmt-5\synchronized_videos")

    add_videos_to_dlc_project(config_path, video_directory)