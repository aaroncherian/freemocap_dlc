from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProjectConfig:
    project_name: str
    experimenter_name: str
    video_directory: Path
    working_directory: Path
    copy_videos_to_dlc_folder: bool
