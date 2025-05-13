import deeplabcut
from pathlib import Path
import shutil

config_path = r"D:\sfn\michael_wobble\wobble_board_project_4-Aaron-2024-02-19\config.yaml"




def copy_and_rename_videos(source_folder:Path, destination_folder:Path, identifier:str):
    # Ensure destination exists
    copied_videos = []

    for video in source_folder.glob('*.mp4'):
        new_name = f"{identifier}_{video.name}"
        dest_path = destination_folder / new_name
        shutil.copy(video, dest_path)
        copied_videos.append(dest_path)

    return copied_videos

def process_recording_folder(path_to_recording_folder):
    path_to_recording_folder = Path(path_to_recording_folder)
    path_to_video_folder = path_to_recording_folder / 'synchronized_videos'
    
    dlc_folder = path_to_recording_folder / 'dlc_data'
    dlc_folder.mkdir(parents=True, exist_ok=True)
    
    videos_to_analyze_folder = dlc_folder / 'videos_to_analyze'
    videos_to_analyze_folder.mkdir(parents=True, exist_ok=True)

    # Generate a unique identifier for the folder based on its path or another characteristic
    video_identifier = path_to_recording_folder.stem  # Using folder name as identifier

    # Copy and rename videos to the 'videos_to_analyze' subfolder with unique identifier
    copy_and_rename_videos(source_folder=path_to_video_folder, destination_folder=videos_to_analyze_folder, identifier=video_identifier)

    deeplabcut.analyze_videos(config=config_path, videos= str(videos_to_analyze_folder), save_as_csv=True, destfolder=str(dlc_folder))

    video_list = sorted(list(Path(videos_to_analyze_folder).glob('*.mp4')))
    for video in video_list:
        deeplabcut.filterpredictions(config=config_path, video=str(video), destfolder=str(dlc_folder), save_as_csv=True)

    deeplabcut.create_labeled_video(config=config_path, videos= str(videos_to_analyze_folder), destfolder=str(dlc_folder))






def process_session_folder(session_folder):
    # Convert the session folder path to a pathlib Path object
    session_folder_path = Path(session_folder)

    # Loop through all subdirectories (recording folders) in the session folder
    for recording_folder in session_folder_path.iterdir():
        if recording_folder.is_dir():
            # Check if the 'dlc_data' folder exists in the recording folder
            if not (recording_folder / 'dlc_data').exists():
                process_recording_folder(recording_folder)

if __name__ == '__main__':
    # process_session_folder(r'D:\2023-06-07_JH\1.0_recordings\treadmill_calib')
    process_recording_folder(r'D:\sfn\michael_wobble\recording_13_56_45_gmt-5')