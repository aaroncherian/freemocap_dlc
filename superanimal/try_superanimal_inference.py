import deeplabcut
import os

filepath = r"D:\deeplabcut-models\pooh-the-cat\pooh-superanimal-inference\VID_20240227_143846.mp4"
video_path = os.path.abspath(filepath)
video_name = os.path.splitext(video_path)[0]

videotype = os.path.splitext(video_path)[1]
scale_list = []

super_animal_name = "superanimal_quadruped"
pcutoff = .3
print(f"Running superanimal inference on {video_path}...")
deeplabcut.video_inference_superanimal(
    [video_path],
    super_animal_name,
    videotype=videotype,
    video_adapt=True,
    scale_list=scale_list,
    pcutoff=pcutoff,
)

print(f"Finished running superanimal inference on {video_path}.")