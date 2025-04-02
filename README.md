# Some notes before you start:

1) This isn't meant to replace the DeepLabCut instructions - just supplement them and clear up some things that took me
   a long time to figure out (such as installation). It is still necessary to read over the
   DeepLabCut user guide [here](https://deeplabcut.github.io/DeepLabCut/docs/standardDeepLabCut_UserGuide.html). When
   looking at these instructions, I would read the DeepLabCut guide for that section first, and then look at these ones
2) Some of these functions are built under the assumptions that its being used on FreeMoCap data/recordings. One
   function in particular for example, will look for the `synchronized_videos` folder in a FreeMoCap recording folder.
   I've tried to be clear about when this happens.
3) These are all instructions for a Windows machine. I'm not certain if they would work for other systems, particularly
   when it comes to the ins and outs of CUDA/TensorFlow/cudnn compatibility

# Installing DeepLabCut

- Create an python3.10 environment (I believe 3.10 is the latest version that can be handled by their tensorflow
  version, need to double-check)
- **Installation:**: `pip install deeplabcut[tf,gui]`
    - this installs both tensorflow and the deeplabcut GUI
- You could just start using DeepLabCut here, but you'll probably want to check that it is able to detect and use your
  GPU. You can run the script mentioned [in this section](#verifying-that-everything-is-installed-correctly) to see if
  your GPU is recognized by tensorflow. If not, you may need to make sure that the proper versions of CUDA and cudnn
  are installed for your GPU to be recognized and used
    - The DLC docs say that their Anaconda env installation should handle installing cudnn, but that didn't seem like
      the case for me

## Installing CUDA, cudnn, and tensorflow

- CUDA, cudnn and tensorflow all have compatibility requirements with each other, so you have to be particular with what
  versions you install. Below is what worked for me.
- By default, tensorflow 2.10 gets installed with DeepLabCut
    - *note*: on Windows, tensorflow 2.10 is the last/latest version that can be installed. Any higher and you need
      Windows Subsystem for Linux or whatever that is-
- Given the tensorflow 2.10 installation, to match it you need
    - CUDA 11.2
    - cudnn 8.1

- Below are two methods of installation that have both seemed to work for me. In the first, you install CUDA
  system-wide. In the second, you install it directly into the environment. My initial thoughts on which is
  optimal would probably depend on ease-of-use and whether you might have more recent CUDA versions and are worried
  about conflicts

### Option 1: Installing CUDA system-wide

- You can download CUDA 11.2
  from [here](https://developer.nvidia.com/cuda-11.2.0-download-archive?target_os=Windows&target_arch=x86_64)
- You can download cudnn 8.1.1 or 8.1.0 from Nvidia's archive [here](https://developer.nvidia.com/rdp/cudnn-archive)
    - To be able to download it, I had to make a developer's account with Nvidia. I also had to specifically *opt in to
      the developer program* to use it. At the time of writing this I regret to say I don't remember
      how exactly I opted in - I believe you have a chance to do so when you make the account, but I missed that, and so
      I google'd something like 'opt into nvidia developer program' and I found it there
- In the cudnn download folder, there are 3 folders of note, `bin`, `include`, and `lib`
- To properly install cudnn, you need to find your CUDA 11.2 folder and:
    - copy the files from `bin` in cudnn and copy/move them over to the `bin` folder of CUDA
    - copy the files from `include` in cudnn and copy/move them over to the `include` folder of CUDA
    - copy the files from the `lib\x64` folder into the `lib\x64` folder of CUDA (not into the `lib` folder, into
      the `x64` folder)

### Option 2: Installing CUDA and cudnn into your environment

- In your Anaconda environment, you can use condaforge to install CUDA and cudnn
- Run this command in your Anaconda prompt `conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0`
    - *Note:* I got these commands from tensorflow's installation website for
      Windows [here](https://www.tensorflow.org/install/pip#windows-native), take a look for more info

### Verifying that everything is installed correctly

- To verify all is well and that tensorflow recognizes your GPU, you can run the `tensorflow_verification.py` script in
  this repository (located [here](/tensorflow_verification.py))
- If it works successfully, you should get an output telling you how many GPU's are available to use
- If cudnn isn't installed correctly, you may see a `cudnn64_8.dll file not found` error. This is what first clued me
  into the fact that the `pip install` of deeplabcut wasn't installing cudnn

## Opening deeplabcut

- in your anaconda prompt, you can use `python -m deeplabcut` to bring up the DeepLabCut GUI and see if it all installed
  correctly
- The instructions below cover how to run DeepLabCut as headlessly as possible

# Creating a DLC config file

- with DLC working, we can make a config file.
- I created [this wrapper function](/create_project.py) in `create_project.py` to make this slightly easier
- there's a `ProjectConfig` model that takes in the necessary info to create a `config.yaml` file.
    - fill in the project name (use underscores, not spaces, if needed) and the experimentor name
    - the `video_directory` is to the folder of initial videos you'd like to include (we can easily add more later).
      The `create_dlc_project` will iterate through all the videos in the folder and add them to the config file
    - the `working_directory` is where you want to save the DLC project folder
    - the `copy_videos_to_dlc_folder` is, as the name says, whether you want to copy the videos from
      the `video_directory` folder into the deeplabcut folder. If `False`, I believe it will create symbolic links to
      the videos. However, if you are using any sort of external drive, it will copy them over regardless. Something
      about symbolic links not working for that.
- Once you run the project and create the config file, you should open the `config.yaml` folder in the newly created
  project folder and change the `bodyparts` to your markers, and also define any connections in `skeleton`. For example,
  for the wobble board, my `bodyparts` looked like:

```
bodyparts:
- roller
- mid_board
- left_board
- right_board
- front_board
- back_board
```

and my skeleton looked like:

```
skeleton:
- - left_board
  - mid_board
- - mid_board
  - right_board
- - mid_board
  - front_board
- - mid_board
  - back_board
```

# Adding New Videos

- To add new videos, I've made a [wrapper function here](/add_new_videos_to_project.py)
  called `add_new_videos_to_project.py`.
- If you have a single video to add, you can just use
  the `deeplabcut.add_new_videos(config=config_path, videos=new_videos, copy_videos=False)` command - where `videos` is
  a list with the `str` path to your video.
- If you had a folder full of videos, just use the wrapper function I wrote to add all of them. You'll see the paths to
  the new videos show up in the `config.yaml`

## **For the next few sections (Extracting Frames until Evaluate Network), no wrapper functions are needed, so I made
a [Jupyter notebook here](/run_deeplabcut.ipynb) called `run_deeplabcut.ipynb` to list out what the headless commands
are**

# Extracting Frames

- Now that videos have been added, we'll want to extract frames from them to label. The `extract_frames` command will
  atuomatically extract frames to label from each video
- run `deeplabcut.extract_frames()` from the [jupyter notebook here](/run_deeplabcut.ipynb)
- The `numframes2pick` in the `config.yaml` file is the number of frames that gets extracted. Default is 20 - I set it
  to 10.
- You might note that there's a `userfeedback` flag in the `extract_frames` command, but that just prompts you to type
  in `y` or `n` before it proceeds with extraction for *each* video, so I wouldn't recommend it. It's annoying.
- Extracted frames can be found in the `labeled-data` folder
- *Note:* Folders for the extracted frames are created based on video name. Because we use the same naming conventions
  for all our freemocap synced videos (`camera_0_synchronized` or whatever it is), every `camera_0` video that you
  extract videos from will go to the same folder in the deeplabcut project. I don't know yet if this is a bad thing or
  not important, I'm just making a note of it.

# Labeling Data

- I cannot get this to run headlessly. The labeling Napari GUI crashes whenever I bring it up
    - to make it work I had to run the GUI (`python -m deeplabcut`) and open the labeling GUI from there
    - *update:* I ran this command from a Jupyter notebook cell for the first time today and I think it worked?
- Run `deeplabcut.label_frames()` from the jupyter notebook
- A Napari GUI should show up. Sometimes the GUI is hidden behind your other windows, but you should see it appear in
  the taskbar
- Use `File > Open Folder` to open the folder of extracted frames. You'll find these folders in
  your `dlc_project_path/labeled-data` folder, with a different folder for each video. You'll have to label each folder,
  so pick one to start.
- Some notes about the Napari GUI
    - On the bottom left, you'll see the color associated with each marker. On the bottom right, you'll see a dropdown
      menu, the one selected is the marker currently being labeled
    - There are different modes of how you can label the marker (located in the top right I think)
        - `sequential` and `quick` have you label every marker per frame before moving to the next frame
        - `loop` lets use label one marker on every frame before moving to the next marker. This is my preferred method
- When finished labeling, use `Ctrl + S` to save the data. I've found that to label the next folder, you need to restart
  the GUI and open the new folder, but if someone finds a different path lmk.

# Check Labels

- Creates a folder of images in the `labeled_data` folder of the DLC project that shows you where the labels were placed

# Create Training Dataset

- Just run the DeepLabCut command for this

# Train Network

- Use this to start training the network. This is the one that might take awhile.
    - the `maxiters` flag sets the maximum iterations that it will train the project for. For the wobble board I noticed
      that the `loss` you see printed in the terminal plateaued after 100k iterations, so I set it to that. If you don't
      run it long enough, it might not minimize the loss, but if you run it too long, it might overfit the data (I
      think? Don't quote me on that)
    - the `saveiters` flag is after what interval of iterations you want to save the data
    - the `displayiters` flag is how how often you want a progress update in the terminal (so every `x` amount of
      iterations)

# Evaluate network

- This command creates a new subdirectory with CSV's that have scores and labeled images. I have no idea how to
  interpret these

# Analyze Videos

- this is the guy that lets you use the model you created on other videos. I've made a wrapper function that does this
  and more [here](/analyze_videos.py) called `analyze_videos.py`
- the `process_recording_folder` function:
    - takes in the path to the `freemocap` recording folder whose videos you want to analyze
    - creates a `dlc_data` folder
    - it copies the videos from the `synchronized_videos` folder into a created subfolder in the `dlc_data`
        - *NOTE*: the reason I copy the videos is because as they're copied, they each get an identifier with the
          recording name. In [Extracting Frames](#extracting-frames), you can see the note I made at the bottom about
          whether having videos of the same name would be a problem. As of writing this note, I believe they will
          actually be a problem, so this is a stop-gap measure to work around that, so that when you extract outlier
          frames below, frames from different videos will save to unique folders instead of the same one. The trade-off
          is the space/time needed to unncessarily copy videos just so that they have a different name.
    - saves `csv` data from the analysis into the `dlc_data` folder
    - Uses `deeplabcut.filterpredictions` to filter the data and saves that data into the `dlc_data` folder (the
      filtered data will have `_filtered` in the name)
    - Creates labeled videos and saves them into the `dlc_data` folder
- the `process_session` function is for batch-processing a bunch of recordings in the same session folder. This is for
  example, how I applied the deeplabcut model I trained for the prosthetic data to all the prosthetic recordings

# Extract Outlier Frames

- This function takes the path to where the videos that you want to analyze are, and the path where the dlc data is
  stored (since we save the data into the dlc_data folder, but store the videos to analyze in a subfolder, we need both
  paths.)
- `Automatic = True` means that it won't prompt you for user feedback to extract the outliers for each
- The extracted frames should be saved out in `your_dlc_project_folder/labeled-folders/[insert_video_name_here]` for
  each video. Each video you extracted from should have its own folder

# Refining the labels

- Use the `refine_labels` function in the Jupyter notebook
- After running this, a DeepLabCut GUI should pop up. Similar to the first time you labeled frames, you'll want to open
  the folders with the frames you extracted in the last step.
- After choosing a folder, **click the machine-labels layer on the left side**, those are the labels you refine
- You can press the letter 'e' on your keyboard and it should bring up little outlines around the markers - if they're
  in red, it means they're especially off
- Press the number '3' to change to the marker selection tool and start dragging the labels around that need refining
- After finishing, save with `Ctrl + S`, and load the next folder

# After you've refined the labels and are ready to retrain the model

- Run the `merge_dataset` code block in the Jupyter notebook. This should merge your datasets to create a new iteration
  of data, and then create a new training dataset based off of that.
- After that, just go ahead and run the `train_network` command again

# Re-analyzing videos

- You can use the same `analyze_videos.py` functions on all the videos you want data from. As a note though,
  re-analyzing the videos will overwrite the existing saved data, so I'll copy the existing data into a subfolder
  called `iteration_[x]` to hold onto it just in case


- I have methods to compile all the saved out csvs into 2d freemocap format and then reconstruct them - if you manage to
  get this far before I've copied them over into this repo and written then up then just let me know