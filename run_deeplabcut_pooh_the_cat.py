# %% import stuff 
import deeplabcut 


# %% create or load project
deeplabcut.create_new_project('pooh-the-cat',
                               'jsm', 
                               [r"C:\Users\jonma\Sync\videos\2024-02-28_pooh_gopros\GX010034_clip1DLC_snapshot-700000_labeled.mp4"], 
                               working_directory=r'D:\deeplabcut-models\pooh-the-cat', 
                               copy_videos=True,)


# %%
