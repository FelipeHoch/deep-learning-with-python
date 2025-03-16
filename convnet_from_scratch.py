# import os, shutil, pathlib

# original_dir = pathlib.Path(r"/home/ai/PetImages")

# new_base_dir = pathlib.Path(r"/home/ai/cats_vs_dogs_small")

# def make_subset(subset_name, start_index, end_index):
#     for category in ("Cat", "Dog"):
#         dir = new_base_dir / subset_name / category

#         os.makedirs(dir)

#         fnames = [f"{i}.jpg"
#                     for i in range(start_index, end_index)]
        
#         for fname in fnames:
#             shutil.copyfile(src=original_dir / category / fname, dst=dir / fname)

# make_subset("train", start_index=0, end_index=1000)

# make_subset("validation", start_index=1000, end_index=1500)

# make_subset("test", start_index=1500, end_index=2500)
