import os
import shutil

def empty_directory(directory):
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

dirs_to_empty = ["raw_videos", "enhanced_videos", "temp_frames", "enhanced_frames"]

for d in dirs_to_empty:
    empty_directory(d)
    os.makedirs(d, exist_ok=True)
