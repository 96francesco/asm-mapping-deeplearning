import os

def delete_local_files(files):
      for file in files:
            if os.path.exists(file):
                  os.remove(file)