from pathlib import Path

def create_directory(directory_path):
    directory_path.mkdir(parents=True,exist_ok=True)
    print(f"{directory_path} created.")


    



