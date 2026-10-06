from pathlib import Path

def create_directory(directory_path):
    """
    Create a directory, including any missing parent directories.

    If the directory already exists, nothing is changed. A message
    with the directory path is printed to the console.

    Parameters
    ----------
    directory_path : pathlib.Path
        Path of the directory to create.

    Returns
    -------
    None
    """
    directory_path.mkdir(parents=True,exist_ok=True)
    print(f"{directory_path} created.")


    



