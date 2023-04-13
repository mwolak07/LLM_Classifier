import os


def cd_from_root(folder: str) -> None:
    """
    Changes directories to given directory, if we are in the root directory.

    Args:
        folder: The folder on the root directory level that we want to cd into.
    """
    if folder in os.listdir():
        os.chdir(folder)
