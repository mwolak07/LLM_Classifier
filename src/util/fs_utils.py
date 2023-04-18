import os


def cd_to_executing_file(file: str) -> None:
    """
    Changes to the directory of the file currently being executed.

    Args:
        file: The name of the caller's file, (__file__)
    """
    executing_file = os.path.abspath(file)
    executing_folder = os.path.dirname(executing_file)
    os.chdir(executing_folder)
