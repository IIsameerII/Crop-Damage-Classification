import os

def count_files_in_directory(directory):
    """
    Count the number of files in the given directory.

    :param directory: Path to the directory
    :return: Number of files in the directory
    """
    if not os.path.isdir(directory):
        print(f"The specified path {directory} is not a directory.")
        return

    # Count the number of files
    file_count = len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])
    return file_count