############################################################
###############  Toolkit for System Operations  #############
############################################################
import os, sys
from os import walk


def create_dir(directory):
    """
    create a directory
    """
    try: os.stat(directory)
    except: 
            print("{} not existing. Just created!".format(directory))
            os.mkdir(directory)



def files_in_folder(folder, format="jpeg"):
    """
    return a list of file names in folder

    Input:   path of folder
    Output:  [file_names]
    """
    try: os.stat(folder)
    except: 
            print("{} is not a valid path!".format(folder))
            return
    imgs = [p[2] for p in walk(folder)][0]
    imgs = list(filter(lambda x:  x.endswith(format), imgs))
    return imgs


def dirs_in_dir(parent_path):
    """get list of directories in parent folder
    Args:
        dir_path: [abs_path for each subdirectory]

    Return: 
    """
    dirs = []
    for dir_path in os.listdir(parent_path):
        dir_path = os.path.join(parent_path, dir_path)
        if not os.path.isdir(dir_path): continue 
        dirs.append(dir_path)
    return dirs 
    

def get_fpath(f):
    """get the absolute path of a file
    Args:
        f: 

    Return: 
    """
    return os.path.dirname(os.path.abspath(f))
