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


def get_fpath(f):
    """get the absolute path of a file
    Args:
        f: 

    Return: 
    """
    return os.path.dirname(os.path.abspath(f))
