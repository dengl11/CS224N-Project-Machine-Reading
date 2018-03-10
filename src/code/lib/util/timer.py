############################################################
##############               Timer                 #########
############################################################
import time

class Timer(object):

    """timer class"""
    starts = {} # {key: start_time}
    ends   = {} # {key: end_time}


    def __init__(self):
        pass

    def start(self, key):
        """start timer for key
        Args:
            key: 

        Return: 
        """
        assert(key not in self.starts, "{} already started!".format(key))
        self.starts[key] = time.time()
        self.ends.pop(key, None)


    def stop(self, key):
        """stop timer for key
        Args:
            key: 

        Return: elapsed time 
        """
        assert(key not in self.ends, "{} already ended!".format(key))
        assert(key in self.starts, "{} not started!".format(key))
        now = time.time()
        self.ends[key] = now 
        elapsed = now - self.starts.pop(key)
        return elapsed 
