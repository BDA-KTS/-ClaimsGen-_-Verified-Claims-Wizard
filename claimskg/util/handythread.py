import sys
import threading
from itertools import count


def foreach(f, l, threads=3, return_=False):
    """
    Apply the function f to each element of the list l in parallel using multiple threads.

    Args:
        f (function): The function to apply to each element of the list.
        l (list): The list of elements to apply the function to.
        threads (int, optional): The number of threads to use for parallel processing (default is 3).
        return_ (bool, optional): If True, the function will return the results in a list (default is False).

    Returns:
        list or None: If return_ is True, the function returns a list containing the results of applying the function
                      to each element of the input list. If return_ is False, the function returns None.

    Raises:
        Exception: If an exception occurs during the execution of the function f on any element in the list.
    """

    if threads > 1:
        iteratorlock = threading.Lock()
        exceptions = []
        if return_:
            n = 0
            dictionary = {}
            i = list(zip(count(), l.__iter__())).__iter__()
        else:
            i = l.__iter__()

        def runall():
            while True:
                iteratorlock.acquire()
                try:
                    try:
                        if exceptions:
                            return
                        v = i.next()
                    finally:
                        iteratorlock.release()
                except StopIteration:
                    return
                try:
                    if return_:
                        n, x = v
                        dictionary[n] = f(x)
                    else:
                        f(v)
                except:
                    e = sys.exc_info()
                    iteratorlock.acquire()
                    try:
                        exceptions.append(e)
                    finally:
                        iteratorlock.release()

        threadlist = [threading.Thread(target=runall) for j in range(threads)]
        for t in threadlist:
            t.start()
        for t in threadlist:
            t.join()
        if exceptions:
            a, b, c = exceptions[0]
            raise a(b).with_traceback(c)
        if return_:
            r = list(dictionary.items())
            r.sort()
            return [v for (n, v) in r]
    else:
        if return_:
            return [f(v) for v in l]
        else:
            for v in l:
                f(v)
            return


def parallel_map(f, l, threads=3):
    """
    Apply the function f to each element of the list l in parallel using multiple threads.

    This function is an alias for foreach with the return_ parameter set to True.

    Args:
        f (function): The function to apply to each element of the list.
        l (list): The list of elements to apply the function to.
        threads (int, optional): The number of threads to use for parallel processing (default is 3).

    Returns:
        list: A list containing the results of applying the function to each element of the input list.

    Raises:
        Exception: If an exception occurs during the execution of the function f on any element in the list.
    """
    return foreach(f, l, threads=threads, return_=True)
