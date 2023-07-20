import requests


class TypedCounter:
    """
    A class that allows counting occurrences of different keys and keeps track of their counts.

    Example usage:
    counter = TypedCounter()
    counter.count('apple')  # Output: 1
    counter.count('banana')  # Output: 1
    counter.count('apple')  # Output: 2

    Attributes:
        counts (dict): A dictionary that stores the counts of different keys.
    """
    def __init__(self):
        """
        Initialize an empty TypedCounter.
        """
        self.counts = dict()

    def count(self, key):
        """
        Increment the count of the given key and return its updated count.

        Args:
            key: The key for which the count should be incremented.

        Returns:
            int: The updated count of the given key.
        """
        if key not in self.counts.keys():
            self.counts[key] = 0
        self.counts[key] += 1
        return self.counts[key]