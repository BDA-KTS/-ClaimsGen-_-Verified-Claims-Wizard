import numpy


class FileMappedEmbeddings:
    """
    An utility class to access word embeddings stored in a file.
    The file is expected to be in word2vec format, where each line starts with the word followed by its corresponding vector.

    Args:
        path (str): The file path of the word embeddings.

    Attributes:
        file (file object): The file object representing the word embeddings file.
        label_index (dict): A dictionary mapping words to their corresponding index in the embeddings file.
        file_start_offsets (list): A list of file offsets for each word in the embeddings file.
        dimensions (int): The number of dimensions of the word embeddings.

    Methods:
        vector(vocab_word: str) -> numpy.ndarray:
            Get the word embedding vector for a given vocabulary word.

    """

    def __init__(self, path):
        """
        Initialize the FileMappedEmbeddings object.

        Args:
            path (str): The file path of the word embeddings.
        """
        self.file = open(path, "r")
        self.label_index = {}
        self.file_start_offsets = []
        self.dimensions = None

        current_start_offset = self.file.tell()
        vocab_index = 0
        while self.file.readable():
            line = self.file.readline()
            self.label_index[line.split(" ")[0]] = vocab_index
            self.file_start_offsets.append(current_start_offset)

            current_start_offset = self.file.tell()
            vocab_index += 1

    def vector(self, vocab_word):
        """
        Get the word embedding vector for a given vocabulary word.

        Args:
            vocab_word (str): The vocabulary word for which to retrieve the embedding vector.

        Returns:
            numpy.ndarray: The word embedding vector for the given vocabulary word as a numpy array.
                           Returns a zero vector if the word is not found in the embeddings file.
        """
        index = self.label_index[vocab_word]
        if index:
            self.file.seek(self.file_start_offsets[index])
            parts = self.file.readline().split(" ")
            if not self.dimensions:
                self.dimensions = len(parts[1:])
            text_vector = " ".join(parts[1:])
            return numpy.fromstring(text_vector, sep=" ")
        else:
            return numpy.zeros((self.dimensions,))
