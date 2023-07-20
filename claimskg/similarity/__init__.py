import math as math
from typing import List, Tuple

from nltk import TreebankWordTokenizer
from nltk.corpus import stopwords
from textacy.similarity import levenshtein

tokenizer = TreebankWordTokenizer()

_stop_words = stopwords.words('english')


def compute_overlap(collection_a: List[str], collection_b: List[str], soft=False):
    """
    Compute the overlap count between two collections of strings.

    Parameters:
        collection_a (List[str]): First collection of strings.
        collection_b (List[str]): Second collection of strings.
        soft (bool, optional): If True, use Levenshtein distance for similarity calculation. Default is False.

    Returns:
        int: The overlap count between the two collections.

    Note:
        If soft is True, the Levenshtein distance is used to measure the similarity between strings.
    """
    overlap_count = 0
    index_a = 0
    while index_a < len(collection_a):
        item_a = collection_a[index_a]
        index_b = 0
        while index_b < len(collection_b):
            item_b = collection_b[index_b]
            if item_a == item_b and not soft:
                overlap_count += 1
            else:
                overlap_count += levenshtein(item_a, item_b)
            index_b += 1
        index_a += 1
    return overlap_count


def tverski_ratio(alpha: float, beta: float, gamma: float, overlap_count: float, difference_a: float,
                  difference_b: float):
    """
    Calculate the Tverski ratio, a similarity measure based on overlap count and differences between items.

    Parameters:
        alpha (float): Weight for overlap count in the ratio calculation.
        beta (float): Weight for difference_a in the ratio calculation.
        gamma (float): Weight for difference_b in the ratio calculation.
        overlap_count (float): The overlap count between two collections.
        difference_a (float): Difference measure for collection_a.
        difference_b (float): Difference measure for collection_b.

    Returns:
        float: The Tverski ratio.

    Note:
        The Tverski ratio is defined as alpha * overlap_count / contrast, where contrast is computed as
        alpha * overlap_count - beta * difference_a - gamma * difference_b.
    """

    contrast = tverski_contrast(alpha, beta, gamma, overlap_count, difference_a, difference_b)
    if contrast == 0:
        return 0
    else:
        return alpha * overlap_count / contrast


def tverski_contrast(alpha: float, beta: float, gamma: float, overlap_count: float, difference_a: float,
                     difference_b: float):
    """
    Calculate the Tverski contrast, used in the Tverski ratio calculation.

    Parameters:
        alpha (float): Weight for overlap count in the contrast calculation.
        beta (float): Weight for difference_a in the contrast calculation.
        gamma (float): Weight for difference_b in the contrast calculation.
        overlap_count (float): The overlap count between two collections.
        difference_a (float): Difference measure for collection_a.
        difference_b (float): Difference measure for collection_b.

    Returns:
        float: The Tverski contrast.
    """
    return alpha * overlap_count - beta * difference_a - gamma * difference_b


def jaccard_count(overlap_count: float, union_count: float):
    """
    Calculate the Jaccard count, which is the ratio of overlap count to the union count of two collections.

    Parameters:
        overlap_count (float): The overlap count between two collections.
        union_count (float): The union count of two collections.

    Returns:
        float: The Jaccard count.
    """
    if union_count == 0:
        return 0
    else:
        return overlap_count / union_count


def jaccard(collection_a, collection_b):
    """
    Calculate the Jaccard similarity coefficient between two collections based on their overlap.

    Parameters:
        collection_a (List[str]): First collection of strings.
        collection_b (List[str]): Second collection of strings.

    Returns:
        float: The Jaccard similarity coefficient.
    """
    overlap = compute_overlap(collection_a, collection_b)
    return jaccard_count(overlap, len(collection_a) + len(collection_b))


def geometric_mean_aggregation(weighted_values: List[Tuple[float, float]]):
    """
    Compute the geometric mean aggregation of weighted values.

    Parameters:
        weighted_values (List[Tuple[float, float]]): List of tuples containing values and their corresponding weights.

    Returns:
        float: The geometric mean aggregation of the weighted values.
    """
    length = len(weighted_values)
    overall_product = 1
    for (v, w) in weighted_values:
        if v is not None:
            if v < 0.00001:
                v = 0.00001
            overall_product *= math.pow(v, w)
    return math.pow(overall_product, 1.0 / float(length))


def arithmetic_mean_aggregation(weighted_values: List[Tuple[float, float]]):
    """
    Compute the arithmetic mean aggregation of weighted values.

    Parameters:
        weighted_values (List[Tuple[float, float]]): List of tuples containing values and their corresponding weights.

    Returns:
        float: The arithmetic mean aggregation of the weighted values.
    """
    length = len(weighted_values)
    overall_sum = 0.0
    for (v, w) in weighted_values:
        overall_sum += v * w

    return overall_sum / float(length)
