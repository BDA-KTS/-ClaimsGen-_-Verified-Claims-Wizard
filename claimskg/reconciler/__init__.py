import itertools
from datetime import timedelta

import numpy
import redis
from nltk.corpus import stopwords
from tqdm import tqdm

from claimskg import similarity as sim
from claimskg.vsm.embeddings import Embeddings

_stop_words = set(stopwords.words('english'))
import re

stopword_pattern = re.compile(r'\b(' + r'|'.join(_stop_words) + r')\b\s*')


def _merge_and_normalise_strings(strings):
    """
    Merge a list of strings and normalize the resulting string.

    Parameters:
        strings (List[str]): List of strings to be merged and normalized.

    Returns:
        str: The merged and normalized string.
    """
    return re.sub(r'\[.*?\]|\(.*?\)|\W', ' ', stopword_pattern.sub("", " ".join(strings).strip().lower()))


def _process_pairwise_sample(sample_size, collection, seed, callback, *args):
    """
    Process pairwise samples from the collection using the given callback function.

    Parameters:
        sample_size (int): The number of pairwise samples to generate.
        collection (List[Any]): The collection from which pairwise samples are generated.
        seed (Optional[int]): Seed for random number generation. Default is None.
        callback (Callable): The callback function to be applied to each pairwise sample.
        *args: Additional arguments to be passed to the callback function.

    Returns:
        List[Any]: The list of results obtained from applying the callback to the pairwise samples.
    """
    count = len(collection)
    generator_list = itertools.combinations(range(count), 2)
    iterlen = int(count * (count - 1) / 2)
    progress_bar = tqdm(total=sample_size)

    random = numpy.random
    if seed is not None:
        random.seed(seed)

    num = 0
    inds = random.random(iterlen) <= (sample_size * 1.0 / (iterlen))
    results = []
    iterator = iter(generator_list)
    gotten = 0
    while gotten < sample_size and num < iterlen - 1:
        b = next(iterator)
        if inds[num]:
            results.append(callback((collection[b[0]], collection[b[1]]), *args))
            gotten += 1
            progress_bar.update(1)
        num += 1
        if num == iterlen:
            num = 0
            iterator = iter(generator_list)
            inds = numpy.random.random(iterlen) <= ((sample_size - gotten) * 1.0 / iterlen / 3)
    print("Post")
    progress_bar.close()
    return results


class FactReconciler:

    def __init__(self, embeddings: Embeddings, caching: bool, mappings_file_path: str, claims, theta: float,
                 keyword_weight,
                 link_weight, text_weight, entity_weight, seed=None, samples=None):
        """
        Initialize the FactReconciler instance.

        Parameters:
            embeddings (Embeddings): The embeddings used for similarity computation.
            caching (bool): Flag indicating whether caching is enabled for the reconciler.
            mappings_file_path (str): The file path for writing the generated mappings. Set to None if no file output is needed.
            claims (List): The list of claims to be reconciled.
            theta (float): The threshold value for generating mappings based on the similarity score.
            keyword_weight: Weight for keyword similarity in the score computation.
            link_weight: Weight for link similarity in the score computation.
            text_weight: Weight for text similarity in the score computation.
            entity_weight: Weight for entity similarity in the score computation.
            seed (Optional[int]): Seed for random number generation. Default is None.
            samples (Optional[int]): Number of pairwise samples to generate. If provided, the reconciler will use a random sampling approach. Default is None.
        """
        self._embeddings = embeddings
        self._caching = caching
        if caching:
            self._redis = redis.StrictRedis()
        else:
            self._redis = None

        if mappings_file_path is not None:
            self.output_file = open(mappings_file_path, "w")
        else:
            self.output_file = None

        self._processed_set = set()

        self.claims = claims

        self.theta = theta
        self.keyword_weight = keyword_weight
        self.link_weight = link_weight
        self.text_weight = text_weight
        self.entity_weight = entity_weight
        self.claim_count = len(claims)
        self.seed = seed
        self.samples = samples

    # if self.output_file is not None:
    # self.output_file.write(
    # self._generate_claim_mapping_string_description(claim_a, claim_b))
    def generate_mappings(self):
        """
        Generate mappings between claims based on the provided threshold and similarity weights.

        Returns:
            List[Tuple[Claim, Tuple[Claim, Claim]]]: A list of mappings between claims. Each mapping is represented as a tuple containing:
                - Claim: The source claim that is mapped to the target claims.
                - Tuple[Claim, Claim]: A tuple of two target claims that are mapped to the source claim. Can be (None, None) if no mapping is found.
        """

        if self.output_file is not None:
            self.output_file.write(FactReconciler._generate_claim_mapping_output_header())

        # Iterate over pairs of claims on the upper diagonal of the grid (order doesn't matter)
        # Compute the score for each evaluated pair and create a new mapping if the score is
        # above the threshold (theta)

        # one_tenth_per_cent = int(total * 0.0001)
        # completed = 0
        # mapped = 0
        # computed = 0
        # index_a = len(self.claims) - 1

        if self.samples is not None:
            result = _process_pairwise_sample(self.samples, self.claims, self.seed, self._evaluate_mapping)
        else:
            count = len(self.claims)
            total = int(count * (count - 1) / 2)
            result = [self._evaluate_mapping((self.claims[pair[0]], self.claims[pair[1]])) for pair in
                      tqdm(itertools.combinations(range(len(self.claims)), 2), total=total)]

        print(len(result))
        mappings = [x for x in result if x is not None]
        mappings = [
            (x, y) for x, y in mappings if (x, y) != (None, None)
        ]
        print(len(mappings))

        if self.output_file is not None:
            for mapping in mappings:
                if mapping is not None and mapping[1] is not None and mapping[1] != (None, None):
                    self.output_file.write(
                        FactReconciler._generate_claim_mapping_string_description(mapping[0], mapping[1][0],
                                                                                  mapping[1][1]))
            self.output_file.flush()
            self.output_file.close()

        return mappings

    @staticmethod
    def _generate_claim_mapping_output_header():
        """
        Generate the header string for the output file containing the claim mapping descriptions.

        Returns:
            str: The header string.
        """
        return "\"Score\", \"CR Author A\",\"CR Author B\", Review URL A, Review URL B,\"Text Fragments A\"," \
               "\"Text Fragments B\",\"Entities A\",\"Entities B\",\"Keywords A\",\"Keywords B\",\"Citations A\"," \
               "\"Citations B\",\"URI A\",\"URI B\"\n"

    @staticmethod
    def _generate_claim_mapping_string_description(score, claim_a, claim_b):
        """
        Generate the description string for a claim mapping.

        Parameters:
            score (float): The score of the claim mapping.
            claim_a (Claim): The source claim.
            claim_b (Claim): The target claim.

        Returns:
            str: The description string for the claim mapping.
        """
        entities_a = claim_a.claim_entities + claim_a.review_entities
        entities_b = claim_b.claim_entities + claim_b.review_entities
        return "{score},\"{cra_a}\",\"{cra_b}\",\"{ruri_a}\",\"{rurib_b}\",\"{tf_a}\",\"{tf_b}\",\"{ent_a}\",\"{ent_b}\"," \
               "\"{kw_a}\",\"{kw_b}\",\"{cit_a}\",\"{cit_b}\", \"{uri_a}\",\"{uri_b}\",\n" \
            .format(
            # Claim A fields
            uri_a=claim_a.creative_work_uri, ent_a=",".join(entities_a), kw_a=",".join(claim_a.keywords),
            cit_a=",".join(claim_a.links), ruri_a=claim_a.claim_review_url, cra_a=claim_a.claimreview_author,
            tf_a=claim_a.text_fragments[0].replace("\"", "''"),
            # Claim B fields
            uri_b=claim_b.creative_work_uri, ent_b=",".join(entities_b), kw_b=",".join(claim_b.keywords),
            cit_b=",".join(claim_b.links), rurib_b=claim_b.claim_review_url, cra_b=claim_b.claimreview_author,
            tf_b=claim_b.text_fragments[0].replace("\"", "''"),
            score=score)

    @staticmethod
    def _pruning_criterion(claim_a, claim_b):
        """
        Determine whether a claim pair should be pruned based on specific criteria.

        Parameters:
            claim_a (Claim): The first claim to compare.
            claim_b (Claim): The second claim to compare.

        Returns:
            bool: True if the claim pair should be pruned, False otherwise.
        """
        prune = False
        author_score = FactReconciler.author_match(claim_a, claim_b)

        entities_a = claim_a.claim_entities + claim_a.review_entities
        entities_b = claim_b.claim_entities + claim_b.review_entities

        entity_score = sim.compute_overlap(entities_a, entities_b)
        num_entities_a = len(entities_a)
        num_entities_b = len(entities_b)

        # Date criterion
        if claim_a.claim_date is not None and claim_b.claim_date is not None \
                and claim_a.claim_date != claim_b.claim_date:
            prune = True
        elif claim_a.review_date is not None and claim_b.review_date is not None \
                and claim_a.review_date - claim_b.review_date > timedelta(days=1):
            prune = True
        elif author_score != 1:
            prune = True
        elif num_entities_a != 0 and num_entities_b != 0 and entity_score <= 0.00001:
            prune = True
        elif num_entities_a == 0 and num_entities_b > 0 or num_entities_a > 0 and num_entities_b == 0:
            prune = True

        return prune

    @staticmethod
    def author_match(first, other):
        """
        Determine if the authors of two claims are a match.

        Parameters:
            first (Claim): The first claim to compare.
            other (Claim): The second claim to compare.

        Returns:
            int: 1 if the authors are a match, 0 otherwise.
        """
        if first.creative_work_author == other.creative_work_author:
            return 1
        else:
            return 0

    def _evaluate_mapping(self, pair):
        """
        Evaluate the mapping between two claims and determine if they should be linked.

        Parameters:
            pair (Tuple[Claim, Claim]): A tuple containing two claims to be evaluated.

        Returns:
            Tuple[float, Tuple[Claim, Claim]]: A tuple containing the similarity score and the mapping result.
        """
        result = None
        score = None
        claim_a = pair[0]
        claim_b = pair[1]
        key = claim_a.creative_work_uri + claim_b.creative_work_uri

        if key not in self._processed_set and claim_a != claim_b \
                and not FactReconciler._pruning_criterion(claim_a, claim_b):
            score = self._claim_similarity(claim_a, claim_b)
            self._processed_set.add(key)

            if score > self.theta:
                result = pair

        return score, result

    def _claim_similarity(self, claim_a, claim_b):
        """
        Calculate the similarity score between two claims.

        Parameters:
            claim_a (Claim): The first claim to compare.
            claim_b (Claim): The second claim to compare.

        Returns:
            float: The similarity score between the two claims.
        """
        entities_a = claim_a.claim_entities + claim_a.review_entities
        entities_b = claim_b.claim_entities + claim_b.review_entities
        categories_a = claim_a.review_entity_categories + claim_a.claim_entity_categories
        categories_b = claim_b.review_entity_categories + claim_b.claim_entity_categories

        if len(claim_a.keywords) == 0 and len(claim_b.keywords) == 0:
            keyword_similarity = None
        else:
            keyword_similarity = sim.jaccard(claim_a.keywords, claim_b.keywords)

        link_similarity = sim.jaccard(claim_a.links, claim_b.links)

        if len(entities_a) == 0 and len(entities_b) == 0:
            entity_similarity = None
        else:
            entity_similarity = sim.jaccard(entities_a, entities_b)

        if len(categories_a) == 0 and len(categories_b) == 0:
            category_similarity = None
        else:
            category_similarity = sim.jaccard(categories_a, categories_b)

        if not entity_similarity and category_similarity:
            entity_similarity = category_similarity * 0.3
        elif entity_similarity and category_similarity:
            entity_similarity = entity_similarity * 0.7 + category_similarity + 0.3

        if self._embeddings:
            text_similarity = self._embeddings.sentence_similarity(
                _merge_and_normalise_strings(claim_a.text_fragments),
                _merge_and_normalise_strings(claim_b.text_fragments[0:1]))
        else:
            text_similarity = None

        score = sim.geometric_mean_aggregation([
            (entity_similarity, self.entity_weight),
            (keyword_similarity, self.keyword_weight),
            (link_similarity, self.link_weight),
            (text_similarity, self.text_weight)])

        return score
