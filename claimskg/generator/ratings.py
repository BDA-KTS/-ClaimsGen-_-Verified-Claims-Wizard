from enum import Enum
from typing import Dict


class NormalizedRatings(Enum):
    """
    Enum class representing the normalized ratings for different fact-checking sources.

    The ratings are standardized and categorized as follows:
    - FALSE: Indicates false information.
    - MIXTURE: Indicates a mixture of true and false information.
    - TRUE: Indicates true information.
    - OTHER: Indicates ratings that do not fall into the above categories.

    Each rating value corresponds to a unique integer.

    Examples:
        NormalizedRatings.FALSE    # Represents false information
        NormalizedRatings.MIXTURE  # Represents a mixture of true and false information
        NormalizedRatings.TRUE     # Represents true information
        NormalizedRatings.OTHER    # Represents other ratings
    """
    FALSE = 1
    MIXTURE = 2
    TRUE = 3
    OTHER = -1


_normalization_dictionary = {  # type: Dict[str, Dict[str,NormalizedRatings]]

    "politifact": {  # type: Dict[str,NormalizedRatings]
     # Dictionary mapping original ratings to normalized ratings for the "politifact" source.
        'incorrect': NormalizedRatings.FALSE,
        'pants-fire': NormalizedRatings.FALSE,
        'pants on fire': NormalizedRatings.FALSE,
        'pants on fire!': NormalizedRatings.FALSE,
        'false': NormalizedRatings.FALSE,
        'mostly correct': NormalizedRatings.MIXTURE,
        'mostly false': NormalizedRatings.MIXTURE,
        'barely true': NormalizedRatings.MIXTURE,
        'half true': NormalizedRatings.MIXTURE,
        'half-true': NormalizedRatings.MIXTURE,
        'mostly true': NormalizedRatings.MIXTURE,
        'true': NormalizedRatings.TRUE,
        'correct': NormalizedRatings.TRUE,
        'half False': NormalizedRatings.MIXTURE,
        'full-flop': NormalizedRatings.OTHER,
        'half-flip': NormalizedRatings.OTHER,
        'no-flip': NormalizedRatings.OTHER
        
        
    },
    "snopes": {  # type: Dict[str,NormalizedRatings]
    # Dictionary mapping original ratings to normalized ratings for the "snopes" source.
        'false': NormalizedRatings.FALSE,
        'legend': NormalizedRatings.OTHER,
        'mixture': NormalizedRatings.MIXTURE,
        'unfounded:': NormalizedRatings.OTHER,
        'true': NormalizedRatings.TRUE,
        'mostly false': NormalizedRatings.MIXTURE,
        'mostly true': NormalizedRatings.MIXTURE,
        'partly true': NormalizedRatings.MIXTURE,
        'mixture': NormalizedRatings.MIXTURE,
        'miscaptioned': NormalizedRatings.FALSE,
        'misattributed': NormalizedRatings.FALSE,
        'unproven': NormalizedRatings.OTHER,
        'originated as satire': NormalizedRatings.OTHER,
        'misattributed': NormalizedRatings.FALSE,
        'outdated': NormalizedRatings.OTHER,
        'correct attribution': NormalizedRatings.TRUE,
        'legit': NormalizedRatings.TRUE,
        'labelled satire': NormalizedRatings.OTHER,
        'lost legend': NormalizedRatings.OTHER,
        'recall': NormalizedRatings.OTHER,
        'MIXTURE OF TRUE AND FALSE INFORMATION': NormalizedRatings.MIXTURE,
        'MIXTURE OF TRUE AND FALSE INFORMATION:': NormalizedRatings.MIXTURE,
        'MIXTURE OF ACCURATE AND  INACCURATE INFORMATION': NormalizedRatings.MIXTURE
    },
    "africacheck": {  # type: Dict[str,NormalizedRatings]
    # Dictionary mapping original ratings to normalized ratings for the "africacheck" source.
        'incorrect': NormalizedRatings.FALSE,
        'mostly-correct': NormalizedRatings.MIXTURE,
        'correct': NormalizedRatings.TRUE,
        "unproven": NormalizedRatings.OTHER,
        "misleading": NormalizedRatings.FALSE,
        "exaggerated": NormalizedRatings.MIXTURE,
        "understated": NormalizedRatings.OTHER,
        "checked": NormalizedRatings.OTHER,
        "true": NormalizedRatings.TRUE,
        "false": NormalizedRatings.FALSE,
        "partlyfalse": NormalizedRatings.MIXTURE,
        "partlytrue": NormalizedRatings.MIXTURE,
        "fake": NormalizedRatings.FALSE,
        "scam": NormalizedRatings.FALSE,
        "satire": NormalizedRatings.OTHER,
    },
    "factscan": {  # type: Dict[str,NormalizedRatings]
    # Dictionary mapping original ratings to normalized ratings for the "factscan" source.
        'false': NormalizedRatings.FALSE,
        'true': NormalizedRatings.TRUE,
        'Misleading': NormalizedRatings.OTHER
    },
    "truthorfiction": {  # type: Dict[str,NormalizedRatings]
    # Dictionary mapping original ratings to normalized ratings for the "truthorfiction" source.
        'fiction': NormalizedRatings.FALSE,
        'truth': NormalizedRatings.TRUE,
        'Mixed': NormalizedRatings.MIXTURE,
        'Reported as Fiction': NormalizedRatings.MIXTURE,
        'truth & misleading': NormalizedRatings.MIXTURE,
        'mostly truth': NormalizedRatings.MIXTURE,
        'Decontextualized': NormalizedRatings.MIXTURE,
        'Not True': NormalizedRatings.MIXTURE,
        'true': NormalizedRatings.TRUE,
        'Unknown': NormalizedRatings.OTHER,
        'Misattributed': NormalizedRatings.OTHER,
        'Disputed': NormalizedRatings.MIXTURE,
        'Outdated': NormalizedRatings.OTHER,
        'Incorrect Attribution': NormalizedRatings.OTHER,
        'Correct Attribution': NormalizedRatings.OTHER,
        'Commentary': NormalizedRatings.OTHER,
        'Reported as Truth': NormalizedRatings.TRUE,
        'Mostly Fiction': NormalizedRatings.FALSE,
        'Unproven': NormalizedRatings.OTHER,
        'Authorship Confirmed': NormalizedRatings.OTHER,
        'Truth! & Fiction! & Unproven!': NormalizedRatings.MIXTURE,
        'Depends on Where You Vote': NormalizedRatings.MIXTURE,
        'Unofficial': NormalizedRatings.OTHER,
        'Truth! But an Opinion!': NormalizedRatings.MIXTURE,
        'Truth! But Postponed!': NormalizedRatings.MIXTURE,
        'Pending Investigation!': NormalizedRatings.OTHER,
        
    },
    "checkyourfact": {  # type: Dict[str,NormalizedRatings]
    # Dictionary mapping original ratings to normalized ratings for the "checkyourfact" source.
        'false': NormalizedRatings.FALSE,
        'true': NormalizedRatings.TRUE,
        'mostly true': NormalizedRatings.MIXTURE,
        'true/false': NormalizedRatings.MIXTURE,
        'verdict false': NormalizedRatings.FALSE,
        'mostly truth': NormalizedRatings.MIXTURE,
        'misleading': NormalizedRatings.FALSE,
        'fal': NormalizedRatings.FALSE,
        'unsubstantiated': NormalizedRatings.OTHER,
        'verdict': NormalizedRatings.FALSE
    },
    "factcheck_aap": {
    # Dictionary mapping original ratings to normalized ratings for the "factcheck_aap" source.
        "True": NormalizedRatings.TRUE,
        "False": NormalizedRatings.FALSE,
        "Mostly True": NormalizedRatings.MIXTURE,
        "Mostly False": NormalizedRatings.MIXTURE,
        "Somewhat True": NormalizedRatings.MIXTURE,
        "Somewhat False": NormalizedRatings.MIXTURE
    },
    "factual_afp": {
    # Dictionary mapping original ratings to normalized ratings for the "factual_afp" source.
        'faux': NormalizedRatings.FALSE,
        'article satirique': NormalizedRatings.FALSE,
        'infondé': NormalizedRatings.FALSE,
        'montage': NormalizedRatings.FALSE,
        'trompeur': NormalizedRatings.MIXTURE,
        'parodie': NormalizedRatings.FALSE,
        'vrai': NormalizedRatings.TRUE,
        'Contexte manquant': NormalizedRatings.FALSE,
        'propos sortis de leur contexte': NormalizedRatings.FALSE,
        'manque de contexte': NormalizedRatings.FALSE,        
        "faux, ces photos montrent un couple britannique sans aucun lien de parenté et illustrent un article satirique": NormalizedRatings.FALSE,
        'faux, manque de contexte : vidéo tronquée': NormalizedRatings.FALSE,
        'totalement vrai': NormalizedRatings.TRUE,
        'plutôt vrai': NormalizedRatings.MIXTURE,        
        'trompeur': NormalizedRatings.MIXTURE,
        'plutôt faux': NormalizedRatings.MIXTURE,
        'presque': NormalizedRatings.MIXTURE,
        'mélangé': NormalizedRatings.MIXTURE,
        'Inexact': NormalizedRatings.MIXTURE,
        'Incertain': NormalizedRatings.MIXTURE,
        'Imprécis': NormalizedRatings.MIXTURE,
        'Exagéré': NormalizedRatings.MIXTURE,
        'Douteux': NormalizedRatings.MIXTURE,

    },

     "factcheck_afp": {
     # Dictionary mapping original ratings to normalized ratings for the "factcheck_afp" source.
        'false': NormalizedRatings.FALSE,
        'partly false': NormalizedRatings.MIXTURE,
        'misleading': NormalizedRatings.FALSE,
        'satire': NormalizedRatings.FALSE,
        'missing context': NormalizedRatings.FALSE,
        'altered image': NormalizedRatings.OTHER,
        'altered photo': NormalizedRatings.OTHER,
        'not recommended' : NormalizedRatings.OTHER,
        'true' : NormalizedRatings.TRUE,
        'unproven': NormalizedRatings.OTHER,
        'no evidence': NormalizedRatings.OTHER,
        'photo out of context': NormalizedRatings.OTHER,
        'misattributed': NormalizedRatings.FALSE,
        'Outdated': NormalizedRatings.OTHER,
        'video lacks context': NormalizedRatings.OTHER
    },
    "fullfact": {
    # Dictionary mapping original ratings to normalized ratings for the "fullfact" source.
        'true': NormalizedRatings.TRUE,
        'false': NormalizedRatings.FALSE,
        'mixture': NormalizedRatings.MIXTURE,
        'other': NormalizedRatings.OTHER
    },
    "eufactcheck": {
    # Dictionary mapping original ratings to normalized ratings for the "eufactcheck" source.
       
        'm': NormalizedRatings.MIXTURE,
        'f': NormalizedRatings.FALSE,
        't': NormalizedRatings.TRUE,        
        'u': NormalizedRatings.OTHER
        
    },
      "polygraph": {
      # Dictionary mapping original ratings to normalized ratings for the "polygraph" source.
        'misleading': NormalizedRatings.MIXTURE,
        'true': NormalizedRatings.TRUE,
        'false': NormalizedRatings.FALSE,       
        'unsubstantiated': NormalizedRatings.FALSE,
        'Dangerously Fake': NormalizedRatings.FALSE,
        'FALSE': NormalizedRatings.FALSE,
        'False': NormalizedRatings.FALSE,
        'False and misleading': NormalizedRatings.FALSE,
        'False or Misleading': NormalizedRatings.FALSE,
        'False or misleading': NormalizedRatings.FALSE,
        'Likely False': NormalizedRatings.MIXTURE,
        'Likely false': NormalizedRatings.MIXTURE,
        'Likely true': NormalizedRatings.MIXTURE,
        'MIsleading': NormalizedRatings.FALSE,
        'Misleading': NormalizedRatings.FALSE,
        'Mostly False': NormalizedRatings.FALSE,
        'Partially False': NormalizedRatings.MIXTURE,
        'Partially True': NormalizedRatings.MIXTURE,
        'Partly False': NormalizedRatings.MIXTURE,
        'Partly False and Misleading': NormalizedRatings.FALSE,
        'True': NormalizedRatings.MIXTURE,
        'UNFOUNDED': NormalizedRatings.OTHER,
        'Uncertain': NormalizedRatings.OTHER,
        'Unclear': NormalizedRatings.OTHER,
        'Unclear and Partially True': NormalizedRatings.MIXTURE,
        'Unsubstantiated': NormalizedRatings.OTHER       
        
    },
    "fatabyyano": {
    # Dictionary mapping original ratings to normalized ratings for the "fatabyyano" source.
        'false': NormalizedRatings.FALSE,
        'altered': NormalizedRatings.MIXTURE,      
        'partially false': NormalizedRatings.MIXTURE,
        'satire': NormalizedRatings.OTHER,
        'missing context': NormalizedRatings.OTHER,
        'true': NormalizedRatings.TRUE     
        
    },
    "factograph": {
    # Dictionary mapping original ratings to normalized ratings for the "factograph" source.
        'не факт': NormalizedRatings.FALSE,
        'это так': NormalizedRatings.TRUE,
        'да_но': NormalizedRatings.MIXTURE,
        'ДА_НО': NormalizedRatings.MIXTURE,
        'ДА_НО…': NormalizedRatings.MIXTURE,
        'ДА_НО': NormalizedRatings.MIXTURE,
        'Тak_но,': NormalizedRatings.MIXTURE,
        'пока не факт': NormalizedRatings.FALSE, # for now it should be false    
        'скорее_так': NormalizedRatings.MIXTURE,# true but not 100%
        'не факт_но': NormalizedRatings.MIXTURE,# false but....
        'не факт_но...': NormalizedRatings.MIXTURE,
        'видимо_так': NormalizedRatings.TRUE,#seems to be true
        'не факт_увы,': NormalizedRatings.FALSE,
        'пока_скорее_так': NormalizedRatings.MIXTURE, # for now it should be truth
        'ПОКА_СКОРЕЕ_ТАК' : NormalizedRatings.MIXTURE,
        'сомнительно': NormalizedRatings.MIXTURE, # we doubts its not true
        'СОМНИТЕЛЬНО': NormalizedRatings.MIXTURE,
        'искажение': NormalizedRatings.OTHER,#interpreted wrongly
        'скорее_так_но,': NormalizedRatings.MIXTURE,#seems to be true but..
        'это так_но,': NormalizedRatings.MIXTURE,#true..but        
        'правда': NormalizedRatings.TRUE,
        'ПРАВДА': NormalizedRatings.TRUE,
        'пока сомнительно': NormalizedRatings.MIXTURE,# for now we dont think its true
        'скорее_правда': NormalizedRatings.MIXTURE,# seems to be true;NOT SURE
        'СКОРЕЕ_ПРАВДА': NormalizedRatings.MIXTURE,
        'неправда': NormalizedRatings.FALSE,              
        'возможно_но': NormalizedRatings.MIXTURE   # may be true but...
        
        
    },
    
    "vishvanews": {
    # Dictionary mapping original ratings to normalized ratings for the "vishvanews" source.
        'false': NormalizedRatings.FALSE,
        'misleading': NormalizedRatings.MIXTURE,      
        'true': NormalizedRatings.TRUE,
        'False': NormalizedRatings.FALSE,
        'Misleading': NormalizedRatings.MIXTURE,      
        'True': NormalizedRatings.TRUE,
        
    }
}


def _standardize_name(original_name: str):
    """
    Standardize the name of a fact-checking source by removing special characters and converting to lowercase.

    Args:
        original_name (str): The original name of the fact-checking source.

    Returns:
        str: The standardized name.
    """
    return original_name.strip().lower().replace("!", "").replace(":", "").replace("-", " ")


def normalize(source_name, original_name) -> NormalizedRatings:
    """
        Generate a normalized rating from the original ratings on each respective site
    :param original_name:
    :return normalized_rating: NormalizedRating
    """
    try:
        source = _normalization_dictionary[source_name]
       
        normalized_value = source[_standardize_name(original_name)]
      
    except KeyError:
        normalized_value = NormalizedRatings.OTHER
    return normalized_value
