import json
import logging.config
from abc import ABC, abstractmethod

from nerd import nerd_client


class Annotator(ABC):
    """
    Abstract base class for annotators.
    """
    def __init__(self):
        pass

    @abstractmethod
    def annotate(self, text) -> str:
        """
        Annotate the given text and return the annotated entities as a JSON string.

        Args:
            text (str): The text to annotate.

        Returns:
            str: JSON string representing the annotated entities.
        """
        pass


class EntityFishingAnnotator(Annotator):
    """
    Annotator implementation using the Entity-Fishing API.
    """
    def __init__(self, api_uri="http://localhost:8090/service/"):
        """
        Initialize the EntityFishingAnnotator.

        Args:
            api_uri (str, optional): The URI of the Entity-Fishing API. Defaults to "http://localhost:8090/service/".
        """
        super().__init__()
        self._api_uri = api_uri
   
        self._client=nerd_client.NerdClient() 
        
     

    def annotate(self, text, language="en") -> str:
        """
        Annotate the text using the Entity-Fishing API.

        Args:
            text (str): The text to annotate.
            language (str, optional): The language of the text. Defaults to "en".

        Returns:
            str: JSON string representing the annotated entities.
       
       
        function that can use entity-fishing online: https://github.com/hirmeos/entity-fishing-client-python
        """
        # Disable the logger.debug message
        logging.config.dictConfig({
            'version': 1,
            'disable_existing_loggers': True,
        })
        try:

            response = self._client.disambiguate_text(text, language=language)
            if response[1] == 200:
                return json.dumps(response[0]['entities'])
            else:
                return json.dumps([])
        except Exception as e:
            print(str(e))
            return json.dumps([])
            
