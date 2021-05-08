import logging
import os
import requests
from typing import Any, List, Optional, Text, Dict

import rasa.utils.endpoints as endpoints_utils
from rasa.shared.nlu.constants import ENTITIES, TEXT
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.extractors.extractor import EntityExtractor
from rasa.nlu.model import Metadata
from rasa.shared.nlu.training_data.message import Message
import rasa.shared.utils.io

logger = logging.getLogger(__name__)

def convert_recognizers_format_to_rasa(
    matches: List[Dict[Text, Any]]
) -> List[Dict[Text, Any]]:
    extracted = []

    for match in matches:
        entity = {
            "start": match["start"],
            "end": match["end"],
            "text": match["text"],
            "value": match.get("resolution", {}).get("value"),
            "confidence": 1.0,
            "additional_info": match["resolution"],
            "entity": match["typeName"],
        }

        extracted.append(entity)

    return extracted


class RecognizersServiceEntityExtractor(EntityExtractor):
    """Searches for structured entites, e.g. dates, using recognizers-service."""

    defaults = {
        # by default all entities recognized by recognizers-service are returned
        # entities can be configured to contain an array of strings
        # with the names of the entities to filter for
        "entities": None,
        # by default all units are returned
        # units can be configured to contain an array of strings
        # with the names of the units to filter for
        "units": None,
        # http url of the running recognizers-service
        "url": None,
        # culture - if not set, we will use English (en-us)
        "culture": 'en-us',
        # a flag to have the service return original numbers in the resolution
        "show_numbers": True,
        # a flag to have the service merge overlapping entities
        "merge_results": True,
        # Timeout for receiving response from http url of the running recognizers-service
        # if not set the default timeout of recognizers-service url is set to 3 seconds.
        "timeout": 3,
    }

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
    ) -> None:
        super().__init__(component_config)

    @classmethod
    def create(
        cls, component_config: Dict[Text, Any], config: RasaNLUModelConfig
    ) -> "RecognizersServiceEntityExtractor":
        return cls(component_config)

    def _url(self) -> Optional[Text]:
        """Return url of recoginzers-service. Environment var will override."""
        if os.environ.get("RECOGNIZERS_SERVICE_URL"):
            return os.environ["RECOGNIZERS_SERVICE_URL"]

        return self.component_config.get("url")

    def _payload(self, text: Text) -> Dict[Text, Any]:
        return {
            "text": text,
            "culture": self.component_config.get("culture"),
            "entities": self.component_config.get("entities"),
            "units": self.component_config.get("units"),
            "showNumbers": self.component_config.get("show_numbers"),
            "mergeResults": self.component_config.get("merge_results"),
        }

    def _recognizers_parse(self, text: Text) -> List[Dict[Text, Any]]:
        """Sends the request to recognizers-service and parses the result.

        Args:
            text: Text for recognizers-service server to parse.
            reference_time: Reference time in milliseconds.

        Returns:
            JSON response from recognizers-service with parse data.
        """
        try:
            payload = self._payload(text)
            headers = {
                "Content-Type": "application/json"
            }
            response = requests.post(
                self._url(),
                json=payload,
                headers=headers,
                timeout=self.component_config.get("timeout"),
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(
                    f"Failed to get a proper response from remote "
                    f"recognizers-service at '{parse_url}. Status Code: {response.status_code}. Response: {response.text}"
                )
                return []
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.ReadTimeout,
        ) as e:
            logger.error(
                "Failed to connect to recognizers-service. Make sure "
                "the recognizers-service is running/healthy/not stale and the proper host "
                "and port are set in the configuration. More "
                "information on how to run the server can be found on "
                "github: "
                "https://github.com/xanthous-tech/recognizers-service "
                "Error: {}".format(e)
            )
            return []

    def process(self, message: Message, **kwargs: Any) -> None:

        if self._url() is not None:
            matches = self._recognizers_parse(message.get(TEXT))
            all_extracted = convert_recognizers_format_to_rasa(matches)
            entities = self.component_config["entities"]
            extracted = RecognizersServiceEntityExtractor.filter_irrelevant_entities(
                all_extracted, entities
            )
        else:
            extracted = []
            rasa.shared.utils.io.raise_warning(
                "recognizers-service component in pipeline, but no "
                "`url` configuration in the config "
                "file nor is `RECOGNIZERS_SERVICE_URL` "
                "set as an environment variable. No entities will be extracted!",
                docs="https://github.com/xanthous-tech/recognizers-service",
            )

        extracted = self.add_extractor_name(extracted)
        message.set(ENTITIES, message.get(ENTITIES, []) + extracted, add_to_output=True)

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Text = None,
        model_metadata: Optional[Metadata] = None,
        cached_component: Optional["RecognizersServiceEntityExtractor"] = None,
        **kwargs: Any,
    ) -> "RecognizersServiceEntityExtractor":
        return cls(meta)
