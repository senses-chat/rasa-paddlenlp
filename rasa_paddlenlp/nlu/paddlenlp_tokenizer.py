from __future__ import annotations
from typing import Any, Dict, List, Optional, Text
import logging

import rasa.shared.utils.io
import rasa.utils.io

from rasa.engine.graph import ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.shared.constants import DOCS_URL_COMPONENTS
from rasa.shared.nlu.training_data.message import Message

logger = logging.getLogger(__name__)

@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_TOKENIZER, is_trainable=False
)
class PaddleNLPTokenizer(Tokenizer):
    """PaddleNLP Transformers-based tokenizer."""

    @staticmethod
    def not_supported_languages() -> Optional[List[Text]]:
        """The languages that are not supported."""
        return []

    @staticmethod
    def required_packages() -> List[Text]:
        """Returns the extra python dependencies required."""
        return ["paddlenlp", "paddle"]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns the component's default config."""
        return {
            # name of the language model to load.
            "model_name": "bert",
            # Pre-Trained weights to be loaded(string)
            "model_weights": "bert-wwm-ext-chinese",
            # Flag to check whether to split intents
            "intent_tokenization_flag": False,
            # Symbol on which intent should be split
            "intent_split_symbol": "_",
            # Regular expression to detect tokens
            "token_pattern": None,
        }

    def __init__(self, config: Dict[Text, Any]) -> None:
        """Initialize the tokenizer."""
        super().__init__(config)

        if "case_sensitive" in self._config:
            rasa.shared.utils.io.raise_warning(
                "The option 'case_sensitive' was moved from the tokenizers to the "
                "featurizers.",
                docs=DOCS_URL_COMPONENTS,
            )

        self._load_model_metadata()
        self._load_model_instance()

    def _load_model_metadata(self) -> None:
        """Load the metadata for the specified model and sets these properties.

        This includes the model name, model weights, cache directory and the
        maximum sequence length the model can handle.
        """
        from .paddlenlp_registry import (
            model_class_dict,
            model_weights_defaults,
        )

        self.model_name = self._config["model_name"]

        if self.model_name not in model_class_dict:
            raise KeyError(
                f"'{self.model_name}' not a valid model name. Choose from "
                f"{str(list(model_class_dict.keys()))} or create"
                f"a new class inheriting from this class to support your model."
            )

        self.model_weights = self._config["model_weights"]

        if not self.model_weights:
            logger.info(
                f"Model weights not specified. Will choose default model "
                f"weights: {model_weights_defaults[self.model_name]}"
            )
            self.model_weights = model_weights_defaults[self.model_name]

    def _load_model_instance(self) -> None:
        """Try loading the model instance.
        """

        from .paddlenlp_registry import (
            model_tokenizer_dict,
        )

        logger.debug(f"Loading Tokenizer for {self.model_name}")

        self.tokenizer = model_tokenizer_dict[self.model_name].from_pretrained(self.model_weights)

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> PaddleNLPTokenizer:
        """Creates a new component (see parent class for full docstring)."""
        # Path to the dictionaries on the local filesystem.
        return cls(config)


    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        text = message.get(attribute)

        # HACK: have to do this to get offset value
        encoded_inputs = self.tokenizer.batch_encode([[text, text]], stride=1, return_token_type_ids=False, return_length=False, return_special_tokens_mask=True)

        tokens = []

        e = encoded_inputs[0]
        tokenized = self.tokenizer.convert_ids_to_tokens(e['input_ids'])
        for i in range(1, len(tokenized)):
            if e['special_tokens_mask'][i] == 1:
                break
            token = Token(tokenized[i], e['offset_mapping'][i][0], e['offset_mapping'][i][1])
            tokens.append(token)

        return self._apply_token_pattern(tokens)
