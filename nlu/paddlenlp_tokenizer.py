import logging
from typing import Any, Dict, List, Text

from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.shared.nlu.training_data.message import Message
from paddlenlp.transformers import BertTokenizer

logger = logging.getLogger(__name__)


class PaddleNLPTokenizer(Tokenizer):
    """PaddleNLP Transformers-based tokenizer."""

    defaults = {
        "model_name": "bert",
        "model_weights": "bert-wwm-ext-chinese",
    }

    def __init__(self, component_config: Dict[Text, Any] = None) -> None:  # noqa: D107
        super().__init__(component_config)
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

        self.model_name = self.component_config["model_name"]

        if self.model_name not in model_class_dict:
            raise KeyError(
                f"'{self.model_name}' not a valid model name. Choose from "
                f"{str(list(model_class_dict.keys()))} or create"
                f"a new class inheriting from this class to support your model."
            )

        self.model_weights = self.component_config["model_weights"]

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
    def required_packages(cls) -> List[Text]:  # noqa: D102
        return ["paddlenlp"]

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:  # noqa: D102
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
