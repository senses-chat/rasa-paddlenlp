import logging
from typing import Any, Dict, List, Text

from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.shared.nlu.training_data.message import Message
from paddlenlp.transformers import BertTokenizer

logger = logging.getLogger(__name__)


class PaddleNLPTokenizer(Tokenizer):
    """PaddleNLP Transformers-based tokenizer."""

    defaults = {
        "model_name": "bert-wwm-ext-chinese",
    }

    def __init__(self, component_config: Dict[Text, Any] = None) -> None:  # noqa: D107
        super().__init__(component_config)
        self.tokenizer = BertTokenizer.from_pretrained(self.component_config["model_name"])

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
