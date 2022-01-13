# Rasa NLU Components using PaddleNLP

[![](https://img.shields.io/pypi/v/rasa_paddlenlp.svg)](https://pypi.python.org/pypi/rasa_paddlenlp)


## Features

- Tokenizer and Dense featurizer using pre-trained models supported by PaddleNLP.

## Usage

```shell
pip install rasa-paddlenlp
```

In your config.yml, use the following configuration:

```yaml
language: zh

pipeline:
  - name: "rasa_paddlenlp.nlu.paddlenlp_tokenizer.PaddleNLPTokenizer"
    model_name: bert
    model_weights: bert-wwm-ext-chinese
    # Flag to check whether to split intents
    intent_tokenization_flag: false
    # Symbol on which intent should be split
    intent_split_symbol: "_"
  - name: "rasa_paddlenlp.nlu.paddlenlp_featurizer.PaddleNLPFeaturizer"
    model_name: bert
    model_weights: bert-wwm-ext-chinese
  # rest of your configurations
```

Currently there is code to support BERT pre-trained models, we just need to add the model definitions and default weights in order for other PaddleNLP-supported models.

## Credits

This package took inspiration from the following projects:

- [Rasa](https://github.com/rasahq/rasa)
- [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)

This package was created with Cookiecutter and the `audreyr/cookiecutter-pypackage` project template.

- [Cookiecutter](https://github.com/audreyr/cookiecutter)
- [`audreyr/cookiecutter-pypackage`](https://github.com/audreyr/cookiecutter-pypackage)

## License

[MIT](./LICENSE)
