# Rasa NLU Components using PaddleNLP

[![](https://img.shields.io/pypi/v/rasa_paddlenlp.svg)](https://pypi.python.org/pypi/rasa_paddlenlp)


## Features

- Tokenizer and Dense featurizer using pre-trained models supported by PaddleNLP.

## Usage

```shell
pip install rasa-paddlenlp
```

In your config.yml, use the following configuration:
bigbrother update:
根据官方文档建议做如下优化，可以明显提高表现（基于data下面的测试数据实测）

```yaml
language: zh

pipeline:
  - name: "rasa_paddlenlp.nlu.paddlenlp_tokenizer.PaddleNLPTokenizer"
    model_name: bert
    model_weights: bert-wwm-ext-chinese
    # Flag to check whether to split intents
    # 下面这两项开启True后，可以进行多意图识别
    # https://rasa.com/docs/rasa/components#:~:text=with%20any%20tokenizer%3A-,intent_tokenization_flag,-indicates%20whether%20to
    intent_tokenization_flag: false
    # Symbol on which intent should be split
    intent_split_symbol: "_"
  - name: "rasa_paddlenlp.nlu.paddlenlp_featurizer.PaddleNLPFeaturizer"
    model_name: bert
    model_weights: bert-wwm-ext-chinese
    # 建议下面这一项设为False（默认是True），不然中文实体识别会有很大问题
    # https://rasa.com/docs/rasa/components#:~:text=the%20DIETClassifier%20components!-,Configuration,-Make%20the%20featurizer
    use_word_boundaries: False
  - name: "LexicalSyntacticFeaturizer"
  - name: "CountVectorsFeaturizer"
  - name: "CountVectorsFeaturizer"
    analyzer: "char_wb"
    min_ngram: 1
    max_ngram: 7
  - name: "DIETClassifier"
    epochs: 100
    #开启下面三项，会在训练时每20轮做一次测试，挑选最优模型，避免过拟合，但eonoe需要至少10个，所以适用于≥15个example的情况，不然应该是有反作用的
    evaluate_every_number_of_epochs: 20
    evaluate_on_number_of_examples: 10
    checkpoint_model: True
  - name: "EntitySynonymMapper"
  # bigbrother add the fallbackclassifier 如果没有意图置信度超过0.7，或者置信度最高的两个意图相差小于0.1都会返回 nlu_fallback
  - name: "FallbackClassifier"
    threshold: 0.7
    ambiguity_threshold: 0.1
```
bigbrother：完整的config.yml文件和用于测试的data,可以参考repo：https://github.com/bigbrother666sh/rasa-paddlenlp-ernie 的 paddle-ernie分支

## Credits

This package took inspiration from the following projects:

- [Rasa](https://github.com/rasahq/rasa)
- [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)

This package was created with Cookiecutter and the `audreyr/cookiecutter-pypackage` project template.

- [Cookiecutter](https://github.com/audreyr/cookiecutter)
- [`audreyr/cookiecutter-pypackage`](https://github.com/audreyr/cookiecutter-pypackage)


## License

[MIT](./LICENSE)
