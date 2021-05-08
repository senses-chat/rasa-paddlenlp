from paddlenlp.transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-wwm-ext-chinese")

print(tokenizer.all_special_tokens)
print(tokenizer.all_special_ids)

text = "你好吗"

output = tokenizer.batch_encode([[text, text]], stride=1, return_token_type_ids=False, return_length=False, return_special_tokens_mask=True)

print(output)

for o in output:
    print(tokenizer.convert_ids_to_tokens(o['input_ids']))
