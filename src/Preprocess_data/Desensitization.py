from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import re
import regular_expression
import random




# 从本地加载实体模型
model_path = "C:/Users/zrh/.cache/huggingface/hub/models--dslim--bert-base-NER/snapshots/7a1d333eb0aadffc59fd1e4f56bfedf56b5028e4"
tokenizer_path = "C:/Users/zrh/.cache/huggingface/hub/models--dslim--bert-base-NER/snapshots/7a1d333eb0aadffc59fd1e4f56bfedf56b5028e4"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = "My name is Wolfgan1 ,My name is Wolfgang,My name is Wolfgang  and I live in Berlin,My organization is Volunteer Service Center,我的电话号码是：123-4567-8901，另一个电话是：987-6543-2109,我的邮箱是：example@email.com，另一个邮箱是：test@example.com,我的身份证号码是：12345619800101123X，另一个是：654321199012123456。"

ner_results = nlp(example)


print("原始数据")
print(example)
name = []

for result in ner_results:
    if result['entity'] == 'B-PER':  # 仅替换 'B-PER' 类型的实体文本
        name.append(result['word'])

name = list(set(name))

for i in range(len(name)):
    prefix = '<name-%s>'
    example = example.replace(name[i], prefix % str(i + 1))


name = []

for result in ner_results:
    if result['entity'] == 'B-LOC':  # 仅替换 'B-PER' 类型的实体文本
        name.append(result['word'])

name = list(set(name))

for i in range(len(name)):
    prefix = '<location-%s>'
    example = example.replace(name[i], prefix % str(i + 1))




name = []

for result in ner_results:
    if result['entity'] == 'B-ORG':  # 仅替换 'B-PER' 类型的实体文本
        name.append(result['word'])

name = list(set(name))

for i in range(len(name)):
    prefix = '<organization-%s>'
    example = example.replace(name[i], prefix % str(i + 1))

for result in ner_results:
    if result['entity'] == 'I-PER':  # 删除掉人名实体的继续部分
        entity_text = result['word']
        replacement = ""  # 以空格代替
        example = re.sub(re.escape(entity_text), replacement, example, count=1)

for result in ner_results:
    if result['entity'] == 'I-LOC':  # 删除掉地点的继续部分
        entity_text = result['word']
        replacement = ""  # 以空格代替
        example = re.sub(re.escape(entity_text), replacement, example, count=1)

for result in ner_results:
    if result['entity'] == 'I-ORG':  # 删除掉组织的继续部分
        entity_text = result['word']
        replacement = ""
        example = re.sub(re.escape(entity_text), replacement, example, count=1)



example = regular_expression.hide_phone_numbers(example)
example = regular_expression.hide_email_addresses(example)
example = regular_expression.hide_id_numbers(example)
example = regular_expression.hide_card_numbers(example)
example = regular_expression.hide_ip_addresses(example)
print("大模型识别实体及其得分")
print(ner_results)
print("加密数据 （将实体替换为随机数据，人名，地名，组织名等等分配相应随机数据，对邮箱，电话号码，身份证号等进行加密处理）")
print(example)

