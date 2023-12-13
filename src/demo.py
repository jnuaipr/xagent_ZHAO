from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

model_path = "C:/Users/zrh/.cache/huggingface/hub/models--dslim--bert-base-NER/snapshots/7a1d333eb0aadffc59fd1e4f56bfedf56b5028e4"
tokenizer_path = "C:/Users/zrh/.cache/huggingface/hub/models--dslim--bert-base-NER/snapshots/7a1d333eb0aadffc59fd1e4f56bfedf56b5028e4"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(tokenizer_path)

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = "My name is Wolfgang and I live in Berlin"

ner_results = nlp(example)
print(ner_results)
