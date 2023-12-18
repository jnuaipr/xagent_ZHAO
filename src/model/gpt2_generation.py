# gpt2_generation.py
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
# chatgpt2的蒸馏版本
def generate_text(prompt):
    # 配置GPU设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 初始化模型和分词器
    tokenizer = GPT2Tokenizer.from_pretrained('C:/Users/zrh/.cache/huggingface/hub/models--distilgpt2\snapshots/38cc92ec43315abd5136313225e95acc5986876c')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = GPT2LMHeadModel.from_pretrained('C:/Users/zrh/.cache/huggingface/hub/models--FredZhang7--distilgpt2-stable-diffusion-v2/snapshots/f839bc9217d4bc3694e4c5285934b5e671012f85').to(device)

    temperature = 0.9
    top_k = 8
    max_length = 80
    repetition_penalty = 1.2
    num_return_sequences = 5

    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)

    # 在GPU上运行生成
    output = model.generate(
        input_ids,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        repetition_penalty=repetition_penalty,
        penalty_alpha=0.6,
        no_repeat_ngram_size=1,
        early_stopping=True
    )

    results = []
    for i in range(len(output)):
        results.append(tokenizer.decode(output[i], skip_special_tokens=True))

    return results

'''

text = "Your input text here"
generated_texts = generate_text(text)
print(generated_texts)
'''
