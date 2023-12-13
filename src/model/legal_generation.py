from transformers import pipeline, set_seed

def generate_text(text):
    import torch
    if torch.cuda.is_available():
        device = 0
    else:
        device = -1

    set_seed(42)

    # 修改这里来从本地加载模型
    model_path = 'C:/Users/zrh/.cache/huggingface/hub/models--umarbutler--open-australian-legal-distilgpt2/snapshots/06071d63a090272147154ebaf04f61df471cdbcf'  # 你的本地模型路径
    generator = pipeline('text-generation', model=model_path, device=device)

    generated_text = generator(text, max_length=50, num_return_sequences=5)
    generated_text = [result['generated_text'] for result in generated_text]

    return generated_text

'''
text = "Your input text here"
generated_texts = generate_text(text)
print(generated_texts)
'''
