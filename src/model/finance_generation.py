from transformers import pipeline
import torch

def generate_text(text, model_path):
    # 检查CUDA是否可用
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_path = "C:/Users/zrh/.cache/huggingface/hub/models--lxyuan--distilgpt2-finetuned-finance/snapshots/e185be9dba22a0c041b26293483b55e39461119b"  # 替换为你的模型路径
    # 加载模型
    generator = pipeline("text-generation", model=model_path, device=device.index if device.type == "cuda" else -1)

    # 生成文本
    generated_texts = generator(
        text,
        pad_token_id=generator.tokenizer.eos_token_id,
        max_new_tokens=200,
        num_return_sequences=2
    )

    return generated_texts

'''
# 使用示例

text = "Your input text here"
generated_texts = generate_text(text)
print(generated_texts)
'''
