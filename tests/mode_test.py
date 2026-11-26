from pathlib import Path
from transformers import AutoTokenizer,AutoConfig,AutoModelForCausalLM
model_ids="Qwen/Qwen3-0.6B-Base"
model_id=Path(__file__).parent.parent/'src/disttrain/model'
print(model_id)

config = AutoConfig.from_pretrained(model_id/'config.json')
tokenizer=AutoTokenizer.from_pretrained(model_id)

# model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

inputs=tokenizer(['who are you','what is it'],return_tensors="pt")
print(inputs.input_ids)

from transformers import pipeline
generator=pipeline(
    'text-generation',
    model=str(model_id),
)
output = generator(
    ["Python 是一门",'你是谁'],
    max_new_tokens=100,
    do_sample=True,
    temperature=0.9,
    top_p=0.9,
    return_full_text=False,
)
for text in output:
    for t in text:
        print(t.get("generated_text"))
# outputs = model.generate(
#     **inputs,
#     max_new_tokens=100,
#     eos_token_id=151645,          # Qwen 的 <|im_end|>，强制停止
#     repetition_penalty=1.15,       # 增加重复惩罚，抑制循环
#     no_repeat_ngram_size=6,        # 禁止 6-gram 重复，防止“卡带”
#     pad_token_id=tokenizer.eos_token_id
# )
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))