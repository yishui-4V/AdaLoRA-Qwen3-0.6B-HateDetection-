import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM

# 设备配置
torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# 加载评估数据集
from datasets import load_dataset

eval_dataset = load_dataset("json", data_files=r"D:\HydroLMM_PEFT\data\racism\annotated_test1.json")["train"]

# 模型路径配置
lora_model_dir = r'Qwen3-0.6B-adLoRA-Racism'
base_model_path = "./dataroot/models/Qwen/Qwen3-0.6B"

nlp_tokenizer = AutoTokenizer.from_pretrained(
    base_model_path,
    trust_remote_code=True,
    local_files_only=True
)

from peft import PeftModel

# 加载基础模型
foundation_model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto")
# 整合适配器权重
finetuned_model = PeftModel.from_pretrained(foundation_model, lora_model_dir).to(torch_device)
finetuned_model = finetuned_model.merge_and_unload()


# 生成模型响应
def get_model_response(prompt, model):
    model_inputs = nlp_tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(torch_device)

    with torch.no_grad():
        model_outputs = model.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=256,
            do_sample=True,
            top_p=0.9,
            temperature=0.2,
            pad_token_id=nlp_tokenizer.eos_token_id
        )

    model_response = nlp_tokenizer.decode(model_outputs[0][len(model_inputs.input_ids[0]):], skip_special_tokens=True)
    return model_response.split("\n")[0].strip()


# 执行模型评估
eval_results = []
for idx, sample in enumerate(eval_dataset):
    user_input = sample["instruction"] + "\n\n我需要你进行类似上述处理的句子是:" + sample["content"] + "\n\n"
    print(f'输入 [{idx + 1}]:', user_input)

    target_value = sample['id']
    print('真实值:', target_value)

    model_output = get_model_response(user_input, finetuned_model)
    print('模型预测:', model_output)

    result_record = {
        "user_input": user_input,
        "model_prediction": model_output,
        "target_value": target_value
    }
    eval_results.append(result_record)

    if idx == 500:
        break

# 保存评估结果
with open("adlora_racism_predictions.json", "w", encoding="utf-8") as f:
    json.dump(eval_results, f, indent=4, ensure_ascii=False)