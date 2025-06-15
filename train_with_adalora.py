import torch
import json
torch.cuda.set_device(0)  # 指定GPU设备
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType, AdaLoraConfig

# ---------------------模型初始化--------------------- #
model_path = "./dataroot/models/Qwen/Qwen3-0.6B"
compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text_tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True,
                                              local_files_only=True)

original_model = AutoModelForCausalLM.from_pretrained(model_path,
                                              device_map="auto",
                                              torch_dtype="auto",
                                              local_files_only=True)

print('基础模型加载完成')
# ---------------------模型初始化--------------------- #

# ------------------配置自适应微调参数------------------ #
ada_lora_settings = AdaLoraConfig(target_modules=["q_proj", "v_proj"], init_r=12)
adapted_model = get_peft_model(original_model, ada_lora_settings)
adapted_model.print_trainable_parameters()  # 可训练参数约4.2M

# ------------------配置自适应微调参数------------------ #

# ---------------------推理函数定义--------------------- #
def get_model_output(query, model):
    model_inputs = text_tokenizer(query, return_tensors="pt", max_length=1024, truncation=True).to(compute_device)

    with torch.no_grad():
        model_outputs = model.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=256,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=text_tokenizer.eos_token_id
        )

    model_response = text_tokenizer.decode(model_outputs[0][len(model_inputs.input_ids[0]):], skip_special_tokens=True)
    return model_response.split("\n")[0].strip()
# ---------------------推理函数定义--------------------- #

# ---------------------数据预处理---------------------- #
from datasets import load_dataset
# 加载训练数据
training_set = load_dataset("json", data_files=r"D:\HydroLMM_PEFT\data\racism\annotated_train.json")["train"]
# 加载测试数据
evaluation_set = load_dataset("json", data_files=r"D:\HydroLMM_PEFT\data\racism\annotated_test1.json")["train"]
print('样本数量', len(training_set))

def preprocess_data(sample):
    MAX_SEQ_LEN = 1024
    # 处理指令文本
    instruction_tokens = text_tokenizer(sample["instruction"] + "\n\n我需要你进行类似上述处理的句子是:" +
        sample["content"] + "\n\n")
    # 处理输出文本
    response_tokens = text_tokenizer(sample["output"] + text_tokenizer.eos_token)
    # 合并输入输出
    combined_ids = instruction_tokens["input_ids"] + response_tokens["input_ids"]
    combined_mask = instruction_tokens["attention_mask"] + response_tokens["attention_mask"]
    # 设置标签（仅计算响应部分的loss）
    label_ids = [-100] * len(instruction_tokens["input_ids"]) + response_tokens["input_ids"]
    # 序列截断
    if len(combined_ids) > MAX_SEQ_LEN:
        combined_ids = combined_ids[:MAX_SEQ_LEN]
        combined_mask = combined_mask[:MAX_SEQ_LEN]
        label_ids = label_ids[:MAX_SEQ_LEN]
    return {
        "input_ids": combined_ids,
        "attention_mask": combined_mask,
        "labels": label_ids
    }

training_set = training_set.map(preprocess_data, remove_columns=training_set.column_names)
# 准备评估数据
evaluation_set = evaluation_set.select(range(10, 11)).map(
    lambda sample: {
        "query": sample["instruction"] + "\n\n我需要你进行类似上述处理的句子是:" +
        sample["content"] + "\n\n",
        "target": sample["id"]
    },
    remove_columns=evaluation_set.column_names
)
print('数据处理完成', f'数据结构:{training_set}')
# 显示样本示例
print(training_set[:2])
# 验证预处理结果
print(text_tokenizer.decode(training_set[1]["input_ids"]))
print(text_tokenizer.decode(list(filter(lambda x: x != -100, training_set[1]["labels"]))))

# ---------------------数据预处理---------------------- #

# -------------------基准模型评估---------------------- #
baseline_results = []
for sample in evaluation_set:
    user_query = sample["query"]
    true_label = sample["target"]

    model_output = get_model_output(user_query, original_model)

    baseline_results.append({
        "input": user_query.split("Assistant:")[0].strip(),
        "output": model_output,
        "true_label": true_label
    })

# 输出前5个评估结果
for idx, result in enumerate(baseline_results[:5]):
    print(f"### 测试样本 {idx + 1}")
    print(f"[输入内容]\n{result['input']}")
    print(f"[模型输出]\n{result['output']}")
    print(f"[真实标签]\n{result['true_label']}")
    print("\n" + "=" * 50 + "\n")

with open("baseline_performance.json", "w", encoding="utf-8") as f:
    json.dump(baseline_results, f, ensure_ascii=False, indent=2)
# -------------------基准模型评估---------------------- #

# -------------------模型训练流程---------------------- #
from transformers import DefaultDataCollator, TrainerCallback, TrainingArguments, Trainer

data_collator = DefaultDataCollator()

training_params = TrainingArguments(
    output_dir="qwen_qa_model",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    save_strategy="epoch",
    learning_rate=5e-4,
    fp16=True,
    logging_dir='./training_logs',
    logging_steps=50,
    dataloader_pin_memory=False,
)

model_trainer = Trainer(
    model=adapted_model,
    args=training_params,
    train_dataset=training_set,
    data_collator=DataCollatorForSeq2Seq(tokenizer=text_tokenizer, padding=True),
)

model_trainer.train()

print('模型训练完成')

# 保存微调后的模型
import os
save_path = 'Qwen3-0.6B-adLoRA-Racism'
os.makedirs(save_path, exist_ok=True)

adapted_model.save_pretrained(save_path)
text_tokenizer.save_pretrained(save_path)

# 保存配置文件
model_config = adapted_model.config.to_dict()
config_file = os.path.join(save_path, 'config.json')
with open(config_file, 'w') as f:
    json.dump(model_config, f, ensure_ascii=False, indent=4)

print(f"模型已保存至: {save_path}")
# -------------------模型训练流程---------------------- #

# -------------------模型性能测试---------------------- #
from peft import PeftModel

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
# 合并适配器权重
merged_model = PeftModel.from_pretrained(base_model, save_path).to(compute_device)
merged_model = merged_model.merge_and_unload()

# 评估微调后模型
final_results = []
for sample in evaluation_set:
    user_query = sample["query"]
    true_label = sample["target"]

    model_output = get_model_output(user_query, merged_model)

    final_results.append({
        "input": user_query.split("Assistant:")[0].strip(),
        "output": model_output,
        "true_label": true_label
    })

# 输出评估结果
for idx, res in enumerate(final_results[:5]):
    print(f"### 评估样本 {idx + 1}")
    print(f"[输入内容]\n{res['input']}")
    print(f"[模型输出]\n{res['output']}")
    print(f"[真实标签]\n{res['true_label']}")
    print("\n" + "=" * 50 + "\n")

with open("fine_tuned_results.json", "w", encoding="utf-8") as f:
    json.dump(final_results, f, ensure_ascii=False, indent=2)
# -------------------模型性能测试---------------------- #