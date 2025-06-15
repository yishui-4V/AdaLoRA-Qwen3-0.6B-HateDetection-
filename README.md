# AdaLoRA-Qwen3-0.6B-HateDetection-
本项目基于通义千问Qwen3-0.6B模型，采用**AdaLoRA**(自适应低秩适应)技术进行微调，构建了针对仇恨言论识别的高效分类系统。包含完整的训练数据预处理、模型微调和评估流程，实现了高效且低资源消耗的NLP解决方案。



## 文件结构
```text
.
├── scripts/
│ ├── lora_FT_hate.py # 主训练脚本（含AdaLoRA微调）
│ └── eval_lora_hate.py # 模型评估脚本
└── README.md # 项目说明文档

**说明**：  
1. `lora_FT_hate.py`：用于模型的训练和微调  
2. `eval_lora_hate.py`：用于模型的评估和预测  
3. `README.md`：项目说明文档（本文档）

**运行时生成的目录**（无需预先创建）：
- `saved_models/`：保存训练后的微调模型
- `results/`：保存评估结果和性能指标
- `data/`：数据集存放目录（需手动添加数据文件）
```
## 运行流程
### 1. 环境配置
```bash
pip install -r requirements.txt
```
运行Qwen3模型下载
```python
python model_download.py --repo_id Qwen\Qwen3-0.6B
```
模型微调
```python
python scripts/lora_FT_hate.py
```
模型评估
```python
python scripts/eval_lora_hate.py
```
