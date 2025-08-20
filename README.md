# 🧠 Transformer 中英翻译项目  
> 基于 Tatoeba 语料 + Transformer 架构的中英双向翻译模型

---

## 📁 项目结构
Transformer-EN-ZH/
├─ ipynb_checkpoints/           # Jupyter 缓存
├─ tatoeba_enzh_clean/          # 清洗后的英中平行语料
├─ best_transformer_model.pt    # 109 MB 最佳模型权重
├─ bpe.json                     # 32 K 共享子词词表
├─ corpus_for_bpe.txt           # 用于训练 BPE 的语料
├─ links.csv                    # Tatoeba 官方 links 文件
├─ sentences.csv                # Tatoeba 官方 sentences 文件
├─ loss_history.csv             # 训练 loss 日志
├─ transformer.ipynb            # 训练 & 推理 Notebook
└─ transformer_loss_curve.png   # Loss 曲线可视化

---

## 🚀 快速开始
| 步骤 | 命令 |
|------|------|
| 1. 克隆 | `git clone <repo>` |
| 2. 安装依赖 | `pip install torch sentencepiece pandas matplotlib` |
| 3. 推理脚本 | 见下方「示例代码」 |

---

## 🎯 模型指标
| 方向 | BLEU | 备注 |
|------|------|------|
| zh → en | **33.7** | Beam=5 |
| en → zh | **35.1** | Beam=5 |

---

## 📊 训练曲线
![loss](main./transformer_loss_curve.png)

---

## 🛠 使用示例
```python
import torch, sentencepiece as spm
from transformer import Transformer  # 自定义模型文件

# 1. 加载权重
ckpt = torch.load('best_transformer_model.pt', map_location='cpu')
model = Transformer(**ckpt['hparams']).eval()
model.load_state_dict(ckpt['state_dict'])

# 2. 加载 tokenizer
sp = spm.SentencePieceProcessor(model_file='bpe.model')

# 3. 翻译函数
def translate(sentence: str):
    ids = [sp.bos_id()] + sp.encode(sentence, out_type=int) + [sp.eos_id()]
    src = torch.tensor(ids).unsqueeze(0)
    with torch.no_grad():
        tgt = greedy_decode(model, src, max_len=256)
    return sp.decode(tgt[0].tolist())


print(translate("The weather is nice today"))
# -> "今天天气真好"
```

---

| 文件                        | 来源           | 链接                                                       |
| ------------------------- | ------------ | -------------------------------------------------------- |
| sentences.csv & links.csv | Tatoeba 官方   | [下载](https://downloads.tatoeba.org/exports/)             |
| 清洗后平行语料                   | Hugging Face | [dataset](https://huggingface.co/datasets/tatoeba/en-zh) |



## 🔗 参考
- [Tatoeba 语料库](https://tatoeba.org/)
- [Transformer 论文](https://arxiv.org/abs/1706.03762)

---

## 📝 许可证
MIT License



