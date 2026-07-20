<div align="right">
  <b>🇬🇧 English</b> | <a href="./README.md">🇨🇳 中文</a>
</div>

# 🪿 Mini-RWKV-V7-LM 
🚀 Let's train our own Mini-RWKV-7 from scratch~ A little goose can fly high too~

🤗 [Pure RWKV V7 HuggingFace Space~~~😜 Have a Try?](https://huggingface.co/spaces/Alic-Li/Mini_RWKV_7_34.2M)

🤗 [RWKV V7 with Deep Embedding HuggingFace Space~~~😜 Have a Try?](https://huggingface.co/spaces/Alic-Li/MiniRWKV_7_DE_34.2M)

🤗 [My trained model weights are now open-sourced~](https://huggingface.co/Alic-Li/Mini_RWKV_7_34.2M)
<div align="center">
  <img src="./miniGoose.png" width="200" height="200" style="display: block; margin: auto;">
</div>

## 🌟 Model Introduction
This model is a **34M parameter** language model `Mini-RWKV-V7-LM-34M` based on the **RWKV-V7 architecture**. While keeping it lightweight, it has good language understanding and generation capabilities, making it ideal for deployment on resource-constrained devices and rapid iterative development.

**💫 Now supports training**
- **RWKV V7 architecture (Pure RWKV V7)**
- **RWKV V7a DE architecture (RWKV V7 with Deep Embedding)**
- **RWKV V7b DEA architecture (RWKV V7 with Deep Embedding Attention)**

Simply modify the bash scripts `demo-training-prepare.sh` | `demo-training-run.sh` | `demo-training-run-sft.sh`:

```bash
MODEL_TYPE="x070"
```
to adjust the model architecture:
```
# Note: "x070" is Pure RWKV-V7
#       "x070#DE" is RWKV-V7 with Deep Embedding
#       "x070#DE#DEA" is RWKV-V7 with Deep Embedding Attention
```

---

## 📦 Model Architecture 
### Pure RWKV V7

| Parameter | Value |
|-----------|-------|
| Parameters | 34.2M 🎯 |
| Actual activated parameters during inference | ~34.2M 🚀 |
| Layers | 8 🧱 |
| Hidden dimension | 512 📐 |
| Context length | 512->1024->2048 📏 |
| Vocabulary size | 6400 📚 |
- Vocabulary is consistent with MiniMind
```
INFO:pytorch_lightning.callbacks.model_summary:
  | Name   | Type       | Params | Mode  | FLOPs
------------------------------------------------------
0 | emb    | Embedding  | 3.3 M  | train | 0    
1 | blocks | ModuleList | 27.6 M | train | 0    
2 | ln_out | LayerNorm  | 1.0 K  | train | 0    
3 | head   | Linear     | 3.3 M  | train | 0    
------------------------------------------------------
34.2 M    Trainable params
0         Non-trainable params
34.2 M    Total params
```
### RWKV V7 with Deep Embedding

| Parameter | Value |
|-----------|-------|
| Parameters | 34.2M + 57.3M (Free Params) 🎯 |
| Actual activated parameters during inference | ~34.2M 🚀 |
| Layers | 8 🧱 |
| Hidden dimension | 512 📐 |
| Context length | 512->1024->2048 📏 |
| Vocabulary size | 6400 📚 |
- Vocabulary is consistent with MiniMind
```
INFO:pytorch_lightning.callbacks.model_summary:
  | Name   | Type       | Params | Mode  | FLOPs
------------------------------------------------------
0 | emb    | Embedding  | 3.3 M  | train | 0    
1 | blocks | ModuleList | 84.9 M | train | 0    
2 | ln_out | LayerNorm  | 1.0 K  | train | 0    
3 | head   | Linear     | 3.3 M  | train | 0    
------------------------------------------------------
91.5 M    Trainable params
0         Non-trainable params
91.5 M    Total params
```
### RWKV V7 with Deep Embedding Attention

| Parameter | Value |
|-----------|-------|
| Parameters | 34.2M + 57.3M (Free Params) + 43.5M (Self-Attention) 🎯 |
| Actual activated parameters during inference | ~34.2M + 43.5M (Self-Attention) 🚀 |
| Layers | 8 🧱 |
| Hidden dimension | 512 📐 |
| Context length | 512->1024->2048 📏 |
| Vocabulary size | 6400 📚 |
- Vocabulary is consistent with MiniMind
```
INFO:pytorch_lightning.callbacks.model_summary:
  | Name   | Type       | Params | Mode  | FLOPs
------------------------------------------------------
0 | emb    | Embedding  | 3.3 M  | train | 0    
1 | blocks | ModuleList | 128 M  | train | 0    
2 | ln_out | LayerNorm  | 1.0 K  | train | 0    
3 | head   | Linear     | 3.3 M  | train | 0    
------------------------------------------------------
135 M     Trainable params
0         Non-trainable params
135 M     Total params
```
---

## 🧪 Training Info

- 🪿 Architecture: [RWKV-V7](https://github.com/BlinkDL/RWKV-LM) 
- 📚 Data source: [minimind_dataset](https://huggingface.co/datasets/jingyaogong/minimind_dataset) Special thanks to MiniMind's author [@jingyaogong](https://github.com/jingyaogong) for open-sourcing the training dataset 🤗
- 📈 Learning rate: Dynamic adjustment  
- 🖥️ Hardware: Can be trained with GPUs like 4060 laptop, or even Radeon 780M integrated graphics on thin-and-light laptops 😜
- 👀 I quickly reproduced this on AMD Instinct MI300X (many thanks to AMD for sponsoring cloud computing resources for me personally and for RWKV) 😊
- 📦 Model size: 68.4MB, 34.2M params
- 📊 Pre-training loss curve: Stable convergence, loss fluctuates around 2.12 (due to relatively small pre-training data volume) 
- 📊 SFT training loss curve: Final loss fluctuates around 0.5  

---
## 🎉 Showcase
![Mars travel novel](./img/img_1.png)
![Bogosort code](./img/img_2.png)
![Psychological Q&A](./img/img_3.png)
---

## 📚 Supported Tasks

- 📝 Pre-training
- 📚 Supervised fine-tuning (SFT)
---

## 🧰 Inference

### 🐍 Install Dependencies

```bash
pip install -r requirements.txt
```
- If you are using AMD GPUs like the AMD Instinct MI300X, do NOT install the torch from requirements.txt; instead, install the corresponding latest version of torch.
- For example: ```pip3 install torch --index-url https://download.pytorch.org/whl/rocm6.3```
- Refer to the [PyTorch official download page](https://pytorch.org/get-started/locally/) for specific installation instructions.

### 🧪 Load Model & Inference Example

```bash
python3 ./API_DEMO_CHAT.py
```

# 🚀 Start Training
## 🗒️ Prepare the Dataset

### 📥 Download Dataset
```bash
modelscope download --dataset gongjy/minimind_dataset sft_512.jsonl sft_1024.jsonl sft_2048.jsonl pretrain_hq.jsonl sft_mini_512.jsonl --local_dir ./minimind_dataset
```
### 🛠️ Pre-training Binidx Dataset Conversion
```bash
python3 ./data/make_data.py pretrain_hq.jsonl 1 512
```
- Here `1` means repeating the dataset once, `512` is the context length.

## 📦 Stage 1: Model Weight Initialization
```bash
sh ./demo-training-prepare.sh
```
- After execution, an `./out/L8-D512-x070` folder will be generated containing the initialized model weight file named `rwkv-init.pth`.

## 📚 Stage 2: Pre-training
```bash
sh ./demo-training-run.sh
```
- Modify `M_BSZ="16"` on line 28 to adjust the batch size.
- Modify `--wandb ""` on line 52 to enable wandb logging of the training process.
- After training completes, a final weight file named `rwkv-final.pth` will be generated in the `./out/L8-D512-x070` folder.

## 🎯 Stage 3: Supervised Fine-tuning (SFT) _ctx_512
- Before training, delete the `rwkv-init.pth` initial weight file in the `./out/L8-D512-x070` folder.
- Then rename the `rwkv-final.pth` weight file in the `./out/L8-D512-x070` folder to `rwkv-init.pth` as the initial weight file for post-training.
- Modify the corresponding configuration parameters according to the dataset info in the Dataset Token Count Reference Table:
```bash
CTX_LEN="512"
MY_EXIT_TOKENS="1843253579"
MAGIC_PRIME="3600053"
DATA_FILE=[your data file path]
```
```bash
sh ./demo-training-run-sft.sh
```

## 🎯 Stage 4: Supervised Fine-tuning (SFT) _ctx_1024
- Before training, delete the `rwkv-init.pth` initial weight file in the `./out/L8-D512-x070` folder.
- Then rename the `rwkv-final.pth` weight file in the `./out/L8-D512-x070` folder to `rwkv-init.pth` as the initial weight file for post-training.
- Modify the corresponding configuration parameters according to the dataset info in the Dataset Token Count Reference Table:
```bash
CTX_LEN="1024"
MY_EXIT_TOKENS="1430197426"
MAGIC_PRIME="1396673"
DATA_FILE=[your data file path]
```
```bash
sh ./demo-training-run-sft.sh
```

## 🎯 Stage 5: Supervised Fine-tuning (SFT) _ctx_2048
- Before training, delete the `rwkv-init.pth` initial weight file in the `./out/L8-D512-x070` folder.
- Then rename the `rwkv-final.pth` weight file in the `./out/L8-D512-x070` folder to `rwkv-init.pth` as the initial weight file for post-training.
- Modify the corresponding configuration parameters according to the dataset info in the Dataset Token Count Reference Table:
```bash
CTX_LEN="2048"
MY_EXIT_TOKENS="2398644915"
MAGIC_PRIME="1171199"
DATA_FILE=[your data file path]
```
```bash
sh ./demo-training-run-sft.sh
```

## 🪿 Learning Rate (LR) Suggestions
### Pre-training 
- LR_INIT=```6e-4``` LR_FINAL=```2e-5```
### Supervised Fine-tuning (SFT) 
- LR_INIT=```2e-5``` LR_FINAL=```1e-6```

## 💫 Dataset Token Count Reference Table
Total tokens in sft_512.jsonl: 
- --my_exit_tokens ```1843253579``` --magic_prime ```3600053``` --ctx_len ```512```

Total tokens in sft_1024.jsonl: 
- --my_exit_tokens ```1430197426``` --magic_prime ```1396673``` --ctx_len ```1024```

Total tokens in sft_2048.jsonl: 
- --my_exit_tokens ```2398644915``` --magic_prime ```1171199``` --ctx_len ```2048```

## 📢 Acknowledgments

- 🖥️ Thanks to AMD for sponsoring cloud computing resources for me personally and for RWKV.
- 🙌 Thanks to the RWKV community for providing open-source code and training frameworks!
- 🚀 Thanks to [MiniMind](https://github.com/jingyaogong/minimind) for inspiration!
- If you find any bugs or have suggestions, feel free to submit an issue or PR 🛠️

---

## 🧩 Related Projects

- [MiniMind](https://github.com/jingyaogong/minimind): A lightweight LLM teaching project 📚
- [RWKV_v7_G1_Translate](https://huggingface.co/Alic-Li/RWKV_v7_G1_Translate_ctx4096_20250620): An ultra-lightweight edge Chinese-English translation model based on full fine-tuning of RWKV-V7-G1 💫 
- [RWKV-LM](https://github.com/BlinkDL/RWKV-LM): RWKV official architecture repository 🧠

---

🎉 Thank you for using **Mini_RWKV_7**! If you like this project, feel free to share it with everyone! 🌟

--- 
