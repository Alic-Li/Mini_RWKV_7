# 🪿 Mini-RWKV-V7-LM 
🚀 让我们来从头训练一个属于自己的Mini-RWKV-7吧~ 小小的鹅也能飞得很高喔~

<div align="center">
  <img src="./miniGoose.png" width="200" height="200" style="display: block; margin: auto;">
</div>

## 🌟 模型简介

本模型是基于 **RWKV-V7 架构** 训练的一个 **34M 参数量** 的语言模型`Mini-RWKV-V7-LM-34M`。它在保持轻量的同时，具备良好的语言理解和生成能力，非常适合资源极其有限的设备部署和快速迭代开发。

---

## 📦 模型结构

| 参数 | 数值 |
|------|------|
| 参数量 | 34.2M 🎯 |
| 层数 | 8 🧱 |
| 隐藏维度 | 512 📐 |
| 上下文长度 | 512->1024->2048 📏 |
| 词表大小 | 6400 📚 |
- Vocab 和MiniMind的保持一致
---

## 🧪 训练信息

- 🪿 架构：[RWKV-V7](https://github.com/BlinkDL/RWKV-LM) 
- 📚 数据源：[minimind_dataset](https://huggingface.co/datasets/jingyaogong/minimind_dataset) 特别感谢MiniMind的作者 [@jingyaogong](https://github.com/jingyaogong)开源了训练数据集 🤗
- 📈 学习率：动态调整  
- 🖥️ 硬件：可以使用4060laptop等显卡进行训练，甚至Radeon 780M 核显也可以在轻薄本上进行训练 🚀  
- 👀我是在AMD Instinct MI300X 上快速复现的(十分感谢AMD公司的对我个人以及RWKV的云算力赞助)
- 📦 模型大小：68.4MB 参数量 34.2M Params
- 📊 损失曲线：训练收敛稳定 loss < 2.0  

---
## 📚 支持任务

- 📝 预训练（Pre-training）
- 📚 监督微调训练（Supervised fine-tuning [SFT]）
---

## 🧰 推理方法

### 🐍 安装依赖

```bash
pip install -r requirements.txt
```
- 如果你使用的是AMD Instinct MI300X，请不要安装requirements.txt中的torch，请安装对应最新版本的torch
- 比如说```pip3 install torch --index-url https://download.pytorch.org/whl/rocm6.3```


### 🧪 加载模型 & 推理示例

```bash
python3 ./API_DEMO_CHAT.py
```

## 🚀开始训练

## 🪿学习率建议(LR)

## 📢 使用须知


## 📢 致谢

- 🙌 感谢 RWKV 社区提供的开源代码和训练框架！
- 🚀 感谢 [MiniMind](https://github.com/jingyaogong/minimind) 提供的 README 模板灵感！
- 如发现 bug 或有任何建议，欢迎提交 issue 或 PR 🛠️


---

## 🧩 相关项目推荐

- [MiniMind](https://github.com/jingyaogong/minimind)：一个轻量级 LLM 教学项目 📚
- [RWKV-V7](https://github.com/BlinkDL/RWKV-LM)：RWKV 最新版本架构仓库 🧠

---

🎉 感谢你使用 **Mini_RWKV_7**！如果你喜欢这个项目，欢迎点赞、分享、Star 和 Follow！🌟

--- 