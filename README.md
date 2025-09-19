# Dive-into-DL-PyTorch 学习笔记

## 📚 项目简介
本仓库记录了我在学习 **《动手学深度学习 (Dive into Deep Learning)》** 过程中，  
在 [ShusenTang/Dive-into-DL-PyTorch](https://github.com/ShusenTang/Dive-into-DL-PyTorch)  
开源实现的基础上，**添加个人理解、注释与实验** 的学习笔记与代码。

目标：
- 熟悉 **PyTorch** 框架下的深度学习基础与经典模型实现  
- 通过逐行注释和实验，巩固书中理论与实践

---

## 🏗 仓库结构

LearningNote_Dive-into-DL-PyTorch/  
├─ d2lzh_pytorch/  
│ ├─ init.py  
│ └─ utils.py  
│  
├─ data/  
│ ├─ img/  
│ ├─ Kaggle_House/  
│ ├─ airfoil_self_noise.dat  
│ ├─ house_tiny.csv  
│ └─ jaychou_lyrics.txt.zip  
│  
├─ src/ # 各章节代码与学习笔记  
│ ├─ Chapter2/  
│ ├─ Chapter3/  
│ ├─ Chapter4/  
│ ├─ Chapter5/  
│ ├─ Chapter6/  
│ ├─ Chapter7/  
│ ├─ Chapter8/  
│ ├─ Chapter9/  
│ └─ Chapter10/  
│  
├─ requirements.txt # 依赖列表  
├─ LICENSE  
└─ README.md  


---

## 🔗 上游项目与参考资料
- **原始书籍**  
  Zhang, Aston; Lipton, Zachary C.; Mu, Li; Smola, Alexander J.  
  *Dive into Deep Learning*  [http://www.d2l.ai](http://www.d2l.ai)

- **PyTorch 版本实现**  
  [ShusenTang/Dive-into-DL-PyTorch](https://github.com/ShusenTang/Dive-into-DL-PyTorch)

---

## ⚙️ 环境要求
- Python ≥ 3.9  
- torch==2.7.0+cu128
- torchvision==0.22.0+cu128
- torchaudio==2.7.0+cu128
- matplotlib==3.9.4
- pandas==2.3.2
- tqdm==4.67.1
- 其余依赖见 [`requirements.txt`](./requirements.txt)


- 注：torchtext不兼容该版本torch，所以Chapter10_Class7代码稍有改动

---

## 📝 版权与许可

本仓库代码基于 Apache License 2.0。

其中部分代码引用并修改自
ShusenTang/Dive-into-DL-PyTorch
，
原始 LICENSE 及版权声明已保留。

本项目仅用于个人学习与分享。

---

## 🙋‍♂️ 说明

本项目主要用于个人学习记录。
