# Dive-into-DL-PyTorch 学习笔记

## 📚 项目简介
本仓库记录了我在学习 **《动手学深度学习 (Dive into Deep Learning)》** 过程中，  
在 [ShusenTang/Dive-into-DL-PyTorch](https://github.com/ShusenTang/Dive-into-DL-PyTorch)  
开源实现的基础上，**添加个人理解、注释与实验** 的学习笔记与代码。

目标：
- 熟悉 **PyTorch** 框架下的深度学习基础与经典模型实现  
- 通过逐行注释和实验，巩固书中理论与实践
- 新书第九章计算机视觉，作者好多坑，甚至直接numpy而不tensor算起来慢得要死，  
- 于是猛猛修代码😒代码修理和调试能力得到极大提升😑

---

## 🔧 修改
- **两本书籍**  
由于ShusenTang新书中9.6后面有缺失，这里直接参考原书进行学习    

- **两版d2l**   
同时，两本书的d2l库略有不同，所以这里也对d2l库中，9.1-9.13做了修改  
修改问题包括**device**，**amp加速**，以及**MultiboxDetection**编码解码不一致的错误等等等等。。。  
如果第九章出现Error可以翻翻**d2lzh_pytorch/utils.py**  
图片不显示可以勤加d2l.plt.show()，有些方法里面我禁掉了
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
