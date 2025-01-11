# 减少帧间与专家激活冗余的视频去雪混合特征调制专家模型

在现实场景中部署深度神经网络常常会面对需要为多个大同小异的任务分别配置一个特化模型而造成冗余的问题。本工作首先在Allweather数据集上复现Zhang等人提出的混合特征调制专家模型去完成多合一图像去除天气任务，用一个模型进行图片中多种天气效果（雨、雪、雾等）的去除。进一步地，我们观察到在视频任务中不同帧的类型和退化程度存在差异但连续帧之间也存在着时序冗余，即对不同帧的处理有异有同，因此基于原文方法增加了参考帧机制和动态路由机制，使其能够用于视频去雪任务，并通过复用上一帧结果和动态早退出进一步减少推理开销。我们在RVSD数据集上进行了训练和测试，实验结果表明我们的方法确实能够实现帧种类的感知并依此激活不同的专家，且在几乎不影响性能的同时使激活专家数量减半。

## 环境配置

```
conda create --name <env_name> --file requirements.txt
```

## 数据集

- Allweather: [https://pan.baidu.com/s/1hIeYU_OolKKUBx8N2FUlhA?pwd=lgap](https://pan.baidu.com/s/1hIeYU_OolKKUBx8N2FUlhA?pwd=lgap)
- RVSD: [https://haoyuchen.com/VideoDesnowing#download](https://haoyuchen.com/VideoDesnowing#download)

将`configs/dataset_cfg.py`的`get_dataset_root`中的数据集路径替换为你自己的数据集路径。

## 训练

修改下面这个脚本中的超参数并运行
```
bash scripts_allweather_mofme_ours.sh
```

## 推理

```
bash infer_video.py
```

## 代码版权说明

1. RVSD类，自主实现，用于数据处理和参考帧机制；
2. DynamicTopK类，自主实现，用于动态路由机制；
3. infer_video.py文件，自主实现，用于视频推理和参考帧机制；
4. 其它代码，源自[MoFME-Pytorch](https://github.com/RoyZry98/MoFME-Pytorch/tree/main)，为原文的用于图片任务的混合特征调制专家模型官方开源代码。