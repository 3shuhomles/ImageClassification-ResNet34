#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/5/15 0:17
# @Author  : name
# @File    : config.py

import pathlib


ProjectPath = pathlib.Path(r"E:\code\githubWorkSpace\ImageClassification-ResNet34")
DatasetPath = pathlib.Path(str(ProjectPath) + "\\dataset")
TrainDataPath = pathlib.Path(str(DatasetPath) + "\\train")
TestDataPath = pathlib.Path(str(DatasetPath) + "\\test")
ValidationDataPath = pathlib.Path(str(DatasetPath) + "\\validation")
CheckpointSaveDir = pathlib.Path(str(ProjectPath) + "\\result" + "\\checkpoint")
FigureSavePath = pathlib.Path(str(ProjectPath) + "\\result" + "\\model_figure")



if __name__ == "__main__":
    print(f"ProjectPathL:{pathlib.Path.cwd().parent.parent}")
    print(f"DatasetPath:{DatasetPath}")
    print(f"TrainDataPath:{TrainDataPath}")
    print(f"TestDataPath:{TestDataPath}")
    print(f"ValidationDataPath:{ValidationDataPath}")
    print(f"CheckpointSaveDir:{CheckpointSaveDir}")
    print(f"FigureSavePath:{FigureSavePath}")

