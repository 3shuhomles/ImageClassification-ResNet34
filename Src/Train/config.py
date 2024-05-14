#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/5/15 0:17
# @Author  : name
# @File    : config.py

import pathlib


ProjectPath = pathlib.Path.cwd().parent.parent
DatasetPath = pathlib.Path(str(ProjectPath) + "\\dataset")
TrainPDataPath = pathlib.Path(str(DatasetPath) + "\\train")
TestDataPath = pathlib.Path(str(DatasetPath) + "\\test")
ValidationDataPath = pathlib.Path(str(DatasetPath) + "\\validation")

if __name__ == "__main__":
    print(f"ProjectPathL:{pathlib.Path.cwd().parent.parent}")
    print(f"DatasetPath:{DatasetPath}")
    print(f"TrainPDataPath:{TrainPDataPath}")
    print(f"TestDataPath:{TestDataPath}")
    print(f"ValidationDataPath:{ValidationDataPath}")

