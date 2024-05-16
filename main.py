
import test
import Src.Train as Train
from Src.Model.ModelPara import LEARNINGRATE_1


if __name__ =="__main__":
    # print(tf.__version__)
    test.ImTest()
    test.LRTest()

    Train.ModelFit(EPOCHES = 2)
    # ModelName = f"ResNet34_30_{LEARNINGRATE_1}"

