
import test
import Src.Train as Train
from Src.Model.ModelPara import LEARNINGRATE_1
from Src import PathConfig


if __name__ =="__main__":
    # print(tf.__version__)
    #test.ImTest()
    #test.LRTest()

    Train.ModelFit(EPOCHES = 30,SaveCheckpoint = True,SaveFigure = True,SaveModel = True)
    # ModelName = f"ResNet34_30_{LEARNINGRATE_1}"

