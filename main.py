
import test
import Src.Train as Train
import Src.Model.ModelPara as ModelPara
from Src import PathConfig


if __name__ =="__main__":
    # print(tf.__version__)
    #test.ImTest()
    #test.LRTest()

    # ModelName = "Res34_TrainData100"
    Train.ModelFit(ModelName = "Res34_TrainData100",EPOCHES = ModelPara.EPOCHES,SaveFigure = True,SaveModel=True,SaveModelFormat="h5",SaveWeightsOnly=True)
    # ModelName = f"ResNet34_30_{LEARNINGRATE_1}"


