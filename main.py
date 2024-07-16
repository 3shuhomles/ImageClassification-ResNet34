
import test
import Src.Train as Train
import Src.Model.ModelPara as ModelPara
from Src import PathConfig


if __name__ =="__main__":
    # print(tf.__version__)
    #test.ImTest()
    #test.LRTest()

    # ModelName = "Res34_TrainData100"
    # "Res34_TrainData1800_AdaMax_0.001_epoch30"
    Train.ModelFit(ModelName = "save_weights_h5",EPOCHES = ModelPara.EPOCHES,SaveFigure = True,SaveModel=True,SaveModelFormat="ckpt",SaveWeightsOnly=True)
    # ModelName = f"ResNet34_30_{LEARNINGRATE_1}"



