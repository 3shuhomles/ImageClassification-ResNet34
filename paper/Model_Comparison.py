
import matplotlib.pyplot as plt

# 显示中文
plt.rcParams["font.sans-serif"] = "SimHei"
plt.rcParams['axes.unicode_minus'] = False

'''config'''
EPOCHES = 30


x_data = range(1,EPOCHES+1,1)

'''ResNet34与CNN_34'''
# ResNet34_data_loss = [1.7667027711868286, 1.167555332183838, 0.9779223203659058, 0.850730836391449, 0.7414120435714722, 0.6850845813751221, 0.6183741092681885, 0.5714433193206787, 0.5254154801368713, 0.47209298610687256, 0.43921148777008057, 0.42278629541397095, 0.3600071966648102, 0.3370297849178314, 0.3129638135433197, 0.28598007559776306, 0.27458128333091736, 0.26711925864219666, 0.24590948224067688, 0.21477116644382477, 0.20398259162902832, 0.19823014736175537, 0.18036669492721558, 0.17420275509357452, 0.14211668074131012, 0.16795465350151062, 0.13800624012947083, 0.16239991784095764, 0.12935645878314972, 0.11582759767770767]
# CNN_34_data_loss = [1.9069080352783203, 1.4055428504943848, 1.1556072235107422, 1.0632003545761108, 0.9868565797805786, 0.8961209058761597, 0.8660752773284912, 0.8140285015106201, 0.752570390701294, 0.6991541981697083, 0.651786208152771, 0.6252066493034363, 0.5645811557769775, 0.5115748047828674, 0.4747845530509949, 0.4537433981895447, 0.41388770937919617, 0.37655535340309143, 0.33274686336517334, 0.3441166281700134, 0.3023914396762848, 0.29404523968696594, 0.23912614583969116, 0.24735313653945923, 0.24385832250118256, 0.2304505705833435, 0.1714702695608139, 0.19631431996822357, 0.17372871935367584, 0.13525068759918213]
#
# ResNet34_data_acc = [0.47733333706855774, 0.659333348274231, 0.7224444150924683, 0.7684444189071655, 0.7986666560173035, 0.8253333568572998, 0.8455555438995361, 0.8591111302375793, 0.8731111288070679, 0.886888861656189, 0.8999999761581421, 0.9026666879653931, 0.9259999990463257, 0.9262222051620483, 0.9359999895095825, 0.9422222375869751, 0.9446666836738586, 0.9493333101272583, 0.9520000219345093, 0.961555540561676, 0.9644444584846497, 0.9635555744171143, 0.972000002861023, 0.9691110849380493, 0.9800000190734863, 0.9708889126777649, 0.9797777533531189, 0.9711111187934875, 0.9795555472373962, 0.9822221994400024]
# CNN_34_data_acc = [0.26644444465637207, 0.4633333384990692, 0.5611110925674438, 0.6002222299575806, 0.6224444508552551, 0.6526666879653931, 0.6633333563804626, 0.6773333549499512, 0.699999988079071, 0.7266666889190674, 0.7517777681350708, 0.7702222466468811, 0.788444459438324, 0.7995555400848389, 0.820888876914978, 0.8308888673782349, 0.851111114025116, 0.8608888983726501, 0.8826666474342346, 0.8733333349227905, 0.891777753829956, 0.8915555477142334, 0.9108889102935791, 0.913777768611908, 0.9135555624961853, 0.9248889088630676, 0.9408888816833496, 0.9311110973358154, 0.9413333535194397, 0.9513333439826965]
#
# fig = plt.figure(figsize=(10,8),dpi=100)
#
# fig.add_subplot(1,2,1)
# plt.plot(x_data,ResNet34_data_loss)
# plt.plot(x_data,CNN_34_data_loss)
# plt.legend(['ResNet34','CNN_34'])
# plt.xlabel('iteration')
# plt.ylabel('loss')
# plt.title("Comparison of loss of ResNet34 and CNN_34")
#
# fig.add_subplot(1,2,2)
# plt.plot(x_data,ResNet34_data_acc)
# plt.plot(x_data,CNN_34_data_acc)
# plt.legend(['ResNet34','CNN_34'])
# plt.xlabel('iteration')
# plt.ylabel('acc')
# plt.title("Comparison of loss of ResNet34 and CNN_34")
#
# plt.show()


'''CNN_34与CNN_34_notBN'''
CNN_34_data_loss = [1.9069080352783203, 1.4055428504943848, 1.1556072235107422, 1.0632003545761108, 0.9868565797805786, 0.8961209058761597, 0.8660752773284912, 0.8140285015106201, 0.752570390701294, 0.6991541981697083, 0.651786208152771, 0.6252066493034363, 0.5645811557769775, 0.5115748047828674, 0.4747845530509949, 0.4537433981895447, 0.41388770937919617, 0.37655535340309143, 0.33274686336517334, 0.3441166281700134, 0.3023914396762848, 0.29404523968696594, 0.23912614583969116, 0.24735313653945923, 0.24385832250118256, 0.2304505705833435, 0.1714702695608139, 0.19631431996822357, 0.17372871935367584, 0.13525068759918213]
CNN_34_notBN_data_loss = [2.302598476409912, 2.2945573329925537, 2.302596092224121, 2.302586317062378, 2.302579164505005, 2.302572727203369, 2.3025600910186768, 2.1898980140686035, 1.9042072296142578, 1.7500808238983154, 1.5992804765701294, 1.5003302097320557, 1.4749559164047241, 1.4222294092178345, 1.4155937433242798, 1.375799536705017, 1.3510797023773193, 1.345116138458252, 1.3137080669403076, 1.2578736543655396, 1.2232085466384888, 1.1897625923156738, 1.1704885959625244, 1.1977686882019043, 1.1284438371658325, 1.1337379217147827, 1.1294140815734863, 1.1171313524246216, 1.1267260313034058, 1.0456569194793701]
plt.plot(x_data,CNN_34_data_loss)
plt.plot(x_data,CNN_34_notBN_data_loss)
plt.legend(['BN','not BN'])
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title("Comparison of loss of BN and notBN")

plt.show()
