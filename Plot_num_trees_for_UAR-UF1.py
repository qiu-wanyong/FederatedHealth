import matplotlib.pyplot as plt

UAR_ = [0.6341911764705883, 0.6011029411764706, 0.6746323529411764, 0.661764705882353, 0.6709558823529411,
       0.6397058823529411]
UF1_ = [0.6336452393441513, 0.6009937300366733, 0.6737050780257205, 0.6595374149659865, 0.6667659086631419,
       0.6317418213969939]
num_trees = [10, 20, 30, 40, 50, 60]
UAR= [i*100 for i in UAR_]
UF1 = [i*100 for i in UF1_]

plt.rcParams['font.family'] = 'Times New Roman' 
plt.rcParams['font.size'] = '12'

plt.plot(num_trees, UAR, 'g.-',num_trees, UF1, 'r.-')
plt.xlabel('Tree number')
plt.ylabel('Metric values[%]')
setx = [10, 20, 30, 40, 50, 60]
#sety = [28, 30, 32]
plt.xticks(setx)
#plt.yticks(sety)
plt.legend(['UAR', 'UF1'])

for i in range(len(num_trees)):
    plt.text(num_trees[i], UAR[i], "%.3f" %UAR[i], fontsize=12, verticalalignment="bottom",horizontalalignment="left")
    plt.text(num_trees[i], UF1[i], "%.3f" %UF1[i], fontsize=12, verticalalignment="top",horizontalalignment="left")


plt.show()
