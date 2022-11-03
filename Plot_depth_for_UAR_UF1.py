import matplotlib.pyplot as plt



UAR_ = [0.6084558823529411,0.6746323529411764,0.6360294117647058,0.6286764705882353,0.6305147058823529]
UF1_ = [0.6079776706419691,0.6737050780257205,0.6350626118067979,0.6279523293607802,0.6304834899682674]
UAR = [i*100 for i in UAR_]
UF1 = [i*100 for i in UF1_]
num_trees = [2,3,4,5,6]

plt.rcParams['font.family'] = 'Times New Roman' 
plt.rcParams['font.size'] = '12'

plt.plot(num_trees, UAR, 'g.-',num_trees, UF1, 'r.-')
plt.xlabel('Tree depth')
plt.ylabel('Metric values[%]')
setx = [2,3,4,5,6]
#sety = [28, 30, 32]
plt.xticks(setx)
#plt.yticks(sety)
plt.legend(['UAR', 'UF1'])


for i in range(len(num_trees)):
    plt.text(num_trees[i], UAR[i], "%.3f" %UAR[i], fontsize=12, verticalalignment="top",horizontalalignment="center")
    plt.text(num_trees[i], UF1[i], "%.3f" %UF1[i], fontsize=12, verticalalignment="bottom",horizontalalignment="center")


plt.show()

