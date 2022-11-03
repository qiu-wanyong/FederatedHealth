import matplotlib.pyplot as plt


Acc_ = [0.6084558823529411,0.6746323529411765,0.6360294117647058,0.6286764705882353,0.6305147058823529]
Sensi_ = [0.6433823529411765,0.6213235294117647,0.5845588235294118,0.5845588235294118,0.6213235294117647]
Speci_ = [0.5735294117647058,0.7279411764705882,0.6875,0.6727941176470589,0.6397058823529411]
Acc = [i*100 for i in Acc_]
Sensi = [i*100 for i in Sensi_]
Speci = [i*100 for i in Speci_]
num_trees = [2,3,4,5,6]

plt.rcParams['font.family'] = 'Times New Roman' 
plt.rcParams['font.size'] = '12' 

plt.plot(num_trees, Sensi, 'b.-', num_trees, Speci, 'g.-', num_trees, Acc, 'r.-')
plt.xlabel('Tree depth')
plt.ylabel('Metric values[%]')
setx = [2,3,4,5,6]
#sety = [28, 30, 32]
plt.xticks(setx)
#plt.yticks(sety)
plt.legend(['Sensi', 'Speci', 'Accuracy'])


for i in range(len(num_trees)):
    plt.text(num_trees[i], Acc[i], "%.3f" %Acc[i], fontsize=12, verticalalignment="bottom",horizontalalignment="center")
    plt.text(num_trees[i], Speci[i], "%.3f" % Speci[i], fontsize=12, verticalalignment="bottom",horizontalalignment="center")
    plt.text(num_trees[i], Sensi[i], "%.3f" % Sensi[i], fontsize=12, verticalalignment="bottom",horizontalalignment="center")

plt.show()

