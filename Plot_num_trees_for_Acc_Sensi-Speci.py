import matplotlib.pyplot as plt

Acc_ = [0.6341911764705882, 0.6011029411764706, 0.6746323529411765, 0.6617647058823529, 0.6709558823529411,
       0.6397058823529411]
Sensi_ = [0.5955882352941176, 0.6176470588235294, 0.6213235294117647, 0.5808823529411765, 0.5588235294117647,
         0.49264705882352944]
Speci_ = [0.6727941176470589, 0.5845588235294118, 0.7279411764705882, 0.7426470588235294, 0.7830882352941176,
         0.7867647058823529]
Acc = [i*100 for i in Acc_]
Sensi = [i*100 for i in Sensi_]
Speci = [i*100 for i in Speci_]
num_trees = [10, 20, 30, 40, 50, 60]

plt.rcParams['font.family'] = 'Times New Roman' 
plt.rcParams['font.size'] = '28' 

plt.plot(num_trees, Sensi, color='b', marker='o')
plt.plot(num_trees, Speci,color='g', marker='s')
plt.plot(num_trees, Acc,color='r',marker='v')
plt.xlabel('Tree number')
plt.ylabel('Metric values[%]')
setx = [10, 20, 30, 40, 50, 60]
#sety = [28, 30, 32]
plt.xticks(setx)
#plt.yticks(sety)
plt.legend(['Sensi', 'Speci', 'Accuracy'], loc=1, bbox_to_anchor=(0.37,1.0))


smallfont = 26
for i in range(len(num_trees)):
    plt.text(num_trees[i], Acc[i], "%.1f" %Acc[i], fontsize=smallfont, verticalalignment="bottom",horizontalalignment="center")
    plt.text(num_trees[i], Speci[i], "%.1f" % Speci[i], fontsize=smallfont, verticalalignment="bottom",horizontalalignment="center")
    plt.text(num_trees[i], Sensi[i], "%.1f" % Sensi[i], fontsize=smallfont, verticalalignment="bottom",horizontalalignment="center")

plt.show()
