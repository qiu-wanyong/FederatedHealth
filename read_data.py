from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)

root_dir = "E:\PycharmSpace\pythonCourse\pytorchTest\dataset\\train"
ants_label_dir = "ants_image"
ants_dataset = MyData(root_dir, ants_label_dir)
img, label = ants_dataset[0]
img.show()

bees_label_dir = "bees_image"
bees_dataset =MyData(root_dir, bees_label_dir)

train_dataset = ants_dataset + bees_dataset
print(len(train_dataset))



