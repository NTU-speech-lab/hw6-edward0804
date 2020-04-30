import os
import pandas as pd
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys


device = torch.device("cuda")
def prob(output):
    x = []
    y = []
    probs = F.softmax(output,dim=1)
    max_p,index = torch.max(probs,dim=1)
    x.append(index.item())
    y.append(max_p.item())
    probs[0][index.item()] = 0
    max_p,index = torch.max(probs,dim=1)
    x.append(index.item())
    y.append(max_p.item())
    probs[0][index.item()] = 0
    max_p,index = torch.max(probs,dim=1)
    x.append(index.item())
    y.append(max_p.item())
    
    return x,y
# 實作一個繼承 torch.utils.data.Dataset 的 Class 來讀取圖片
class Adverdataset(Dataset):
    def __init__(self, root, label, transforms):
        # 圖片所在的資料夾
        self.root = root
        # 由 main function 傳入的 label
        self.label = torch.from_numpy(label).long()
        # 由 Attacker 傳入的 transforms 將輸入的圖片轉換成符合預訓練模型的形式
        self.transforms = transforms
        # 圖片檔案名稱的 list
        self.fnames = []

        for i in range(200):
            self.fnames.append("{:03d}".format(i))

    def __getitem__(self, idx):
        # 利用路徑讀取圖片
        img = Image.open(os.path.join(self.root, self.fnames[idx] + '.png'))
        # 將輸入的圖片轉換成符合預訓練模型的形式
        img = self.transforms(img)
        # 圖片相對應的 label
        label = self.label[idx]
        return img, label
    
    def __len__(self):
        # 由於已知這次的資料總共有 200 張圖片 所以回傳 200
        return 200
x1 = []
x2 = []
y1 = []
y2 = []

class Attacker:
    def __init__(self, img_dir, label):
        # 讀入預訓練模型 vgg16
        self.model = models.densenet121(pretrained = True)
        self.model.cuda()
        self.model.eval()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        # 把圖片 normalize 到 0~1 之間 mean 0 variance 1
        self.normalize = transforms.Normalize(self.mean, self.std, inplace=False)
        transform = transforms.Compose([                
                        transforms.Resize((224, 224), interpolation=3),
                        transforms.ToTensor(),
                        self.normalize
                    ])
        # 利用 Adverdataset 這個 class 讀取資料
        input_prefix = sys.argv[1]
        self.dataset = Adverdataset(os.path.join( input_prefix,'images'), label, transform)
        
        self.loader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size = 1,
                shuffle = False)

    # FGSM 攻擊
    def fgsm_attack(self, image, epsilon, data_grad):
        # 找出 gradient 的方向
        sign_data_grad = data_grad.sign()
        # 將圖片加上 gradient 方向乘上 epsilon 的 noise
        perturbed_image = image + epsilon * sign_data_grad
        return perturbed_image
    
    def attack(self, epsilon):            
        # 存下一些成功攻擊後的圖片 以便之後顯示
        cnt=-1
        adv_examples = []
        wrong, fail, success = 0, 0, 0
        for (data, target) in self.loader:
            cnt+=1
            output_prefix = sys.argv[2]
            data, target = data.to(device), target.to(device)
            data_raw = data
            data.requires_grad = True
            # 將圖片丟入 model 進行測試 得出相對應的 class
            output = self.model(data)
            init_pred = output.max(1, keepdim=True)[1]
            
            if (cnt == 0 or cnt == 4 or cnt == 16):
                x,y = prob(output)
                x1.append(x)
                y1.append(y)
            # 如果 class 錯誤 就不進行攻擊
            if init_pred.item() != target.item():

                adv_ex = data_raw * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
                adv_ex = adv_ex.squeeze().detach().cpu().numpy() 
                adv_ex = np.transpose(adv_ex, (1, 2, 0))
                np.clip(adv_ex,0,1,out=adv_ex)

                plt.imsave(os.path.join(output_prefix, '%03d'%cnt +'.png'),adv_ex)

                wrong += 1
                continue
            
            # 如果 class 正確 就開始計算 gradient 進行 FGSM 攻擊
            loss = F.nll_loss(output, target)
            self.model.zero_grad()
            loss.backward()
            data_grad = data.grad.data
            perturbed_data = self.fgsm_attack(data, epsilon, data_grad)

            # 再將加入 noise 的圖片丟入 model 進行測試 得出相對應的 class        
            output = self.model(perturbed_data)        
            final_pred = output.max(1, keepdim=True)[1]

            if (cnt == 0 or cnt == 4 or cnt == 16):
                x,y = prob(output)
                x2.append(x)
                y2.append(y)

            adv_ex = perturbed_data * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
            adv_ex = adv_ex.squeeze().detach().cpu().numpy() 
            adv_ex = np.transpose(adv_ex, (1, 2, 0))
            np.clip(adv_ex,0,1,out=adv_ex)

            plt.imsave(os.path.join(output_prefix, '%03d'%cnt +'.png'),adv_ex)
            if final_pred.item() == target.item():
                # 辨識結果還是正確 攻擊失敗
                fail += 1
            else:
                # 辨識結果失敗 攻擊成功
                success += 1
                # 將攻擊成功的圖片存入
                if len(adv_examples) < 5:
                  adv_ex = perturbed_data * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
                  adv_ex = adv_ex.squeeze().detach().cpu().numpy() 
                  data_raw = data_raw * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
                  data_raw = data_raw.squeeze().detach().cpu().numpy()

                  adv_examples.append( (init_pred.item(), final_pred.item(), data_raw , adv_ex) )  

        final_acc = (fail / (wrong + success + fail))
        
        print("Epsilon: {}\tTest Accuracy = {} / {} = {}\n".format(epsilon, fail, len(self.loader), final_acc))
        return adv_examples, final_acc


if __name__ == '__main__':
    # 讀入圖片相對應的 label
    input_prefix = sys.argv[1]

    df = pd.read_csv(os.path.join( input_prefix,"labels.csv"))
    df = df.loc[:, 'TrueLabel'].to_numpy()
    label_name = pd.read_csv(os.path.join( input_prefix,"categories.csv"))
    label_name = label_name.loc[:, 'CategoryName'].to_numpy()
    # new 一個 Attacker class
    attacker = Attacker(os.path.join( input_prefix,'images'), df)
    # 要嘗試的 epsilon
    epsilons = [0.1]

    accuracies, examples = [], []
    # 進行攻擊 並存起正確率和攻擊成功的圖片
    for eps in epsilons:
        ex, acc = attacker.attack(eps)
        accuracies.append(acc)
        examples.append(ex)
import matplotlib.pyplot as plt
index = [0,1,2]
name = ['1.png','2.png','3.png','4.png','5.png','6.png']
for i in range(len(x1)):
    plt.cla()
    plt.ylabel("Probability")
    plt.xlabel("class")
    plt.title("Origin picture")
    plt.xticks(index,x1[i])
    plt.bar(index,y1[i])
    plt.savefig(name[i*2])

    plt.cla()
    plt.ylabel("Probability")
    plt.xlabel("class")
    plt.title("Adversarial picture")
    plt.xticks(index,x2[i])
    plt.bar(index,y2[i])
    plt.savefig(name[i*2+1],color='red')
































        
