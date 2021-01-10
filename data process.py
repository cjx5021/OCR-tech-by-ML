# encoding=utf-8
import imageio
import numpy as np

class data:
    def __init__(self,path,batch_size):
        with open(path,'rb') as f:
            data  = f.load
        self.x = data[0] #input
        self.y = data[1] #expective value
        self.batch = batch_size
        self.pos = 0



def main(src, dst):   
    with open(src, 'r') as f:  
        list = f.readlines()
    data = []
    labels = []
    for i in list:
        name, label = i.strip('\n').split(' ')  
        print(name + ' processed')
        img = imageio.imread("D:\py project\ocr data\\" + name)
        img = img/255 
        img.resize((img.size, 1))  
        data.append(img)
        labels.append(int(label))

    print('write to npy')
    np.save(dst, [data, labels])  
    print('completed')


if __name__ == '__main__':
    main('D:\py project\ocr data\\validate.txt','D:\py project\ocr data\\validate')