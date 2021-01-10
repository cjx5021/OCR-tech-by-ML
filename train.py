import imageio
import numpy as np


class Data:
    def __init__(self, path, batch_size):
        with open(path, 'rb') as f:
            data = np.load(f, allow_pickle=True)
        self.x = data[0] #input
        self.y = data[1] #expective value
        self.batch = batch_size
        self.pos = 0

    def forward(self):
        pos = self.pos
        bat = self.batch
        l = len(self.x)
        if pos+bat >= l:
            ret = (self.x[pos:l], self.y[pos:l])
            self.pos = 0
            index = range(l)
            np.random.shuffle(list(index))
            self.x = self.x[index]
            self.y = self.y[index]
        else:
            ret = (self.x[pos:pos+bat], self.y[pos:pos+bat])
            self.pos += self.batch
        return ret, self.pos


class Accuracy:
    def __init__(self):
        pass

    def forward(self, x, lable):
        accuracy = np.sum([np.argmax(xx) == ll for xx, ll in zip(x, lable)])
        accuracy = accuracy/x.shape[0]
        return accuracy


class QuadraticLoss:
    def __init__(self):
        pass

    def forward(self, x, lable):
        self.x = x
        self.lable = np.zeros_like(x)
        for a, b in zip(self.lable, lable):
            a[b] = 1.0
        self.y = np.sum(np.square(x - self.lable))/self.x.shape[0]/2
        return self.y

    def backward(self):
        self.dx = (self.x - self.lable)/self.x.shape[0]
        return self.dx


class CrossEntropy:
    def __init__(self):
        pass

    def forward(self, x, lable):
        self.x = x
        self.lable = np.zeros_like(x)
        for a, b in zip(self.lable, lable):
            a[b] = 1.0
        self.y = np.nan_to_num(-self.lable * np.log(self.x) - (1 - self.lable) * np.log(1 - self.x))
        self.y = np.sum(self.y)/self.x.shape[0]
        return self.y

    def backward(self):
        self.dx = (self.x - self.lable) / self.x / (1 - self.x) / self.x.shape[0]
        return self.dx


class FullyConnect:
    def __init__(self, l_x, l_y):
        self.weight = np.random.randn(l_y, l_x) / np.sqrt(l_x)
        self.bias = np.random.randn(l_y, 1)
        self.lr = 900


    def forward(self,x):
        self.x = x
        self.y = np.array([np.dot(self.weight, xx) + self.bias for xx in x])
        return self.y

    def backward(self, d):
        ddw = np.array([np.dot(dd, xx.T) for dd, xx in zip(d, self.x)])
        self.dw = np.sum(ddw, axis=0)/self.x.shape[0]
        self.db = np.sum(d, axis=0)/self.x.shape[0]
        self.dx = np.array([np.dot(self.weight.T, dd) for dd in d])

        self.weight -= self.lr * self.dw
        self.bias -= self.lr * self.db

        return self.dx


class Output:
    def __init__(self):
        pass

    def forward(self, x):
        self.x = x
        output = chr(np.argmax(self.x) + 65)
        return output


class Sigmoid:
    def __init__(self):
        pass

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        self.x = x
        self.y = self.sigmoid(x)
        return self.y

    def backward(self, d):
        sig = self.sigmoid(self.x)
        self.dx = d * sig * (1 - sig)
        return self.dx


class Relu:
    def __init__(self):
        pass

    def forward(self, x):
        return (np.abs(x) + x) / 2

    def backward(self, d):
        d[d <= 0] = 0
        return d

def main():
    #  1. Epoch
    #  2. batch
    #  3. forward/back
    #  4. loss/accuracy
    #  5. Print batch/epoch information
    epoch = 1
    datalayer1 = Data('D:\py project\ocr data\\train.npy', 900)
    datalayer2 = Data('D:\py project\ocr data\\validate.npy',10000)
    inner_layers = []
    inner_layers.append(FullyConnect(17 * 17, 26))
    inner_layers.append(Sigmoid())
    inner_layers.append(FullyConnect(26, 26))  # 增加一个隐层
    inner_layers.append(Sigmoid())
    loss_layer = CrossEntropy()
    accuracy = Accuracy()
    for layer in inner_layers:
        layer.lr = 900.0  # 为所有中间层设置学习速率

    while epoch <= 100:
        loss_sum = 0
        count = 0
        print('epoch:', epoch)
        while True:
            count += 1
            data, pos = datalayer1.forward()
            x, label = data
            # forward
            for layer in inner_layers:  # 前向计算
                x = layer.forward(x)

            loss = loss_layer.forward(x, label)  # 调用损失层forward函数计算损失函数值
            loss_sum += loss
            count += 1

            # backward
            d = loss_layer.backward()
            for layer in inner_layers[::-1]:  # 反向传播
                d = layer.backward(d)

            if pos == 0:
                break

        total_loss = loss_sum/count
        print('total loss:', total_loss)
        epoch += 1

        data2, pos2 = datalayer2.forward()
        x2, label2 = data2

        for layer in inner_layers:
            x2 = layer.forward(x2)

        accuracy_for_each_epoch = accuracy.forward(x2, label2)
        print('accuracy:', accuracy_for_each_epoch)

    np.save('D:\py project\saveed para test\\weight1', inner_layers[0].weight)
    np.save('D:\py project\saveed para test\\bias1', inner_layers[0].bias)
    np.save('D:\py project\saveed para test\\weight2', inner_layers[2].weight)
    np.save('D:\py project\saveed para test\\bias2', inner_layers[2].bias)


if __name__ == '__main__':
    main()
