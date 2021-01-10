import imageio
import numpy as np
import cv2

class Data:
    def __init__(self):
        pass

    def dilate(self, img):
        kernel = np.ones((2, 2), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
        return img

    def forward(self, path):
        img = cv2.imread(path, 0)
        h, w = img.shape
        count = 0
        while h > 17 and w > 17:
            h = h / 2
            w = w / 2
            h = int(h)
            w = int(w)
            img = cv2.resize(img, (h, w))
            if count % 2 == 0:
                img = self.dilate(img)
            count += 1
        img = cv2.resize(img, (17, 17))
        img = img / 255
        img[img > 0] = 1
        #print(img)
        img.resize((img.size, 1))
        self.x = np.asarray(img)
        return self.x


class Output:
    def __init__(self):
        pass

    def forward(self, x):
        self.x = x
        output = chr(np.argmax(self.x) + 65)
        return output


class FullyConnect:
    def __init__(self, dst, layer):
        with open(dst + 'weight' + layer + '.npy', 'rb') as f:
            self.weight = np.load(f, allow_pickle=True)
        with open(dst + 'bias' + layer + '.npy', 'rb') as f:
            self.bias = np.load(f, allow_pickle=True)

    def forward(self, x):
        self.x = x
        self.y = np.array(np.dot(self.weight, x) + self.bias)
        return self.y


class Sigmoid:
    def __init__(self):
        pass

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        self.x = x
        self.y = self.sigmoid(x)
        return self.y


def main():
    paradst = 'D:\py project\saveed para test\\'
    inner_layers = []
    inner_layers.append(FullyConnect(paradst, '1'))
    inner_layers.append(Sigmoid())
    inner_layers.append(FullyConnect(paradst, '2'))
    inner_layers.append(Sigmoid())
    inner_layers.append(Output())
    path = "D:\py project\saveed para test\prove pic\\B.png"
    x = Data()
    x = x.forward(path)

    for layer in inner_layers:
        x = layer.forward(x)

    #print(inner_layers[3].y)
    print('The character is:', x)


if __name__ == '__main__':
    main()
