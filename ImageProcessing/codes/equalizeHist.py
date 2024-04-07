import cv2
import numpy as np 
from numpy import zeros
import matplotlib.pyplot as plt

class GetHistogram:
    def __init__(self,path):
        self.path=path
        self.img=cv2.imread(self.path)
        self.grayscaledImg =cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        self.height,self.width = self.grayscaledImg.shape
        
    def getHighestGrayVal(self,tempArr):
        highestGrayVal = 0
        height,width = tempArr.shape
        for i in range(height):
            for j in range(width):
                if(tempArr[i][j]>highestGrayVal):
                    highestGrayVal=tempArr[i][j]
        return highestGrayVal        
    def getCorresponding2DHistogramVec(self,tempArr):
        highestVal = self.getHighestGrayVal(tempArr)
        height,width = tempArr.shape
        valArr = [0]*(highestVal+1)
        for i in range(height):
            for j in range(width):
                valArr[tempArr[i][j]] += 1 
        return valArr
    def getNormalizedCumulativeSum(self,array):
        arr = self.getCorresponding2DHistogramVec(array)
        arrIterator = iter(arr)##define iterator
        b = [next(arrIterator)]
        for iteratedElement in arrIterator:
            lastAddedElement = b[-1]
            b.append(lastAddedElement + iteratedElement)
        cumulative_freq = np.array(b)
        nj = (cumulative_freq ) * (self.getHighestGrayVal(array)-1)
        N = cumulative_freq.max()
        cumulative_freq = nj / N
        cumulative_freq = cumulative_freq.astype('int')
        return cumulative_freq
    def get1DArrayoftheImg(self):
        flatArr = self.grayscaledImg.flatten()
        return flatArr
    def exec(self):
        chestHistArr = self.getCorresponding2DHistogramVec(self.grayscaledImg)
        plt.plot(chestHistArr)
        plt.xlabel("Pixel Value")
        plt.ylabel("Number Of Pixels")
        plt.title("Histogram Graph before Equalization")
        cum_sum=self.getNormalizedCumulativeSum(self.grayscaledImg)
        img_normalized = cum_sum[self.get1DArrayoftheImg()]
        img_normalized = np.reshape(img_normalized, self.grayscaledImg.shape)
        fig = plt.figure()
        fig.add_subplot(1,2,1)
        plt.imshow(self.grayscaledImg, cmap='gray')
        fig.add_subplot(1,2,2)
        plt.imshow(img_normalized, cmap='gray')
        plt.show(block=True)
        normalizedImgHist = self.getCorresponding2DHistogramVec(img_normalized)
        plt.plot(normalizedImgHist)
        plt.xlabel("Pixel Value")
        plt.ylabel("Number Of Pixels")
        plt.title("Histogram Graph after Equalization")
        plt.show(block=True)

        


chest = GetHistogram('images/soru1-chest.tif')
chest.exec()

lena = GetHistogram('images/soru1-lena.bmp')
lena.exec()

pepper = GetHistogram('images/soru1-pepper.bmp')
pepper.exec()