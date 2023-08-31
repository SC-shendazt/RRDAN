import numpy as np


def gamma_transform(img,gamma):
    """
    gamma transform 2d grayscale image, convert uint image to float
    param: input img: input grayscale image
    param: input c: scale of the transform
    param: input gamma: gamma value of the transoform
    """
    img = img.astype(float)  # 先要把图像转换成为float，不然结果点不太相同
    img=img/(img.max())
    epsilon = 1e-5  # 非常小的值以防出现除0的情况

    img_dst = np.power(img + epsilon, gamma)*255
    img_dst=img_dst.astype(np.uint8)

    return img_dst