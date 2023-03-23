import numpy as np
import cv2

'''
双线性插值

1、几何中心重合 
src_x - 0.5 = (dst_x - 0.5)*srcWidth/dstWidth
src_y - 0.5 = (dst_y - 0.5)*srcHeight/dstHeight
2、双线性插值公式
f(i+u, j+v) = (1-u) * (1-v) * f(i, j) + (1-u) * v * f(i, j+1) + u * (1-v) * f(i+1, j) + u * v * f(i+1, j+1)

思路：
建立一个空的图像，尺寸为缩放后的尺寸，通道为原图像的通道数，
循环通道，用双线性插值方法计算单通道上每个像素点位的像素值，
生成结果图像

'''

def biliner_interpolation(img,out_shape):
    sw,sh,c=img.shape
    dst_w,dst_h = out_shape[0],out_shape[1]

    if sw == dst_w and sh == dst_h:
        return img

    result_img = np.zeros((dst_w,dst_h,c),img.dtype)
    for i in range(c):
        for dst_x in range(dst_w):
            for dst_y in range(dst_h):
                # 几何中心重合，计算出原像素点相对于目标点的位置 (虚拟点位)
                # src_x - 0.5 = (dst_x - 0.5) * srcWidth / dstWidth
                # src_y - 0.5 = (dst_y - 0.5) * srcHeight / dstHeight
                src_x = (dst_x + 0.5)*sw/dst_w - 0.5
                src_y = (dst_y + 0.5)*sh/dst_h - 0.5

                # 构造src虚拟像素点的最邻近的真实像素点，构成一个框
                src_x1 = int(np.floor(src_x))
                src_x2 = min(src_x1 + 1, sw-1)
                src_y1 = int(np.floor(src_y))
                src_y2 = min(src_y1 + 1, sh-1)

                # 双线性插值，计算目标点像素值
                temp0 = (src_x2 - src_x)*img[src_x1,src_y1,i] + (src_x - src_x1)*img[src_x2,src_y1,i]
                temp1 = (src_x2 - src_x)*img[src_x1,src_y2,i] + (src_x - src_x1)*img[src_x2,src_y2,i]
                result_img[dst_x,dst_y,i] = int((src_y2 - src_y)*temp0 + (src_y - src_y1)*temp1)

    return result_img


img = cv2.imread("dog1.jpg")
result_img = biliner_interpolation(img,(800,700))

cv2.imshow("normal figure", img)
cv2.imshow("after biliner figure", result_img)
cv2.waitKey(0)












