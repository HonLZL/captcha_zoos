import copy
import os.path

import cv2
import numpy as np


def get_point_area(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    # img - 数据类型为float32的输入图像。
    # blockSize - 角点检测中要考虑的邻域大小。
    # ksize-Sobel求导中使用的窗口大小
    # k - Harris 角点检测方程中的自由参数, 取值参数为[0.04, 0.06].
    dst = cv2.cornerHarris(gray, 3, 3, 0.04)

    # 膨胀, 提升后续图像角点标注的清晰准确度
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    dst = cv2.dilate(dst, kernel)
    img[dst > 0.01 * dst.max()] = [0, 0, 0]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mp = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)  # 划分后的区域

    # cv2.imshow("mp", mp)

    # 提取轮廓, 第一个返回值为 列表：每个轮廓边缘点
    # 第二个返回值是个矩阵为轮廓之间的关系，大小为 轮廓个数×4，
    contours, _ = cv2.findContours(mp, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    idx = 0
    max_area = 0
    max_area_idx = 0
    for contour in contours:
        cv2.drawContours(img, [contour], -1, (255, 0, 255), 2)

        tem_area = cv2.contourArea(contour)
        if tem_area > max_area:
            max_area_idx = idx
            max_area = tem_area
        idx += 1

    max_area_x = 0
    max_area_y = 0

    # 计算到轮廓的距离
    max_dist = 0  # 半径
    for i in range(mp.shape[1]):
        for j in range(mp.shape[0]):
            # 点到轮廓的最大距离
            dist = cv2.pointPolygonTest(contours[max_area_idx], (i, j), True)
            if dist > max_dist:
                max_dist = dist
                max_area_x, max_area_y = abs(i), abs(j)

    return max_area_x, max_area_y, max_area


if __name__ == "__main__":
    img_path = "imgs/1.png"
    img = cv2.imread(img_path)
    result_img = copy.deepcopy(img)
    x, y, area = get_point_area(img)
    cv2.circle(result_img, (x, y), 6, color=(30, 144, 255), thickness=-1)
    img_root, img_name = os.path.split(img_path)
    cv2.imwrite(os.path.join(img_root, "res_"+img_name), result_img)
    cv2.imshow("result", result_img)
    cv2.waitKey(0)
