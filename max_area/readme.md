# 面积最大验证码

## 一 哈里斯角点

### 1.1 角点

​		角点可以简单理解为是两条边的交点。可以类比我们的桌角、笔尖等，如果将桌子和笔拍下来，桌角和笔尖都会是角点。

严格来说：角点指的是在邻域内具有两个（及以上）主方向的特征点。它在图像中有具体的坐标和某些数学特征，通常表现为：

- 轮廓之间的交点
- 对于同一场景，即使视角发生变化，通常具备稳定性质的特征
- 该点附近区域的像素点无论在梯度方向上还是其梯度幅值上有着较大变化

如下图所示，在各个方向上移动小窗口，如果在所有方向上移动，窗口内灰度都发生变化，则认为是角点；如果任何方向都不变化，则是均匀区域；如果灰度只在一个方向上变化，则可能是图像边缘。

![在这里插入图片描述](D:\Desktop\1data\max_area\doc_imgs\watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2Fhcm9uam55,size_16,color_FFFFFF,t_70)

### 1.2 哈里斯角点检测原理

[原理](https://blog.csdn.net/lwzkiller/article/details/54633670) 

> 算法基本思想是使用一个固定窗口在图像上进行任意方向上的滑动，比较滑动前与滑动后两种情况，窗口中的像素灰度变化程度，如果存在任意方向上的滑动，都有着较大灰度变化，那么我们可以认为该窗口中存在角点。

Opencv 中的函数 cv2.cornerHarris() 可以用来进行角点检测。参数如
下:
　　• img - 数据类型为 float32 的输入图像。
　　• blockSize - 角点检测中要考虑的邻域大小。
　　• ksize - Sobel 求导中使用的窗口大小
　　• k - Harris 角点检测方程中的自由参数,取值参数为 [0.04,0.06].

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
```

dst 即为检测出的角点





## 二 区域面积计算

### 2.1 检测轮廓

先膨胀，使各点扩大，达到点碰点，密不透风的程度，将角点置为纯黑，二值化后其他点为纯白，方便检测轮廓。

```python
# 膨胀, 提升后续图像角点标注的清晰准确度
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
dst = cv2.dilate(dst, kernel)
img[dst > 0.01*dst.max()] = [0, 0, 0]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, mp = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)  # 划分后的区域
```

![image-20211224095726376](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20211224095726376.png)

![image-20211224095553489](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20211224095553489.png)





### 2.2 计算面积

```python
# 提取轮廓, 第一个返回值为 列表：每个轮廓边缘点
# 第二个返回值是个矩阵为轮廓之间的关系，大小为 轮廓个数×4
contours, _ = cv2.findContours(mp, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# 在原图绘制每一个区域
for contour in contours:
    cv2.drawContours(img, [contour], -1, (255, 0, 255), 2)
    part_area = cv2.contourArea(contour)  # 计算面积
    cv2.imshow("img", img)
    cv2.waitKey(0)


```

![image-20211224095801931](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20211224095801931.png)

### 2.3 选取区域的某一点

1. 拿边缘点当目标点

   是最简单的，但是处于，效果可能达不到预期。

2. 将质心当做目标点

   > **图像的矩(moment)** 
   >
   > (1) 一阶矩： $m_{10}$ 和 $m_{01}$ 表示用来确定图像的灰度中心 , $(\bar{x},\ \ \bar{y})$ 表示质心
   > $$
   > \bar{x} = \frac{m_{10}}{m_{00}},\ \ \ \ \ \ \bar{y}=\frac{m_{01}}{m_{00}}\\
   > m_{10} = \sum_x \sum_y xf(x,y)\\
   > m_{01} = \sum_x \sum_y yf(x,y)\\
   > $$
   > (2) 二阶矩：$m_{11}\ \ m_{02}\ \  m_{20}$  也成为惯性矩它们可以确定物体的几个特性：
   >
   > - 二阶中心矩用来确定目标物体的主轴，长轴和短轴分别对应最大和最小的二阶中心矩。可以计算主轴方向角。 
   > - 图像椭圆：由一阶、二阶矩可以确定一个与原图像惯性等价的图像椭圆。所谓图像椭圆是一个与原图像的二阶矩及原图像的灰度总和均相等的均匀椭圆。使得主轴与图像的主轴方向重合，一边分析图像性质。 
   >
   > (3) 高阶矩 ······
   >
   >  

   ```python
   # 求图像的矩，返回一个字典，{'m00': 352.0, 'm10': 130730.0, 'm01': 87776.5,···}
   M = cv2.moments(contour)   # 图像的矩
   x = int(M['m10'] // M['m00'])
   y = int(M['m01'] // M['m00'])
   cv2.circle(img, (x, y), 6, color=(64, 224, 208), thickness=-1)
   ```

   ![image-20211224103206992](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20211224103206992.png) 

   

3. 最大内接圆当做目标点

   这样既满足目标点一定在区域内，又能保证目标点处于图形的 “中心”

   > 枚举图像中的点，选择到轮廓距离最大的点

![image-20211224111212551](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20211224111212551.png) 

```python
x, y = 0, 0
# 计算到轮廓的距离
max_dist = 0   # 半径
for i in range(mp.shape[1]):
    for j in range(mp.shape[0]):
        # 点到轮廓的最大距离
        dist = cv2.pointPolygonTest(contour, (i, j), True)
        if dist > max_dist:
            #print(i, j)
            max_dist = dist
            x, y = abs(i), abs(j)
            if max_dist == 0:
                continue
                print(x, y)
                cv2.circle(img, (x, y), 6, color=(30, 144, 255), thickness=-1)
                cv2.circle(img, (x, y), int(max_dist), color=(64, 224, 208), thickness=3)
```

## 三 输出结果

### 3.1 最大面积区域

可先计算最大面积区域，再去求最大内接圆圆心，可大大优化时间复杂度。

```python
# 枚举每一个轮廓，并保存结果
part_area = cv2.contourArea(contour)
area.append([part_area, [x, y]])

# 排序输出
area.sort()
area.reverse()
print("Max area: {}".format(area[0][0]))
print("Max area point: ({}, {})".format(area[0][1][0], area[0][1][1]))
cv2.imshow("Result", img)
```





### 3.2 完整代码

含有调试代码

```python
import cv2
import numpy as np


if __name__ == "__main__":
    img_path = "imgs/img1.png"
    img = cv2.imread(img_path)
    # cv2.imshow("img", img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # 膨胀, 提升后续图像角点标注的清晰准确度
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    dst = cv2.dilate(dst, kernel)
    img[dst > 0.01*dst.max()] = [0, 0, 0]
    # cv2.imshow("mp", img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mp = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)  # 划分后的区域



    # 提取轮廓, 第一个返回值为 列表：每个轮廓边缘点
    # 第二个返回值是个矩阵为轮廓之间的关系，大小为 轮廓个数×4，
    contours, _ = cv2.findContours(mp, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)



    area = []
    for contour in contours:
        cv2.drawContours(img, [contour], -1, (255, 0, 255), 2)

        x, y = 0, 0
        # 计算到轮廓的距离
        max_dist = 0   # 半径
        for i in range(mp.shape[1]):
            for j in range(mp.shape[0]):
                # 点到轮廓的最大距离
                dist = cv2.pointPolygonTest(contour, (i, j), True)
                if dist > max_dist:
                    #print(i, j)
                    max_dist = dist
                    x, y = abs(i), abs(j)
        if max_dist == 0:
            continue

        part_area = cv2.contourArea(contour)
        # 保存结果

        area.append([part_area, [x, y]])

        print(x, y)
        cv2.circle(img, (x, y), 6, color=(30, 144, 255), thickness=-1)
        cv2.circle(img, (x, y), int(max_dist), color=(64, 224, 208), thickness=3)
		
        # 用质心当做目标点
        # M = cv2.moments(contour)   # 图像的矩
        # x = int(M['m10'] // M['m00'])
        # y = int(M['m01'] // M['m00'])
        # cv2.circle(img, (x, y), 6, color=(64, 224, 208), thickness=-1)
        # part_area = cv2.contourArea(contour)
        # # cv2.fillPoly()
        # # rect_box = cv2.minAreaRect(contour)
        # area.append([part_area, [x, y]])
        # # print(part_area, x, y)

    area.sort()
    area.reverse()
    print("Max area: {}".format(area[0][0]))
    print("Max area point: ({}, {})".format(area[0][1][0], area[0][1][1]))

    cv2.imshow("img", img)
    cv2.waitKey(0)
    exit(0)
```

## 四 扩展

### 4.1 存在的问题

对于背景平滑的，尤其是自然存在的图像识别效果好。如下：

![img5](D:\users\LiZhengli\max_area\imgs\img5.jpg) <img src="D:\users\LiZhengli\max_area\imgs\img1.png" alt="img1" style="zoom: 50%;" />

![img6](D:\users\LiZhengli\max_area\imgs\img6.png) 



但是对于人为设计的，有规律的背景识别效果差，如下

![img2](D:\users\LiZhengli\max_area\imgs\img2.png)   <img src="D:\users\LiZhengli\max_area\imgs\img4.png" alt="img4" style="zoom: 25%;" />

<img src="D:\users\LiZhengli\max_area\imgs\img3.png" alt="img3" style="zoom: 33%;" /> 



### 4.2 问题的解决方法

```python
# 调大窗口，
dst = cv2.cornerHarris(gray, 2, 15, 0.04)
```

j即使轮廓识别错误，在荧光线被识别出的前提下，背景的一些其他元素也被标注，显得不够整洁，但最大面积的轮廓位置基本固定，可以看出标注的位置是正确的。

截取100张图片，正确率为 98/100

截取图片位置：[test_imgs](test_imgs)

结果图片位置：[result_imgs](result_imgs)

| ![2](D:\Desktop\1data\max_area\test_imgs\2.png) | ![1.png](D:\Desktop\1data\max_area\result_imgs\2.png.png) |
| ----------------------------------------------- | --------------------------------------------------------- |
| ![1](D:\Desktop\1data\max_area\test_imgs\1.png) | ![1.png](D:\Desktop\1data\max_area\result_imgs\1.png.png) |

