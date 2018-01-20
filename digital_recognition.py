import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
from PIL import Image
import numpy
import cv2

#从mnist数据集中加载训练和测试的数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#x用来存放图像，维度为[None, 784]，None意味着图像数量不限定，784表示每张图表示成784个点的一维向量
x = tf.placeholder(tf.float32, [None, 784])
#线性回归的参数W，维度为[784, 10]
W = tf.Variable(tf.zeros([784,10]))
#线性回归的参数b，维度为[10]
b = tf.Variable(tf.zeros([10]))
#让W和x矩阵相乘，加上b，然后计算softmax，softmax(x)=normalize(exp(x))
y = tf.nn.softmax(tf.matmul(x,W) + b)
#y_为用来存放正确的y结果
y_ = tf.placeholder("float", [None,10])
#使用交叉熵，以在后面使用梯度下降法根据交叉熵确定最佳的W和b
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#使用梯度下降法最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#初始化所有变量
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#训练1000次
for i in range(1000):
    #每次拿100张图像和标签来训练，batch_xs为训练的图像，batch_ys为对应的标签
    batch_xs, batch_ys = mnist.train.next_batch(100)
    #根据最小梯度下降法降低交叉熵
    sess.run(train_step, {x: batch_xs, y_: batch_ys})

#correct_prediction表示下面的预测是否正确
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#accuracy表示准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#使用mnist.test里面的数据进行测试，计算准确率
print("accuracy", sess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels}))


#img_arr是2维的，转为正方形
def resize_img(img_arr):
    heigh, width = img_arr.shape
    if heigh >= width:
        mark = True
        new_arr = numpy.zeros(shape = (heigh, heigh))
    else:
        mark = False
        new_arr = numpy.zeros(shape = (width, width))
    #如果高>宽 转为(heigh, heigh)
    if mark:
        i1 = int((heigh - width) / 2)
        new_arr[:, 0:i1] = 255
        new_arr[:, i1:i1 + width] = img_arr
        new_arr[:, i1 + width:] = 255
    #如果宽>高 转为(width, width)
    else:
        i1 = int((width - heigh) / 2)
        new_arr[0:i1, :] = 255
        new_arr[i1:i1 + heigh, :] = img_arr
        new_arr[i1 + heigh:, :] = 255
    return new_arr        
    

#图像中正确的数字
r = [6, 6, 6, 3, 2, 7, 5, 6, 4, 8, 1, 1, 2, 2, 3, 6, 9]
#用于记录检测准确的数量
count = 0
#拿文件夹test_digits里面的0-9图片进行测试
for i in range(17):
    #读取图像为灰度图
    img = Image.open("digits/" + str(i) + ".png").convert("L")

    arr = numpy.array(img)
    #通过补255来填补为正方形，防止数字变形
    arr = resize_img(arr)
    img = Image.fromarray(arr)
    #将补零之后的图像转为(28, 28)
    img = img.resize((28, 28), Image.ANTIALIAS)
    #获取灰度矩阵
    arr = numpy.array(img, dtype = "float32")
    #图片二值化
    ret, arr = cv2.threshold(255 - arr, 90, 255, cv2.THRESH_BINARY)
    #像素值归一化和展平为一维向量
    arr = (arr / 255).flatten()
    #进行测试，输出预测结果
    narr = numpy.zeros((1, 784))
    narr[0] = arr
    y_r = sess.run(y, {x: narr})
    rec_r = sess.run(tf.argmax(y_r, 1))[0]
    if rec_r == r[i]:
    	count += 1
    print("检测值:", rec_r, "真实值:", r[i])

print("准确率:", count / 17)






