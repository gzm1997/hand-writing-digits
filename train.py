import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
from PIL import Image
import numpy
import cv2
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)



def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
x_image = tf.reshape(x, [-1,28,28,1])


h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)



W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)



keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)



W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)



cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

init = tf.global_variables_initializer()
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
saver = tf.train.Saver()
sess.run(init)


for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    print(i, "次迭代")
    #print("x", batch_xs)
    #print("y", batch_ys)
    #print("")
    sess.run(train_step, {x: batch_xs, y_: batch_ys, keep_prob: 0.5})


save_path = saver.save(sess, "./model.ckpt")


#print("accuracy", sess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 0.5}))

"""
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
    y_r = sess.run(y_conv, {x: narr, keep_prob: 0.5})
    rec_r = sess.run(tf.argmax(y_r, 1))[0]
    if rec_r == r[i]:
        count += 1
    print("检测值:", rec_r, "真实值:", r[i])

print("准确率:", count / 17)
"""