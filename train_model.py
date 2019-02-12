# @Time    : 2019/1/31 10:36
# @Author  : lxw
# @Email   : liuxuewen@inspur.com
# @File    : train_model.py
import tensorflow as tf
import time
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D, Dense, Activation, Flatten, Dropout, \
    BatchNormalization

from config import captcha_size, num_classes, img_rows, img_cols, img_channels, log_dir_train, log_dir_test, model_dir, \
    img_train_path, batch_size, img_test_path
from gen_captcha import read_captcha_text_and_image, get_next_batch


def compute_accuracy(y_pred, y_true):
    # 准确率
    y_pred_reshape = tf.reshape(y_pred, shape=(-1, captcha_size, num_classes))
    y_true_reshape = tf.reshape(y_true, shape=(-1, captcha_size, num_classes))
    argmax_pred = tf.argmax(y_pred_reshape, axis=-1)
    argmax_label = tf.argmax(y_true_reshape, axis=-1)
    correct_pred = tf.equal(argmax_pred, argmax_label)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy


def cnn_model(input):
    # 第一层卷积
    conv1 = Convolution2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same')(input)
    bath_normal1 = BatchNormalization()(conv1)
    dropout1 = Dropout(0.2)(bath_normal1)
    activate1 = Activation('relu')(dropout1)

    # 第二层卷积
    conv2 = Convolution2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(activate1)
    bath_normal2 = BatchNormalization()(conv2)
    dropout2 = Dropout(0.2)(bath_normal2)
    activate2 = Activation('relu')(dropout2)

    # 第s三层卷积
    conv3 = Convolution2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same')(activate2)
    bath_normal3 = BatchNormalization()(conv3)
    dropout3 = Dropout(0.2)(bath_normal3)
    activate3 = Activation('relu')(dropout3)
    

    # flatten
    flatten = Flatten()(activate3)
    # 第一层Dense层
    dense1 = Dense(units=512)(flatten)
    dense1_activation = Activation('relu')(dense1)
    bath_normal_dense = BatchNormalization()(dense1_activation)
    # 第二层Dense层
    dense2 = Dense(units=num_classes * captcha_size)(bath_normal_dense)
    # out = Activation('sigmoid')(dense2)
    return dense2


def my_loss(y_true, y_pred):
    # loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_true))
    return loss


def train_cnn():
    start = time.time()
    # 输入层
    img = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, img_channels))
    labels = tf.placeholder(tf.float32, shape=(None, num_classes * captcha_size))
    # 输出层
    out = cnn_model(img)
    # 损失函数
    loss = my_loss(y_true=labels, y_pred=out)
    tf.summary.scalar('loss', loss)
    # 优化器
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    # 准确率
    accuracy = compute_accuracy(y_pred=out, y_true=labels)

    tf.summary.scalar('accuracy', accuracy)
    # tensorboard融合
    merge_summary = tf.summary.merge_all()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(log_dir_train, sess.graph)
        test_writer = tf.summary.FileWriter(log_dir_test)
        step = 1
        checkpoint = tf.train.latest_checkpoint(model_dir)
        if checkpoint:
            saver.restore(sess, checkpoint)  # 从模型中读取数据
            step = int(checkpoint.split('-')[-1])
        else:
            sess.run(tf.global_variables_initializer())

        while True:
            train_start_time = time.time()
            # batch_x, batch_y = read_captcha_text_and_image(img_train_path, batch_size=batch_size)
            batch_x, batch_y = get_next_batch(img_train_path, batch_size=batch_size, train=True)
            
            merge_summary_, _, accuracy_, loss_ = sess.run([merge_summary, optimizer, accuracy, loss],
                                                           feed_dict={img: batch_x, labels: batch_y})

            train_writer.add_summary(merge_summary_, step)

            # 每100 step计算一次准确率
            if step % 10 == 0:
                batch_x_test, batch_y_test = get_next_batch(img_test_path, batch_size=batch_size, train=False)
                merge_summary_, accuracy_, loss_ = sess.run([merge_summary, accuracy, loss],
                                                            feed_dict={img: batch_x_test, labels: batch_y_test})
                test_writer.add_summary(merge_summary_, step)
                print(
                    'step:{}, loss:{}, acc:{}, time use:{}'.format(step, loss_, accuracy_,
                                                                   time.time() - train_start_time))
                # 如果准确率大于99%,保存模型,完成训练
                # if accuracy_ > 0.999:
                #     saver.save(sess, model_path, global_step=step)
                #     break
            if step >= 5000:
                break
            step += 1


if __name__ == "__main__": 
    
    # import win32process
    # import win32api
    # print('Start')
    # # 绑定到CPU 1
    # win32process.SetProcessAffinityMask(win32api.GetCurrentProcess(), 0x0007)
    train_cnn()


