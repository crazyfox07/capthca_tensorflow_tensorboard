# @Time    : 2019/1/9 8:05
# @Author  : lxw
# @Email   : liuxuewen@inspur.com
# @File    : gen_captcha.py
from captcha.image import ImageCaptcha
import numpy as np
from PIL import Image
import cv2
import os
import string
import random
import time
from config import char_set, captcha_size, num_classes, img_rows, img_cols, img_channels, img_train_path, img_test_path


def random_captcha_text(char_set=char_set, captcha_size=captcha_size):
    captcha_text = random.sample(char_set, captcha_size)
    return ''.join(captcha_text)


def gen_captcha_text_and_image(img_path, batch_size=128):
    # 生成图片
    for _ in range(batch_size):
        image = ImageCaptcha()
        captcha_text = random_captcha_text()  # 生成标签
        captcha = image.generate(captcha_text)

        captcha_image = Image.open(captcha)
        captcha_image = np.array(captcha_image)
        
        img_name = captcha_text + '_' + str(int(time.time())) + '.jpg'        
        cv2.imwrite(os.path.join(img_path, img_name), captcha_image)  # 保存


def img_label_one_hot(img_label):
    y = np.zeros(num_classes * captcha_size)
    for i, ch in enumerate(img_label):
        ch_index = char_set.index(ch)
        y[i * num_classes + ch_index] = 1
    return y


def remove_captcha(img_path):
    imgs = os.listdir(img_path)
    for img in imgs:
        os.remove(os.path.join(img_path, img))


def read_captcha_text_and_image(img_path, batch_size=64):
    os.makedirs(img_path, exist_ok=True)
    remove_captcha(img_path)
    gen_captcha_text_and_image(img_path, batch_size=batch_size)
    x = np.zeros(shape=(batch_size, img_rows, img_cols, img_channels))
    y = np.zeros(shape=(batch_size, num_classes * captcha_size))
    imgs = os.listdir(img_path)
    random.shuffle(imgs)
    for i, img in enumerate(imgs):
        img_array = np.array(Image.open(os.path.join(img_path, img)))
        img_array = img_array / 255.0
        x[i:i + 1] = img_array
        img_label = img_label_one_hot(img.split('_')[0])
        y[i:i + 1] = img_label
    # x = np.transpose(x, axes=(0, 3, 1, 2))
    return x, y

imgs_train = os.listdir(img_train_path)
imgs_test = os.listdir(img_test_path)

def get_next_batch(img_path, batch_size=32, train=True):
    x = np.zeros(shape=(batch_size, img_rows, img_cols, img_channels))
    y = np.zeros(shape=(batch_size, num_classes * captcha_size))
    if train:
        imgs = random.sample(imgs_train, batch_size)
    else:
        imgs = random.sample(imgs_test, batch_size)
    for i, img in enumerate(imgs):
        img_array = np.array(Image.open(os.path.join(img_path, img)))
        img_array = img_array / 255.0
        x[i:i + 1] = img_array
        img_label = img_label_one_hot(img.split('_')[0])
        y[i:i + 1] = img_label
    # x = np.transpose(x, axes=(0, 3, 1, 2))
    return x, y


if __name__ == '__main__':
    print('begin')
    train_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', 'img_train')
    test_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', 'img_test')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    gen_captcha_text_and_image(train_path, batch_size=5000)
    gen_captcha_text_and_image(test_path, batch_size=1000)
#     read_captcha_text_and_image(train_path, batch_size=4)
    # remove_captcha(train_path)
