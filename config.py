# @Time    : 2019/1/31 14:57
# @Author  : lxw
# @Email   : liuxuewen@inspur.com
# @File    : config.py
import os
import string

number = string.digits
alphabet = string.ascii_lowercase
ALPHABET = string.ascii_uppercase

char_set = number
num_classes = len(char_set)
captcha_size = 4
img_cols = 160
img_rows = 60
img_channels = 3
batch_size = 32
img_train_path = os.path.join(os.path.split(os.path.realpath(__file__))[0],'dataset', 'img_train')
img_test_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'dataset', 'img_test')

model_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'model_dir')
model_path = os.path.join(model_dir, 'mnist_tensorboard_model')
os.makedirs(model_dir, exist_ok=True)
log_dir_train = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'logs', 'train')
log_dir_test = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'logs', 'test')
os.makedirs(log_dir_train, exist_ok=True)
os.makedirs(log_dir_test, exist_ok=True)