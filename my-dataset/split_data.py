import os
import shutil


def split_file(src_file_path, train_path, test_path):
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    if not os.path.exists(test_path):
        os.mkdir(test_path)
    dataset_len = len(os.listdir(src_file_path))
    for i, data in enumerate(os.listdir(src_file_path)):
        if i <= int(dataset_len * 0.8):
            print(os.path.join(src_file_path, data))
            shutil.copytree(os.path.join(src_file_path, data), os.path.join(train_path, data))
        else:
            shutil.copytree(os.path.join(src_file_path, data), os.path.join(test_path, data))


def split_label(src_file_path, label_path, train_path):
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    dataset_len = len(os.listdir(src_file_path))
    print(dataset_len)
    for i, data in enumerate(os.listdir(src_file_path)):
        file_name = data + '.txt'
        print(file_name)
        shutil.copy(os.path.join(label_path, file_name), os.path.join(train_path, file_name))


if __name__ == '__main__':
    split_file('images/', 'train/', 'test/')
    split_label('train/', 'labels/', 'train_label/')
    split_label('test/', 'labels/', 'test_label/')
