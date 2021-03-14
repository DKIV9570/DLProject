import os
import shutil

import cv2
from keras.preprocessing.image import ImageDataGenerator


def rename(string):
    main_dir = "Data/champions/train/"
    champions = os.listdir(main_dir)
    for champion in champions:
        icons_dir = os.path.join(main_dir, champion)
        icon_list = os.listdir(icons_dir)
        count = 0
        for icon in icon_list:
            os.rename(os.path.join(icons_dir, icon), os.path.join(icons_dir, string+str(count) + ".jpg"))
            count += 1
        if count < 50:
            print(champion)


def get_validation():
    train_dir = "Data/champions/train/"
    validation_dir = "Data/champions/validation/"
    champions = os.listdir(train_dir)

    for champion in champions:
        train_icons_dir = os.path.join(train_dir, champion)
        validation_icons_dir = os.path.join(validation_dir, champion)
        icon_list = os.listdir(train_icons_dir)

        for i in range(10):
            src = os.path.join(train_icons_dir, str(icon_list[i]))
            dst = validation_icons_dir
            shutil.move(src, dst+str(i)+".jpg")


def check_files():
    main_dir = "Data/champions/train/"
    champions = os.listdir(main_dir)
    for champion in champions:
        print(champion)
        icons_dir = os.path.join(main_dir, champion)
        icons = os.listdir(icons_dir)
        for icon in icons:
            print(icon)
            if int((icon.split(".")[0])) > 49:
                os.remove(os.path.join(icons_dir, icon))


def rename_folder():
    main_dir = "Data/champions/train/"
    folders = os.listdir(main_dir)
    for folder in folders:
        old_name = folder
        new_name = (folder.split("-"))[0]
        src = os.path.join(main_dir, old_name)
        dst = os.path.join(main_dir, new_name)
        os.rename(src, dst)


def generator():
    train_dir = "Data/champions/"
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')
    train_generator = train_datagen.flow_from_directory(os.path.join(train_dir, "train"),
                                                        target_size=(100, 100),
                                                        save_to_dir=os.path.join(train_dir, "test"),
                                                        shuffle=True,
                                                        batch_size=10)
    # print(train_generator.class_indices)
    # for i in range(10):
    #     train_generator.next()

    validation_dir = "Data/champions/validation/current_game"
    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                                  target_size=(24, 24),
                                                                  save_to_dir=os.path.join(train_dir, "test"),
                                                                  shuffle=True,
                                                                  batch_size=10)
    datas = []
    labels = []
    main_dir = "Data/champions/test/"
    for data_batch, label_batch in validation_generator:
        datas = data_batch
        labels = label_batch
        tags = validation_generator.class_indices
        new_tags = {v: k for k, v in tags.items()}
        count = 0

        for data in datas:
            label = labels[count].tolist()
            print(data)
            # cv2.imshow(str(new_tags[label.index(max(label))]), data)
            # cv2.waitKey(0)
            count += 1
        # return datas, labels
        break

generator()

