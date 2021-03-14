import os

import numpy as np
import pyautogui
import cv2
import time
from classifier_10 import Champions10
import pickle
import copy


class DataCollector:
    data = []
    classifier = Champions10()
    coords = {}
    coord_jun = [0, 0]
    coord_jun_last = [0, 0]
    in_game = False
    is_up = False

    def __init__(self):
        self.classifier.train_model()
        self.classifier.get_generator()
        key_list = []
        coord_list = []
        for tag in self.classifier.tag.keys():
            key_list.append(tag[:tag.find("-")])
        for i in range(len(key_list)):
            coord_list.append([0, 0])
        self.coords = dict(zip(key_list, coord_list))

    def main(self):
        input_side = input("Red team (up) or blue team (down)? 1 for up, 2 for down")
        if input_side == str(1):
            self.is_up = True
        else:
            self.is_up = False

        for i in list(range(4))[::-1]:
            print(i + 1)
            time.sleep(1)

        while True:
            try:
                self.get_screen()
            except KeyboardInterrupt:
                cv2.destroyAllWindows()
                if self.data:
                    self.writefile()
                break

    def get_screen(self):
        last_time = time.time()

        time_begin = time.time()
        while True:
            screen = pyautogui.screenshot(region=[2200, 1080, 360, 360])  # x,y,w,h
            screen_numpy = np.array(screen.getdata(), dtype='uint8') \
                .reshape((screen.size[1], screen.size[0], 3))
            screen_numpy = cv2.cvtColor(screen_numpy, cv2.COLOR_BGR2RGB)
            screen_origin = copy.deepcopy(screen_numpy)
            b, g, r = cv2.split(screen_numpy)

            # binarize the three channels of img
            inranger = cv2.inRange(r, 180, 255)
            inrangeg = cv2.inRange(g, 180, 255)
            inrangeb = cv2.inRange(b, 210, 255)

            # get the blue and red scale
            red_channel = inranger - inrangeg - inrangeb
            blue_channel = inrangeb - inranger

            self.get_champions(screen_numpy, red_channel, blue_channel, self.classifier)
            time_spend = time.time() - time_begin
            if self.in_game:
                coords = copy.deepcopy(self.coords)
                self.data.append([screen_origin, coords, self.coord_jun_last, time_spend, self.coord_jun])

            print('Loop took {} seconds'.format((time.time() - last_time)))
            last_time = time.time()

    def writefile(self):
        if self.is_up:
            backup_dir = "Data/BackUp/phrase2_datas/up/" + self.classifier.jungle + str(time.time())+".txt"
        else:
            backup_dir = "Data/BackUp/phrase2_datas/down/" + self.classifier.jungle + str(time.time())+".txt"

        file_back = open(backup_dir, "wb")
        pickle.dump(self.data, file_back)

        print("Successfully added ", len(self.data), "img data to the file")

    def get_champions(self, screen_numpy, red_channel, blue_channel, classifier):
        coords = []  # list the coordinates of champions found
        champion_list = []  # list the icon of champions found
        jun_name = classifier.jungle[:classifier.jungle.find("-")]

        self.get_side_champion(screen_numpy, red_channel, coords, champion_list)
        self.get_side_champion(screen_numpy, blue_channel, coords, champion_list)

        if champion_list:
            self.in_game = True
            champion_list = np.stack(champion_list, axis=0, )
            champion_list = champion_list.reshape((champion_list.shape[0], 24, 24, 3))

            # using classifier to classify the recorded champions and print on screen
            champion_list_text = classifier.predict(champion_list)

            for n in range(len(champion_list_text)):
                if champion_list_text[n][0] == jun_name:
                    cv2.putText(screen_numpy, champion_list_text[n][0] + "-" + str(champion_list_text[n][1]),
                                (coords[n][0] - 12, coords[n][1] - 12), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0),
                                1)
                    self.coord_jun_last = self.coord_jun
                    self.coords[champion_list_text[n][0]] = self.coord_jun_last
                    self.coord_jun = [coords[n][0], coords[n][1]]
                else:
                    cv2.putText(screen_numpy, champion_list_text[n][0] + "-" + str(champion_list_text[n][1]),
                                (coords[n][0] - 12, coords[n][1] - 12), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255),
                                1)
                    self.coords[champion_list_text[n][0]] = coords[n]

                # # save the champion icons in disk
                for i in range(len(champion_list)):
                    champion_dir = "Data/new/" + champion_list_text[i][0]
                    if not os.path.exists(champion_dir):
                        os.mkdir(champion_dir)
                    img = champion_list[i]
                    img = img * 255.0
                    img = img.astype(np.float32)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(champion_dir + "/" + str(time.time()) + ".jpg", img)

            cv2.imshow("img", screen_numpy)
            cv2.waitKey(1)
        else:
            self.in_game = False



    @staticmethod
    def get_side_champion(screen_numpy, channel, coords, champion_list):
        circles = cv2.HoughCircles(channel, cv2.HOUGH_GRADIENT, 1, 10, param1=30, param2=20, minRadius=11, maxRadius=40)
        if circles is not None:
            for n in range(circles.shape[1]):
                x = int(circles[0][n][0])
                y = int(circles[0][n][1])
                coords.append([x, y])
                radius = int(circles[0][n][2])
                cropped = screen_numpy[y - radius:y + radius, x - radius:x + radius].copy()
                try:
                    to_append = cv2.resize(cropped, (24, 24))
                    to_append = cv2.cvtColor(to_append, cv2.COLOR_BGR2RGB)
                    to_append = to_append / 255.0
                    champion_list.append(to_append)
                    cv2.rectangle(screen_numpy, (x - radius, y - radius), (x + radius, y + radius), (255, 255, 255), 1)
                except cv2.error:
                    continue


d = DataCollector()
d.main()

# for i in range(len(d.data)):
#     cv2.imshow("img", d.data[i][0])
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
