import os
import xlrd


def get_champions():
    data = xlrd.open_workbook("Data/champions-stats-10-02.xlsx")
    table = data.sheets()[0]

    champions = []

    for i in range(1, table.nrows):
        p1 = table.row_values(i)
        champions.append([p1[1], p1[2]])

    return champions


def create_folders(champions):
    for champion in champions:
        os.mkdir("Data/champions/validation/"+str(champion[0])+"-"+str(champion[1]))


def print_form():
    champions = get_champions()
    count = 0
    for champion in champions:
        print(str(count)+":"+champion[1], end="\t")
        count += 1
        if count % 9 == 0:
            print("")


# create_folders(get_champions())
