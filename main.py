import numpy as np

import methods

from PyQt5.QtWidgets import (QWidget, QRadioButton, QHBoxLayout,
                             QButtonGroup, QLabel)
import sys
from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd
import imutils
import os
import cv2
import glob
from PIL import Image


def start(method, data_X, data_y, X_train, y_train, n_for_test):

    n = 0
    n_for_graph = 6
    results_of_testing = []
    if (n_for_graph < n_for_test):
        n_for_graph = n_for_test
    result_for_graph = [0] * n_for_graph

    for j in range(n_for_test):
        result = method(data_X[j])
        if (j < n_for_graph):
            result_for_graph[j] = result
        results_of_testing.append(methods.classif(result, X_train, data_y[j], y_train, method))
        if (results_of_testing[-1] == data_y[j]):
            n = n + 1

    percentage = n / n_for_test
    return result_for_graph, percentage, results_of_testing

def parallel(data_X, data_y, X_train, y_train, n_for_test):
    n = 0
    n_for_graph = 6
    result = [0]*5
    results_of_testing = []
    if (n_for_graph < n_for_test):
        n_for_graph = n_for_test
    result_for_graph = [0] * n_for_graph

    for j in range(n_for_test):
        result[0] = methods.get_histogram(data_X[j])
        result[1] = methods.get_dft(data_X[j])
        result[2] = methods.get_dct(data_X[j])
        result[3] = methods.get_scale(data_X[j])
        result[4] = methods.get_gradient(data_X[j])
        if (j == 0):
            result_for_graph = result
        results_of_testing.append(methods.classif_parallel(result, X_train, data_y[j], y_train))
        if (results_of_testing[-1] == data_y[j]):
            n = n + 1
    percentage = n / n_for_test
    return result_for_graph, percentage, results_of_testing

def images(folder):
    imgs = []
    for j in '0123456789':
        img = Image.open(folder + j + '.jpg')
        im = np.array(img.convert('L'))
        if img is not None:
            imgs.append(im)
    return imgs

class Example(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        self.method = "Histogram"
        self.n_for_test = 10
        self.data_type = "test"
        self.results_for_table = []
        self.tagrets_for_table = []
        self.data_for_graph = [0]*5
        self.flag_table = 0
        self.type_of_data = "default"
        self.size_of_test_set = 0
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        vbox = QVBoxLayout()
        hbox1 = QHBoxLayout()

        bg1 = QButtonGroup(self)

        rb1 = QRadioButton("Histogram", self)
        rb1.toggled.connect(self.setHistogram)

        rb2 = QRadioButton("DFT", self)
        rb2.toggled.connect(self.setDFT)

        rb3 = QRadioButton("DCT", self)
        rb3.toggled.connect(self.setDCT)

        rb4 = QRadioButton("Scale", self)
        rb4.toggled.connect(self.setScale)

        rb5 = QRadioButton("Gradient", self)
        rb5.toggled.connect(self.setGradient)

        rb6 = QRadioButton("Parallel", self)
        rb6.toggled.connect(self.setParallel)

        hbox2 = QHBoxLayout()
        bg2 = QButtonGroup(self)

        rb7 = QRadioButton("Test sample", self)
        rb7.toggled.connect(self.setTestSample)

        rb8 = QRadioButton("Train sample", self)
        rb8.toggled.connect(self.setTrainSample)

        hbox3 = QHBoxLayout()
        bg3 = QButtonGroup(self)
        bg4 = QButtonGroup(self)

        rb9 = QRadioButton("10", self)
        rb9.toggled.connect(self.setN10)

        rb10 = QRadioButton("20", self)
        rb10.toggled.connect(self.setN20)

        rb11 = QRadioButton("30", self)
        rb11.toggled.connect(self.setN30)

        rb12 = QRadioButton("40", self)
        rb12.toggled.connect(self.setN40)

        rb13 = QRadioButton("50", self)
        rb13.toggled.connect(self.setN50)

        rb14 = QRadioButton("Default", self)
        rb14.toggled.connect(self.setDefault)

        rb15 = QRadioButton("Fawkes", self)
        rb15.toggled.connect(self.setFawkes)

        rb16 = QRadioButton("Mask", self)
        rb16.toggled.connect(self.setMask)


        self.label1 = QLabel('', self)
        self.label2 = QLabel('', self)
        self.label3 = QLabel('', self)
        self.label4 = QLabel('', self)

        bg1.addButton(rb1)
        bg1.addButton(rb2)
        bg1.addButton(rb3)
        bg1.addButton(rb4)
        bg1.addButton(rb5)
        bg1.addButton(rb6)

        bg2.addButton(rb7)
        bg2.addButton(rb8)

        bg3.addButton(rb9)
        bg3.addButton(rb10)
        bg3.addButton(rb11)
        bg3.addButton(rb12)
        bg3.addButton(rb13)

        bg4.addButton(rb14)
        bg4.addButton(rb15)
        bg4.addButton(rb16)

        hbox1.addWidget(rb1)
        hbox1.addWidget(rb2)
        hbox1.addWidget(rb3)
        hbox1.addWidget(rb4)
        hbox1.addWidget(rb5)
        hbox1.addWidget(rb6)

        hbox2.addWidget(rb7)
        hbox2.addWidget(rb8)

        hbox3.addWidget(rb9)
        hbox3.addWidget(rb10)
        hbox3.addWidget(rb11)
        hbox3.addWidget(rb12)
        hbox3.addWidget(rb13)

        hbox5 = QHBoxLayout()
        hbox5.addWidget(rb14)
        hbox5.addWidget(rb15)
        hbox5.addWidget(rb16)

        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        vbox.addLayout(hbox3)
        vbox.addLayout(hbox5)
        vbox.addWidget(self.label1)
        vbox.addWidget(self.label2)
        vbox.addWidget(self.canvas)
        vbox.addWidget(self.label3)
        vbox.addWidget(self.label4)

        rb1.setChecked(True)
        rb7.setChecked(True)
        rb9.setChecked(True)
        rb14.setChecked(True)

        pb = QPushButton("Start", self)
        pb.clicked.connect(self.push_clicked)
        pb2 = QPushButton("Show result table", self)
        pb2.clicked.connect(self.open_table)
        pb3 = QPushButton("Show statistics of parallel", self)
        pb3.clicked.connect(self.draw_plot)
        hbox4 = QHBoxLayout()
        hbox4.addWidget(pb)
        hbox4.addWidget(pb2)
        hbox4.addWidget(pb3)
        vbox.addLayout(hbox4)

        self.setLayout(vbox)

        self.setGeometry(300, 300, 800, 500)
        self.setWindowTitle('Face recognition')
        self.show()

    def setFawkes(self, value):

        rbtn = self.sender()

        if rbtn.isChecked() == True:
            self.type_of_data = "fawkes"

    def setMask(self, value):

        rbtn = self.sender()

        if rbtn.isChecked() == True:
            self.type_of_data = "mask"

    def setDefault(self, value):

        rbtn = self.sender()

        if rbtn.isChecked() == True:
            self.type_of_data = "default"

    def setN50(self, value):

        rbtn = self.sender()

        if rbtn.isChecked() == True:
            self.n_for_test = 0.5

    def setN40(self, value):

        rbtn = self.sender()

        if rbtn.isChecked() == True:
            self.n_for_test = 0.4

    def setN30(self, value):

        rbtn = self.sender()

        if rbtn.isChecked() == True:
            self.n_for_test = 0.3

    def setN20(self, value):

        rbtn = self.sender()

        if rbtn.isChecked() == True:
            self.n_for_test = 0.2

    def setN10(self, value):

        rbtn = self.sender()

        if rbtn.isChecked() == True:
            self.n_for_test = 0.1

    def setHistogram(self, value):

        rbtn = self.sender()

        if rbtn.isChecked() == True:
            self.method = methods.get_histogram

    def setDFT(self, value):

        rbtn = self.sender()

        if rbtn.isChecked() == True:
            self.method = methods.get_dft

    def setDCT(self, value):

        rbtn = self.sender()

        if rbtn.isChecked() == True:
            self.method = methods.get_dct

    def setScale(self, value):

        rbtn = self.sender()

        if rbtn.isChecked() == True:
            self.method = methods.get_scale

    def setGradient(self, value):

        rbtn = self.sender()

        if rbtn.isChecked() == True:
            self.method = methods.get_gradient

    def setParallel(self, value):

        rbtn = self.sender()

        if rbtn.isChecked() == True:
            self.method = methods.classif_parallel

    def setTestSample(self, value):

        rbtn = self.sender()

        if rbtn.isChecked() == True:
            self.data_type = "test"

    def setTrainSample(self, value):

        rbtn = self.sender()

        if rbtn.isChecked() == True:
            self.data_type = "train"

    def open_table(self):
        if (self.flag_table):
            lst = range(1, self.size_of_test_set+1)
            dict = {'number of sample': lst, 'predicted result': self.results_for_table,
                    'correct value': self.tagrets_for_table}

            df = pd.DataFrame(dict)

            fig, ax = plt.subplots()

            fig.patch.set_visible(False)
            ax.axis('off')
            ax.axis('tight')
            ax.table(cellText=df.values, colLabels=df.columns, loc='center')

            fig.tight_layout()

            plt.show()

        else:
            self.label2.setText("No results to show")

    def draw_plot(self):
        plt.plot([10, 20, 30, 40, 50], self.data_for_graph)
        plt.show()

    def push_clicked(self):
        self.flag_table = 1
        X_train, X_test, y_train, y_test = methods.preparing_data(self.n_for_test)

        if (self.type_of_data == "mask"):
            X_test = images('D:/faces_mask/')
            self.size_of_test_set = 10
        else:
            if (self.type_of_data == "fawkes"):
                X_test = images('D:/faces_fawkes/')
                self.size_of_test_set = 10
            else:
                self.size_of_test_set = 10
                #self.size_of_test_set = len(X_test)
               # print(self.size_of_test_set)


        if (self.data_type == "test"):
            data_X = X_test
            y_data = y_test
        else:
            data_X = X_train
            y_data = y_train

        self.tagrets_for_table = y_data[:self.size_of_test_set]
        if (self.method == methods.classif_parallel):
            graph, percentage, self.results_for_table = parallel(data_X, y_data, X_train, y_train, self.size_of_test_set)
            self.label2.setText("Results for one sample:")
            self.figure.clear()
            for j in range(5):
                ax = self.figure.add_subplot(2, 3, j + 1)
                if (j ==0 or j == 4):
                    ax.plot(graph[j])
                else:
                    ax.imshow(graph[j], cmap='gray') #np.log(abs(graph[j]))
                self.canvas.draw()
            n = (int)(self.n_for_test*10 - 1)
            self.data_for_graph[n] = percentage*100

        else:
            self.label2.setText("Some first results:")
            graph, percentage, self.results_for_table = start(self.method, data_X, y_data, X_train, y_train, self.size_of_test_set)
            self.figure.clear()
            for j in range(6):
                ax = self.figure.add_subplot(2, 3, j + 1)
                if (self.method == methods.get_histogram or self.method == methods.get_gradient):
                    ax.plot(graph[j])
                else:
                    ax.imshow(graph[j], cmap='gray')
                self.canvas.draw()
        percentage = percentage*100
        ans_string = '{0:0.2f}'.format(percentage)
        self.label3.setText("Percentage of recognized samples:")
        self.label4.setText(ans_string)


def main():

    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
