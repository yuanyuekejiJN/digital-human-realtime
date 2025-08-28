import os

import time

from PyQt5.QtWidgets import *
from PyQt5.QtWebEngineWidgets import *
from PyQt5.QtCore import *
from PyQt5 import QtWidgets

from scheduler.thread_manager import MyThread


class MainWindow(QMainWindow):
    SigSendMessageToJS = pyqtSignal(str)

    # 图形化界面的初始化方法
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle('元岳科技')
        self.setGeometry(0, 0, 16 * 70, 9 * 70)
        self.showMaximized()
        self.browser = QWebEngineView()
        #清空缓存
        profile = QWebEngineProfile.defaultProfile()
        profile.clearHttpCache()

        # 指定接口地址
        self.browser.load(QUrl('http://127.0.0.1:5000'))
        self.setCentralWidget(self.browser)
        MyThread(target=self.runnable).start()

    def runnable(self):
        while True:
            if not self.isVisible():
                os.system("taskkill /F /PID {}".format(os.getpid()))
            time.sleep(0.05)

    def center(self):
        screen = QtWidgets.QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) / 2, (screen.height() - size.height()) / 2)

