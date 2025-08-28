from PyQt5.QtCore import Qt, QRect, QSize, QPoint
from PyQt5.QtWidgets import QLayout, QWidget, QSizePolicy


# class FlowLayout(QLayout):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         if parent is not None:
#             self.setContentsMargins(0, 0, 0, 0)
#         self.itemList = []
#
#     def addItem(self, item):
#         self.itemList.append(item)
#
#     def count(self):
#         return len(self.itemList)
#
#     def itemAt(self, index):
#         if 0 <= index < len(self.itemList):
#             return self.itemList[index]
#         return None
#
#     def takeAt(self, index):
#         if 0 <= index < len(self.itemList):
#             return self.itemList.pop(index)
#         return None
#
#     def setGeometry(self, rect):
#         super().setGeometry(rect)
#         x = rect.x()
#         y = rect.y()
#         lineHeight = 0
#         for item in self.itemList:
#             wid = item.widget()
#             spaceX = self.spacing()
#             spaceY = self.spacing()
#             nextX = x + item.sizeHint().width() + spaceX
#             if nextX - spaceX > rect.right() and lineHeight > 0:
#                 x = rect.x()
#                 y = y + lineHeight + spaceY
#                 nextX = x + item.sizeHint().width() + spaceX
#                 lineHeight = 0
#             item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))
#             x = nextX
#             lineHeight = max(lineHeight, item.sizeHint().height())
#
#     def sizeHint(self):
#         return self.minimumSize()
#
#     def minimumSize(self):
#         size = QSize()
#         for item in self.itemList:
#             size = size.expandedTo(item.minimumSize())
#         size += QSize(2 * self.contentsMargins().left(), 2 * self.contentsMargins().top())
#         return size

class FlowLayout(QLayout):
    def __init__(self, parent=None, margin=0, spacing=-1):
        super(FlowLayout, self).__init__(parent)

        if parent is not None:
            self.setContentsMargins(margin, margin, margin, margin)

        self.setSpacing(spacing)

        self.itemList = []

    def __del__(self):
        item = self.takeAt(0)
        while item:
            item = self.takeAt(0)

    def addItem(self, item):
        self.itemList.append(item)

    def count(self):
        return len(self.itemList)

    def itemAt(self, index):
        if index >= 0 and index < len(self.itemList):
            return self.itemList[index]

        return None

    def takeAt(self, index):
        if index >= 0 and index < len(self.itemList):
            return self.itemList.pop(index)

        return None

    def expandingDirections(self):
        return Qt.Orientations(Qt.Orientation(0))

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        height = self.doLayout(QRect(0, 0, width, 0), True)
        return height

    def setGeometry(self, rect):
        super(FlowLayout, self).setGeometry(rect)
        self.doLayout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()

        for item in self.itemList:
            size = size.expandedTo(item.minimumSize())

        margin, _, _, _ = self.getContentsMargins()

        size += QSize(2 * margin, 2 * margin)
        return size

    def doLayout(self, rect, testOnly):
        margin, _, _, _ = self.getContentsMargins()
        x = rect.x()+margin
        y = rect.y()+margin
        lineHeight = 0

        for item in self.itemList:
            wid = item.widget()
            spaceX = self.spacing() + wid.style().layoutSpacing(QSizePolicy.PushButton, QSizePolicy.PushButton, Qt.Horizontal)
            spaceY = self.spacing() + wid.style().layoutSpacing(QSizePolicy.PushButton, QSizePolicy.PushButton, Qt.Vertical)
            nextX = x + item.sizeHint().width() + spaceX
            if nextX - spaceX > rect.right() and lineHeight > 0:
                x = rect.x()+margin
                y = y + lineHeight + spaceY
                nextX = x + item.sizeHint().width() + spaceX
                lineHeight = 0

            if not testOnly:
                item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))

            x = nextX
            lineHeight = max(lineHeight, item.sizeHint().height())

        return y + lineHeight - rect.y()+margin