import sys
from Displayer import Displayer
from Reciever import Receiver
from PyQt5.QtWidgets import QApplication


def main():
    receiver = Receiver() 
    app = QApplication(sys.argv)
    display = Displayer(receiver)
    display.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
