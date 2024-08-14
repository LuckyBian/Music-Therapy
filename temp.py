import threading
import time
import _thread as thread

def test1():
    while True:
        print("11111")


def test2():
    while True:
        print("22222")


if __name__ == '__main__':
    t2 = threading.Thread(target=test1)
    #t2.setDaemon(True)
    t1 = threading.Thread(target=test2)
    #t1.setDaemon(True)

    #t1 = thread.start_new_thread(test1())

    t1.start()
    t2.start()
    #t2.join()