import os
from tkinter import *
import numpy as np          # 배열 라이브러리
from PIL import ImageGrab   # 스크린 캡쳐 라이브러리
from PIL import ImageTk
from PIL import Image
import cv2                  # OpenCV, 이미지 처리 라이브러리
import yolo as yl


def start_click():
    if btn['text'] == 'start':
        btn.configure(text='reset')
    else:
        btn.configure(text='start')
        cvs.delete('all')
        return

    screenshot_img_file = os.path.join(dir_path, 'output/test.png')

    # frame screen capture. x, y, width, height
    # 스크린샷 범위에 frame 테두리는 포함되지 않도록 처리
    x = frm.winfo_rootx() + frame_bd
    y = frm.winfo_rooty() + frame_bd
    wd = frm.winfo_width() + x - frame_bd
    hg = frm.winfo_height() + y - frame_bd

    screen = ImageGrab.grab(bbox=(x, y, wd, hg))    # 스크린 캡쳐해서 변수에 저장
    #print(screen.getbbox())
    screen = np.array(screen)           # 이미지를 배열로 변환
    #cv2.imshow('window', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))   # window라는 이름의 창을 생성하고, 그 곳에 이미지를 출력함

    cv2.imwrite(screenshot_img_file, cv2.cvtColor(screen, cv2.COLOR_BGR2RGB), [cv2.IMWRITE_PNG_COMPRESSION, 100])

    yolo_directory = os.path.join(dir_path, 'yolo-coco')
    open_img_file = os.path.join(dir_path, 'input/baggage_claim.jpg')
    save_img_file = os.path.join(dir_path, 'output/test.png')
    transparent = True

    img, objdata = yl.yolo_image(screenshot_img_file, yolo_directory, transparent)

    cv2.imwrite(save_img_file, img, [cv2.IMWRITE_PNG_COMPRESSION, 100])

    image = Image.open(save_img_file)
    photoimage = ImageTk.PhotoImage(image)

    cvs.delete('all')
    cvs.configure(width=wd, height=hg)
    cvs.create_image(0, 0, image=photoimage, anchor=NW)
    cvs.image = photoimage
    
    info_window(objdata)


def test_click():
    pass


def resize_control(event):
    wd = root.winfo_width() - 3
    hg = root.winfo_height() - 60
    frm.configure(width=wd, height=hg)


def info_window(data):
    global info

    if Toplevel.winfo_exists(info) == 0:
        info = Toplevel(root)
        info.title('detail')
    else:
        info.withdraw()
        info = Toplevel(root)
        info.title('detail')

    x = root.winfo_rootx() + root.winfo_width()
    y = root.winfo_rooty() - 31
    wd = 400
    hg = root.winfo_height()

    info.geometry('%dx%d+%d+%d' % (wd, hg, x, y))

    if len(data) == 0:
        lbl = Label(info, text='출력할 데이터 없음')
        lbl.pack()
    else:
        txt = Text(info)
        for d in data:
            txt.insert('current', '%s\n' % (d))
        txt.pack()


if __name__ == '__main__':
    
    # 현재 작업 폴더
    dir_path = os.path.dirname(os.path.abspath(__file__))

    # 투명용으로 쓸 컬러 설정
    use_transparent_color = '#FF00FF'

    root = Tk()

    root.title('Object Detection')
    # width, height, xpos, ypos
    root.geometry('500x400+300+200')

    frame_bd = 2
    frm = Frame(root, relief='solid', bg=use_transparent_color, bd=frame_bd)
    frm.configure(width=400, height=250)
    frm.grid(row=0, column=0)

    cvs = Canvas(frm, bg=use_transparent_color, width=400, height=250, highlightthickness=0)
    cvs.pack()

    btn = Button(root, text='start', bg='yellow', width=20, height=2, command=start_click)
    btn.grid(row=1, column=0)

    root.attributes('-transparentcolor', use_transparent_color)
    root.bind('<Configure>', resize_control)

    info = Toplevel(root)
    info.title('detail')

    root.mainloop()
