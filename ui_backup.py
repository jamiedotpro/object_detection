import os
from tkinter import Tk, Frame, Canvas
from PIL import ImageTk

root = Tk()
root.title('Object Detection')
# width, height, xpos, ypos
root.geometry('500x400+300+200')

frm = Frame(root, relief='solid', bg='#FF00FF', bd=2)
frm.configure(width=400, height=250)
frm.grid(row=0, column=0)

canvas = Canvas(frm, bg="#FF00FF", width=500, height=500)
canvas.pack()

dir_path = os.path.dirname(os.path.abspath(__file__))
png_file = os.path.join(dir_path, 'output/test.png')

photoimage = ImageTk.PhotoImage(file=png_file)
canvas.create_image(150, 150, image=photoimage)

root.attributes('-transparentcolor', '#FF00FF')
root.mainloop()