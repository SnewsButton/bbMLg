from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog
from tkinter import scrolledtext
from PIL import ImageTk, Image
import os

window = Tk()
#To do: Make it scale
window.title("DiceChecker9000")

rbVar = IntVar()
inType1 = Radiobutton(window, text="Single Image", value=0, variable=rbVar)
inType2 = Radiobutton(window, text="Multiple Images", value=1, variable=rbVar)

def inputChoice():
    if(rbVar.get()==0):
        fBtn = Button(window, text="Browse", command=getFile)
        fBtn.grid(column=0,row=3,sticky=N+W)
    elif(rbVar.get()==1):
        fBtn = Button(window, text="Browse", command=getFiles)
        fBtn.grid(column=0,row=3,sticky=N+W)
    else:
        lbl = Label(window, text="Pick Again")
        lbl.grid(column=0,row=3)
    
def getFile():
    clearGrid()
    inFile = filedialog.askopenfilename(filetypes = (("image files",["*.png","*.jpg","*.jpeg"]),("all files","*.*")))
    img = ImageTk.PhotoImage(Image.open(inFile).resize((250, 250), Image.ANTIALIAS))
    pLbl = Label(window, image=img)
    pLbl.image = img
    pLbl.grid(column=1,row=0,rowspan=4,sticky=W)
    fLbl = Label(window, text=inFile)
    fLbl.grid(column=1,row=4,columnspan=2)
    calc = Button(window, text="Calculate", command=lambda: diceResult(inFile))
    calc.grid(column=0,row=4,sticky=N+W)

def getFiles():
    clearGrid()
    inFiles = filedialog.askopenfilenames(filetypes = (("image files",["*.png","*.jpg","*.jpeg"]),("all files","*.*")))
    print(type(inFiles))
    files = scrolledtext.ScrolledText(window)
    for line in inFiles:
        files.insert(INSERT,line+'\n')
    files.grid(column=1,row=0, rowspan=4)
    calc = Button(window, text="Calculate", command=lambda: diceResults(inFiles))
    calc.grid(column=0,row=4,sticky=N+W)

def diceResult(file):
    txt = {file:[2,10,16]}
    result = Label(window, text="Results:\n"+str(txt[file]).strip('[]'), font=("Trebuchet", 20))
    result.grid(column=2,row=0,rowspan=2,sticky=N+W)

def diceResults(files):
    txt = {"C:/Users/crash/Pictures/Congratulations2.PNG":[1,6], "C:/Users/crash/Pictures/driving.png":[13], "C:/Users/crash/Pictures/Congratulations.PNG":[20]}
    pTxt = scrolledtext.ScrolledText(window)
    pTxt.insert(INSERT,'Results:\n')
    for line in txt.items():
        pTxt.insert(INSERT,os.path.basename(line[0])+': '+str(line[1])+'\n')
    pTxt.grid(column=2,row=0, rowspan=4)
    down = Button(window, text="Download", command=lambda: download(txt))
    down.grid(column=2,row=4,sticky=N+W)

def download(txt):
    f = open(filedialog.asksaveasfilename(title="Select File",defaultextension=".txt",filetypes=(("text files","*.txt"),("all files","*.*"))),"w+")
    f.write(str(txt))
    f.close()

def clearGrid():
    for label in window.grid_slaves():
        if int(label.grid_info()["column"]) > 0:
            label.grid_forget()
        elif int(label.grid_info()["row"]) == 4:
            label.grid_forget()
    
btn = Button(window, text="Select Mode", command=inputChoice)

inType1.grid(column=0,row=0, sticky=W)
inType2.grid(column=0,row=1, sticky=W)
btn.grid(column=0,row=2, sticky=W)


#Code goes above this
window.mainloop()
