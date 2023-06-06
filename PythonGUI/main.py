import tkinter
import customtkinter
from pytube import YouTube

def startDownload():
    try:
        ytlink = link.get()
        ytObject = YouTube(ytlink)
        video = ytObject.streams.get_lowest_resolution()
        video.download()
    except:
        print("Youtibe link is invalid")
    print("Process Finished")



# Setting up System settings
customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("blue")


#Our app frame
app = customtkinter.CTk()
app.geometry("780x420")
app.title("Youtube downloader")


#adding ui elements
title = customtkinter.CTkLabel(app, text = "testing")
title.pack(padx = 10, pady = 10)

def label_clicked(event):
    print("Label clicked!")

title.bind("<Button-1>", label_clicked)

#link input
url_holder = tkinter.StringVar()
link = customtkinter.CTkEntry(app, width=300, height=10, textvariable=url_holder)
link.pack()

#button
button = customtkinter.CTkButton(app, width=20, height=20, command=startDownload)
button.pack()

app.mainloop()
