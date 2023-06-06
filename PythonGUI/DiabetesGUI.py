import tkinter as tk
import customtkinter as ctk
# import liza_wia1006 as Diab
import torch
from PIL import ImageTk, Image, ImageEnhance
import joblib
import numpy as np


#To check whether commanded element works
def executed():
    print('excecuted!! var is ', var.get())
    print('text box is ', text_box.get())

# Function for making predictions for torch neural network
def predict_NN(model, dp):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert the data point to a PyTorch tensor
    dp_tensor = torch.Tensor(dp).to(device)

    # Reshape the tensor to match the expected input shape of the model
    dp_tensor = dp_tensor.view(1, -1)

    # Make a prediction
    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode
        output = model(dp_tensor)
        _, predicted = torch.max(output.data, 1)

    # Interpret the prediction
    predicted_label = predicted.item()
    if(predicted_label <= 0): return 0
    else: return 0
    pass


#Pregnancies Glucose	BP	SkinThickness	Insulin	BMI	DPF	Age	Outcome
Preg, Gluc, Bp, Skin, Ins, Bmi, Dpf, Age  = 0,0,0,0,0,0,0,0

def start():
    label.configure(text = 'Welcome to our Application to predict whether you have diabetes')

    radio1.configure(text='Polynomial Regression', command = executed)
    radio1.grid(pady = 10)
    radio2.configure(text='Decision Trees', command = executed)
    radio2.grid(pady = 10)
    radio3.configure(text='Linear Regression', command = executed)
    radio3.grid(pady = 10)
    radio4.configure(text='Logistic Regression', command = executed)
    radio4.grid(pady = 10)
    radio5.configure(text='Neural Network', command = executed)
    radio5.grid(pady = 10)
    radio6.configure(text='KNN', command = executed)
    radio6.grid(pady = 10)

    next_button.configure(text = 'next', command = page1)



    label2.grid_remove()
    set_button.grid_remove()
    text_box.grid_remove()



def page1():
    label.configure(text = 'How many pregnancies have you had?' + ' {Current:}'+ str(Preg))

    #show textbox
    text_box.delete(0, ctk.END)
    text_box.grid(pady = 10)
    label2.grid_remove()


    def set(): 
        global Preg
        Preg = int(text_box.get())
        print('preg is ', Preg)

        set_button.configure(fg_color='green')


    set_button.configure(command = set, fg_color='transparent')
    set_button.grid(pady = 10)

    next_button.configure(text = 'next', command = page2)
    next_button.grid(pady = 20)

    radio1.grid_remove()
    radio2.grid_remove()
    radio3.grid_remove()
    radio4.grid_remove()
    radio5.grid_remove()
    radio6.grid_remove()
    label2.grid_remove()



def page2():
    label.configure(text = 'What is your glucose level?' + ' {Current:}'+ str(Gluc))

    radio1.grid_remove()
    radio2.grid_remove()
    radio3.grid_remove()
    radio4.grid_remove()
    radio5.grid_remove()
    radio6.grid_remove()
    label2.grid_remove()

    #show textbox
    text_box.delete(0, ctk.END)
    text_box.grid(pady = 10)

    def set(): 

        global Gluc
        Gluc = int(text_box.get())
        print('Gluc is ', Gluc)

        set_button.configure(fg_color='green')

    set_button.configure(command = set, fg_color='transparent')
    set_button.grid(pady = 10)

    next_button.configure(text = 'next', command = page3)
    next_button.grid(padx = 20)


def page3():
    label.configure(text = 'What is your Blood Pressure?' + ' {Current:}'+ str(Bp))

    radio1.grid_remove()
    radio2.grid_remove()
    radio3.grid_remove()
    radio4.grid_remove()
    radio5.grid_remove()
    radio6.grid_remove()
    label2.grid_remove()

    #show textbox
    text_box.delete(0, ctk.END)
    text_box.grid(pady = 10)

    def set(): 
        global Bp
        Bp = int(text_box.get())
        print('Bp is ', Bp)

        set_button.configure(fg_color='green')

    set_button.configure(command = set, fg_color='transparent')
    set_button.grid(pady = 10)

    next_button.configure(text = 'next', command = page4)
    next_button.grid(padx = 20)

def page4():
    label.configure(text = 'What is your SkinThickness?' + ' {Current:}'+ str(Skin))

    radio1.grid_remove()
    radio2.grid_remove()
    radio3.grid_remove()
    radio4.grid_remove()
    radio5.grid_remove()
    radio6.grid_remove()
    label2.grid_remove()
    #show textbox
    text_box.delete(0, ctk.END)
    text_box.grid(pady = 10)

    def set(): 
        global Skin
        Skin = int(text_box.get())
        print('Skin is ', Skin)

        set_button.configure(fg_color='green')

    set_button.configure(command = set, fg_color='transparent')
    set_button.grid(pady = 10)

    next_button.configure(text = 'next', command = page5)
    next_button.grid(padx = 20)

def page5():
    label.configure(text = 'What is your Insulin level?' + ' {Current:}'+ str(Ins))

    radio1.grid_remove()
    radio2.grid_remove()
    radio3.grid_remove()
    radio4.grid_remove()
    radio5.grid_remove()
    radio6.grid_remove()
    label2.grid_remove()
    #show textbox
    text_box.delete(0, ctk.END)
    text_box.grid(pady = 10)

    def set(): 
        global Ins
        Ins = int(text_box.get())
        print('Ins is ', Ins)

        set_button.configure(fg_color='green')

    set_button.configure(command = set, fg_color='transparent')
    set_button.grid(pady = 10)

    next_button.configure(text = 'next', command = page6)
    next_button.grid(padx = 20)

def page6():
    label.configure(text = 'What is your BMI level?' + ' {Current:}'+ str(Bmi))

    radio1.grid_remove()
    radio2.grid_remove()
    radio3.grid_remove()
    radio4.grid_remove()
    radio5.grid_remove()
    radio6.grid_remove()
    label2.grid_remove()
    #show textbox
    text_box.delete(0, ctk.END)
    text_box.grid(pady = 10)

    def set(): 
        global Bmi
        Bmi = float(text_box.get())
        print('Bmi is ', Bmi)

        set_button.configure(fg_color='green')

    set_button.configure(command = set, fg_color='transparent')
    set_button.grid(pady = 10)

    next_button.configure(text = 'next', command = page7)
    next_button.grid(padx = 20)
    
def page7():
    label.configure(text = 'What is your DiabetesPedigreeFunction (DPF)?' + ' {Current:}'+ str(Dpf))

    radio1.grid_remove()
    radio2.grid_remove()
    radio3.grid_remove()
    radio4.grid_remove()
    radio5.grid_remove()
    radio6.grid_remove()
    label2.grid_remove()
    #show textbox
    text_box.delete(0, ctk.END)
    text_box.grid(pady = 10)

    def set(): 
        global Dpf
        Dpf = float(text_box.get())
        print('Dpf is ', Dpf)

        set_button.configure(fg_color='green')

    set_button.configure(command = set, fg_color='transparent')
    set_button.grid(pady = 10)

    next_button.configure(text = 'next', command = page8)
    next_button.grid(padx = 20)

def page8():
    label.configure(text = 'What is your Age?' + ' {Current:}'+ str(Age))

    radio1.grid_remove()
    radio2.grid_remove()
    radio3.grid_remove()
    radio4.grid_remove()
    radio5.grid_remove()
    radio6.grid_remove()
    label2.grid_remove()
    #show textbox
    text_box.delete(0, ctk.END)
    text_box.grid(pady = 10)

    def set(): 
        global Age
        Age = int(text_box.get())
        print('Age is ', Age)

        set_button.configure(fg_color='green')

    set_button.configure(command = set, fg_color='transparent')
    set_button.grid(pady = 10)

    next_button.configure(text = 'Show Results', command = outcome)
    next_button.grid(padx = 20)


def outcome():
    i = int(var.get())
    model_name = 'none'
    if(i == 0): model_name = 'Polynomial Regression'
    if(i == 1): model_name = 'Decision Tree Classifier'
    if(i == 2): model_name = 'Linear Regression'
    if(i == 3): model_name = 'KNN'
    if(i == 4): model_name = 'Logistic Regression'
    if(i == 5): model_name = 'Neural Network'

    label.configure(text = f'Your predicted outcome using {model_name} is:')

    #get outscaler only
    _, scaler = joblib.load('PythonGUI\\models\\model0')
    #You must rescale the input values
    arr = np.array([[Preg, Gluc, Bp, Skin, Ins, Bmi, Dpf, Age, 0]])
    arr = scaler.transform(arr)
    arr = np.delete(arr, 8, axis = 1)
    print('Scaled Inputs ',arr)

    
    if(model_name != 'Neural Network'):
        model, _ = joblib.load('PythonGUI\\models\\model'+str(i))
        pred = model.predict(arr)
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0    
    else:
        pred = predict_NN(torch.jit.load('PythonGUI\\models\\model5.pt'), arr)

    print('Prediciton is ',pred)


    #Output on Screen
    if((pred == 1)):
        sentence = 'Yes, The model predicted that the following syptoms are of Diabetic Person with type II of Vals:' +  '\n '.join(str(e) for e in [[Preg, Gluc, Bp, Skin, Ins, Bmi, Dpf, Age]])
    else:
        sentence = 'No, The model predicted that the following symptoms are not of Diabetic Person with type II of Vals:' + '\n '.join(str(e) for e in [[Preg, Gluc, Bp, Skin, Ins, Bmi, Dpf, Age]])

    label2.configure(text = sentence)
    label2.grid(row = 5, ipadx = 30)

    next_button.configure(text = 'Restart', command = start)
    next_button.grid(padx = 20)


    radio1.grid_remove()
    radio2.grid_remove()
    radio3.grid_remove()
    radio4.grid_remove()
    radio5.grid_remove()
    radio6.grid_remove()
    text_box.grid_remove()
    set_button.grid_remove()








#``````````````````````````````````````````````````````````````````````

#Setting up system settings
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

#create window
app = ctk.CTk()
print('hello')
app.geometry("560x400")
app.title("Diabetes Checking App")

image = Image.open('PythonGUI\\LIZA.png')
enhancer = ImageEnhance.Brightness(image)
bright_image = enhancer.enhance(0.5)  # Increase brightness factor
img=ImageTk.PhotoImage(bright_image, gamma=0.3)


#Creating a frame (panel)
menu = ctk.CTkFrame(app, width=530, height=50)
menu.grid(pady = 20)
frame = ctk.CTkFrame(app, width=530, height=250)
frame.grid(pady = 20)

ctk.CTkLabel(frame, image=img, width=50, height=50).place(x = 0, y = 0)


# # Adding buttons
start1 = ctk.CTkButton(menu, width=40, height=20, text='start', command=start)
start1.grid(row = 0, column = 0, padx=5)

step1 = ctk.CTkButton(menu, width=40, height=20, text='1', command = page1)
step1.grid(row = 0, column = 1, padx=5)

step2 = ctk.CTkButton(menu, width=40, height=20, text='2', command=page2)
step2.grid(row = 0, column = 2, padx=5)

step3 = ctk.CTkButton(menu, width=40, height=20, text='3', command=page3)
step3.grid(row = 0, column = 3, padx=5)

step4 = ctk.CTkButton(menu, width=40, height=20, text='4', command=page4)
step4.grid(row = 0, column = 4, padx=5)

step5 = ctk.CTkButton(menu, width=40, height=20, text='5', command=page5)
step5.grid(row = 0, column = 5, padx=5)

step6 = ctk.CTkButton(menu, width=40, height=20, text='6', command=page6)
step6.grid(row = 0, column = 6, padx=5)

step7 = ctk.CTkButton(menu, width=40, height=20, text='7', command=page7)
step7.grid(row = 0, column = 7, padx=5)

step8 = ctk.CTkButton(menu, width=40, height=20, text='8', command=page8)
step8.grid(row = 0, column = 8, padx=5)

results = ctk.CTkButton(menu, width=40, height=20, text='results', command=outcome, fg_color='orange')
results.grid(row = 0, column = 9, padx=5)





#Question Label
label = ctk.CTkLabel(frame, width=530, text='hi')
label.grid(row = 1, ipadx = 30)
label2 = ctk.CTkLabel(frame, width=530, text='result will show here', wraplength=530)

#First page Radio buttons
var = ctk.IntVar();
radio1 = ctk.CTkRadioButton(frame, text = '', variable = var, value=0)
radio2 = ctk.CTkRadioButton(frame,text = '', variable = var, value=1)
radio3 = ctk.CTkRadioButton(frame, text = '', variable = var, value=2)
radio6 = ctk.CTkRadioButton(frame, text = '', variable = var, value=3)
radio4 = ctk.CTkRadioButton(frame,text = '', variable = var, value=4)
radio5 = ctk.CTkRadioButton(frame, text = '', variable = var, value=5)

#Text box
text = ctk.StringVar() #the value taken
text_box = ctk.CTkEntry(frame, height=20, width= 100, textvariable=text, state=ctk.NORMAL)

#Set button and Next Button
set_button = ctk.CTkButton(frame, width=40, height=20, text='Set', fg_color='red')
next_button = ctk.CTkButton(frame, width=40, height=20, text='Next', fg_color='transparent')



app.mainloop()