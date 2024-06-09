import customtkinter as ctk
import joblib
import pandas as pd
import numpy as np
from PIL import Image, ImageTk

model = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

user_input_features = [
    'n_tokens_title', 'n_tokens_content', 'num_hrefs', 'num_self_hrefs',
    'num_imgs', 'num_videos', 'average_token_length', 'num_keywords',
    'data_channel_is_lifestyle', 'data_channel_is_entertainment',
    'data_channel_is_bus', 'data_channel_is_socmed',
    'data_channel_is_tech', 'data_channel_is_world',
    'global_subjectivity', 'global_sentiment_polarity',
    'title_subjectivity', 'title_sentiment_polarity'
]

user_input_features1 = [
    "Title Word Count", "Content Word Count", "Number of Hyperlinks", "Number of Internal Links",
    "Number of Images", "Number of Videos", "Average Word Length", "Number of Keywords", 1,2,3,4,5,6,
    "Global Subjectivity", "Global Sentiment Polarity",
    "Title Subjectivity", "Title Sentiment Polarity"
]

channel_features = [
    'data_channel_is_lifestyle', 'data_channel_is_entertainment',
    'data_channel_is_bus', 'data_channel_is_socmed',
    'data_channel_is_tech', 'data_channel_is_world'
]

channel_features1 = ["Lifestyle", "Entertainment", "Business", "Social Media", "Tech", "World"]

channel_values = {channel: 0 for channel in channel_features}

def predict_viral_content(user_input):
    user_input_df = pd.DataFrame([user_input], columns=user_input_features)
    user_input_scaled = scaler.transform(user_input_df)
    prediction = model.predict(user_input_scaled)
    probability = model.predict_proba(user_input_scaled)[0][1]
    return prediction[0], probability

def get_prediction():
    user_input = []
    for feature in user_input_features:
        if feature in channel_features:
            user_input.append(channel_values[feature])
        else:
            user_input.append(float(entries[feature].get()))
    
    is_viral, viral_probability = predict_viral_content(user_input)
    viral_probability = round(viral_probability * 100, 2)  # Round to two decimal places
    result_label.configure(text=f"Is the content viral? {'Yes' if is_viral else 'No'}\nProbability of being viral: {viral_probability:.2f}%")
    update_image(is_viral)

def update_image(is_viral):
    if is_viral:
        img = Image.open("gambar/viral.jpg")
    else:
        img = Image.open("gambar/not_viral.jpg")
    
    img = img.resize((300, 300), Image.LANCZOS) 
    img = ImageTk.PhotoImage(img)
    image_label.configure(image=img)
    image_label.image = img  

def on_channel_select(channel):
    for key in channel_values:
        channel_values[key] = 0
    channel_values[channel_features[channel_features1.index(channel)]] = 1

root = ctk.CTk()
root.title("Prediksi Konten Viral")
root.geometry("800x600")
root.resizable(False, False)

entries = {}
for idx, feature in enumerate(user_input_features):
    row = idx // 2
    column = idx % 2
    if feature not in channel_features:
        label = ctk.CTkLabel(root, text=user_input_features1[idx] + ":", font=("Arial", 12))
        label.grid(row=row, column=2*column, padx=10, pady=5, sticky='e')
        entry = ctk.CTkEntry(root, width=200)
        entry.grid(row=row, column=2*column+1, padx=10, pady=5, sticky='w')
        entries[feature] = entry

label = ctk.CTkLabel(root, text="Data Channel:", font=("Arial", 12))
label.grid(row=len(user_input_features)//2, column=0, padx=10, pady=5, sticky='e')
selected_channel = ctk.StringVar()
channel_dropdown = ctk.CTkOptionMenu(root, variable=selected_channel, values=channel_features1, command=on_channel_select)
channel_dropdown.grid(row=len(user_input_features)//2, column=1, padx=10, pady=5, sticky='w')

predict_button = ctk.CTkButton(root, text="Predict", command=get_prediction, font=("Arial", 12))
predict_button.grid(row=len(user_input_features)//2 + 1, column=0, columnspan=2, pady=20)

result_label = ctk.CTkLabel(root, text="", font=("Arial", 12))
result_label.grid(row=len(user_input_features)//2 + 2, column=0, columnspan=2, pady=10)

image_label = ctk.CTkLabel(root, text="")
image_label.grid(row=len(user_input_features)//2 + 2, column=3, columnspan=2, pady=20)

root.mainloop()
