import tkinter as tk
from tkinter import messagebox
import joblib
import re
from nltk.corpus import stopwords
import nltk
from PIL import Image, ImageTk
from tkinter import ttk, messagebox

nltk.download('stopwords')

# Fonction de prétraitement des textes
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Retirer les caractères spéciaux
    text = re.sub(r'\s+', ' ', text)  # Retirer les espaces multiples
    text = ' '.join(word for word in text.split() if word not in stopwords.words('french'))
    return text

# Charger les outils sauvegardés
rf_model = joblib.load("random_forest_model_optimized.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Fonction pour la prédiction
def predict_sentiment():
    commentaire = review_entry.get("1.0", tk.END).strip()
    if not commentaire:
        messagebox.showwarning("Entrée vide", "Veuillez entrer un commentaire pour la prédiction.")
        return
    try:
        # Prétraiter le texte
        commentaire_propre = preprocess_text(commentaire)
        commentaire_tfidf = tfidf_vectorizer.transform([commentaire_propre])
        prediction_encoded = rf_model.predict(commentaire_tfidf)
        prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]
        gif_path = {
            "Positif": "thumbs-up-2584_128.gif",
            "Négatif": "Animation - 1734352727124.gif",
        }.get(prediction_label, "Animation - 1734353264961.gif")
    
        animate_gif(gif_path)
    except Exception as e:
        messagebox.showerror("Erreur", f"Une erreur est survenue : {e}")

def animate_gif(gif_path):
    """Affiche un GIF animé."""
    global gif_frames, gif_index

    gif = Image.open(gif_path)
    gif_frames = []
    try:
        while True:
            frame = ImageTk.PhotoImage(gif.copy())
            gif_frames.append(frame)
            gif.seek(len(gif_frames))
    except EOFError:
        pass

    gif_index = 0
    update_gif()

def update_gif():
    """Met à jour l'affichage d'un GIF."""
    global gif_frames, gif_index
    if gif_frames:
        sticker_label.config(image=gif_frames[gif_index])
        sticker_label.image = gif_frames[gif_index]
        gif_index = (gif_index + 1) % len(gif_frames)
        root.after(100, update_gif)

# Fenêtre principale
root = tk.Tk()
root.title("Analyse d'Avis Client")
root.geometry("500x600")
root.configure(bg="#f5f5f5")  # Couleur d'arrière-plan

# Titre principal
title_label = tk.Label(root, text="Analyse d'Avis Client", font=("Helvetica", 20, "bold"), bg="#4CAF50", fg="white", pady=10)
title_label.pack(fill=tk.X)

# Section d'entrée
frame_input = tk.Frame(root, bg="#f5f5f5")
frame_input.pack(pady=20)

instruction_label = tk.Label(frame_input, text="Entrez un avis client :", font=("Helvetica", 12), bg="#f5f5f5")
instruction_label.pack(anchor="w", padx=10, pady=5)

review_entry = tk.Text(frame_input, height=5, width=60)
review_entry.pack(padx=10, pady=5)

analyze_button = ttk.Button(frame_input, text="Analyser", command=predict_sentiment)
analyze_button.pack(pady=10)

# Section de résultat
result_label = tk.Label(root, text="Sentiment détecté: ", font=("Helvetica", 14), bg="#f5f5f5")
result_label.pack(pady=10)

sticker_label = tk.Label(root, bg="#f5f5f5")
sticker_label.pack(pady=20)

# Footer
footer_label = tk.Label(root, text="Powered by Your ML Model", font=("Helvetica", 10), bg="#4CAF50", fg="white")
footer_label.pack(fill=tk.X, side=tk.BOTTOM)

# Lancement de l'interface
root.mainloop()
