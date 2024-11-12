import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc


def load_images_from_folders(main_folder):
    images = []
    labels = []
    subfolders = ['bike', 'car']  # On définit les sous-dossiers
    for subfolder in subfolders:
        folder_path = os.path.join(main_folder, subfolder)
        label = subfolder[:-1]  # "bike" pour le dossier "bikes" et "car" pour "cars"
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img = Image.open(os.path.join(folder_path, filename))
                if img is not None:
                    images.append(img)
                    labels.append(label)
    return images, labels

# Charger les données à partir du dossier data1
main_folder = 'data1'  # Remplacez par le chemin réel
images, labels = load_images_from_folders(main_folder)


print(f"Nombre d'images : {len(images)}")

formats = [img.format for img in images]
sizes = [img.size for img in images]

print(f"Formats des images : {set(formats)}")
print(f"Tailles des images : {set(sizes)}")


def visualize_random_image(images, labels):
    idx = np.random.randint(0, len(images))
    plt.imshow(images[idx])
    plt.title(f"Label: {labels[idx]}")
    plt.show()

visualize_random_image(images, labels)


def preprocess_images(images, labels, target_size=(224, 224)):
    processed_images = []
    for img in images:
        # Si l'image a de la transparence, on la convertit en RGBA
        if img.mode == 'P' or img.mode == 'RGBA':
            img = img.convert('RGBA')

        # Si l'image est en RGBA (transparence), on la convertit en RGB en enlevant la transparence
        if img.mode == 'RGBA':
            img = img.convert('RGB')

        img_resized = img.   resize(target_size)  # Redimensionner l'image
        processed_images.append(np.array(img_resized))  # Convertir en array numpy

    return np.array(processed_images), labels


# Appeler la fonction avec les images et les labels
processed_images, processed_labels = preprocess_images(images, labels)

nb_images = len(processed_images)
nb_features = processed_images[0].shape[0] * processed_images[0].shape[1] * processed_images[0].shape[2]

flattened_images = processed_images.reshape(nb_images, nb_features)

X_train, X_val, y_train, y_val = train_test_split(flattened_images, processed_labels, test_size=0.2, random_state=42)

# Entraîner l'arbre de décision
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

# Prédictions sur les données d'entraînement
y_train_pred_tree = tree_model.predict(X_train)