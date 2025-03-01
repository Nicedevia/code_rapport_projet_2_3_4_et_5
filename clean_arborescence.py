import os

def remove_empty_dirs(directory):
    """
    Parcours récursivement le répertoire donné et supprime les dossiers vides.
    """
    for root, dirs, files in os.walk(directory, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if not os.listdir(dir_path):  # Vérifie si le dossier est vide
                os.rmdir(dir_path)
                print(f"Dossier supprimé : {dir_path}")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))  # Récupère le répertoire du script
    remove_empty_dirs(project_root)
