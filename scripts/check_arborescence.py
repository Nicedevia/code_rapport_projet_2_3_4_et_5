import os

PROJECT_ROOT = os.getcwd()

IGNORED_DIRS = {".git", ".idea", "__pycache__"}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
AUDIO_EXTENSIONS = (".wav",)

def list_directory_structure(root_dir, output_file="arborescence.txt"):
    """
    Génère l'arborescence des dossiers et fichiers en excluant :
      - les dossiers inutiles (définis dans IGNORED_DIRS et les dossiers commençant par '.')
      - les fichiers d'images et audios, qui sont seulement comptés.
    Le résultat est écrit dans le fichier output_file.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for root, dirs, files in os.walk(root_dir):
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRS and not d.startswith(".")]

            level = root.replace(root_dir, "").count(os.sep)
            indent = " " * 4 * level
            folder_name = os.path.basename(root) if os.path.basename(root) else root
            f.write(f"{indent}📂 {folder_name}/\n")

            sub_indent = " " * 4 * (level + 1)

            # Compter les fichiers images et audios
            wav_count = sum(1 for file in files if file.lower().endswith(AUDIO_EXTENSIONS))
            png_count = sum(1 for file in files if file.lower().endswith(".png"))
            jpg_count = sum(1 for file in files if file.lower().endswith((".jpg", ".jpeg")))

            for file in files:
                if file.startswith("."):
                    continue
                if file.lower().endswith(AUDIO_EXTENSIONS + (".png",) + (".jpg", ".jpeg")):
                    continue
                f.write(f"{sub_indent}📄 {file}\n")

            # Afficher les comptes pour les fichiers images et audios
            if wav_count > 0:
                f.write(f"{sub_indent}🎵 {wav_count} fichiers .wav\n")
            if png_count > 0:
                f.write(f"{sub_indent}🖼 {png_count} fichiers .png\n")
            if jpg_count > 0:
                f.write(f"{sub_indent}📸 {jpg_count} fichiers .jpg/.jpeg\n")
    print(f"✅ Arborescence générée dans {output_file}")

if __name__ == "__main__":
    list_directory_structure(PROJECT_ROOT)
