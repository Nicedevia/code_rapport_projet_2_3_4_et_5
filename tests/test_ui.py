import os
import time
import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# Configuration du webdriver Chrome en mode headless
@pytest.fixture(scope="module")
def driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Exécution sans interface graphique
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    yield driver
    driver.quit()

# URL de l'application Streamlit
APP_URL = "http://localhost:8501"

# Chemins vers des fichiers de test (assurez-vous que ces fichiers existent)
IMAGE_FILE = os.path.abspath("data/images/cleaned/test_set/cats/cat.16.jpg")
AUDIO_FILE = os.path.abspath("data/audio/cleaned/test/cats/cat_1.wav")

def test_title_presence(driver):
    """Vérifie que le titre de l'application est affiché correctement."""
    driver.get(APP_URL)
    time.sleep(2)  # Attente du rendu de la page
    title = driver.find_element(By.TAG_NAME, "h1").text
    assert "Classification Chat / Chien" in title, "Le titre de l'application ne correspond pas."

def test_image_upload(driver):
    """Teste l'upload d'une image et vérifie l'affichage d'un message de succès."""
    driver.get(APP_URL)
    time.sleep(2)
    file_inputs = driver.find_elements(By.CSS_SELECTOR, "input[type='file']")
    assert len(file_inputs) >= 2, "Nombre insuffisant de champs d'upload détectés."
    # Premier input : upload d'image
    image_input = file_inputs[0]
    image_input.send_keys(IMAGE_FILE)
    time.sleep(1)
    # Vérifier la présence d'un message de succès (par exemple "Téléchargement OK")
    success_msgs = driver.find_elements(By.XPATH, "//*[contains(text(), 'Téléchargement OK')]")
    assert len(success_msgs) > 0, "Message de succès introuvable après l'upload de l'image."

def test_audio_upload(driver):
    """Teste l'upload d'un audio et vérifie l'affichage d'un message de succès."""
    driver.get(APP_URL)
    time.sleep(2)
    file_inputs = driver.find_elements(By.CSS_SELECTOR, "input[type='file']")
    assert len(file_inputs) >= 2, "Nombre insuffisant de champs d'upload détectés."
    # Second input : upload d'audio
    audio_input = file_inputs[1]
    audio_input.send_keys(AUDIO_FILE)
    time.sleep(1)
    # Vérifier la présence d'un message de succès
    success_msgs = driver.find_elements(By.XPATH, "//*[contains(text(), 'Téléchargement OK')]")
    assert len(success_msgs) > 0, "Message de succès introuvable après l'upload de l'audio."

def test_prediction(driver):
    """Vérifie que la prédiction s'exécute correctement après upload."""
    driver.get(APP_URL)
    time.sleep(2)
    file_inputs = driver.find_elements(By.CSS_SELECTOR, "input[type='file']")
    # Upload image et audio
    file_inputs[0].send_keys(IMAGE_FILE)
    file_inputs[1].send_keys(AUDIO_FILE)
    time.sleep(1)
    # Cliquer sur le bouton "🔮 Prédire"
    predict_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Prédire')]")
    predict_button.click()
    time.sleep(3)  # Attente du traitement de la prédiction
    # Vérifier l'apparition d'un message indiquant la prédiction
    prediction_msgs = driver.find_elements(By.XPATH, "//*[contains(text(), 'Prédiction :')]")
    assert len(prediction_msgs) > 0, "La prédiction n'est pas affichée après le clic sur le bouton."

def test_change_image(driver):
    """Teste le bouton permettant de changer l'image affichée."""
    driver.get(APP_URL)
    time.sleep(2)
    # Récupérer la source de l'image affichée
    initial_img = driver.find_element(By.XPATH, "//img[contains(@src, 'data:image')]")
    initial_src = initial_img.get_attribute("src")
    # Cliquer sur le bouton "🔄 Changer l'image"
    change_image_button = driver.find_element(By.XPATH, "//button[contains(text(), \"Changer l'image\")]")
    change_image_button.click()
    time.sleep(2)
    # Vérifier que l'image affichée a changé
    new_img = driver.find_element(By.XPATH, "//img[contains(@src, 'data:image')]")
    new_src = new_img.get_attribute("src")
    assert initial_src != new_src, "L'image n'a pas changé après le clic sur 'Changer l'image'."

def test_change_audio(driver):
    """Teste le bouton permettant de changer l'audio affiché."""
    driver.get(APP_URL)
    time.sleep(2)
    # Récupérer la source audio affichée
    initial_audio = driver.find_element(By.TAG_NAME, "audio")
    initial_src = initial_audio.get_attribute("src")
    # Cliquer sur le bouton "🔄 Changer l'audio"
    change_audio_button = driver.find_element(By.XPATH, "//button[contains(text(), \"Changer l'audio\")]")
    change_audio_button.click()
    time.sleep(2)
    # Vérifier que la source audio a changé
    new_audio = driver.find_element(By.TAG_NAME, "audio")
    new_src = new_audio.get_attribute("src")
    assert initial_src != new_src, "L'audio n'a pas changé après le clic sur 'Changer l'audio'."
