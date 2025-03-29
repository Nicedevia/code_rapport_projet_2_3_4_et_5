import os
import time
import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# Configuration du webdriver Chrome 
@pytest.fixture(scope="module")
def driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    yield driver
    driver.quit()

# URL de l'application Streamlit
APP_URL = "http://localhost:8501"

IMAGE_FILE = os.path.abspath("data/images/cleaned/test_set/cats/cat.16.jpg")
AUDIO_FILE = os.path.abspath("data/audio/cleaned/test/cats/cat_3.wav")

def test_title_presence(driver):
    """Vérifie que le titre de l'application est affiché correctement."""
    driver.get(APP_URL)
    time.sleep(2)  
    title = driver.find_element(By.TAG_NAME, "h1").text
    assert "Classification Chat / Chien" in title, "Le titre de l'application ne correspond pas."

def test_image_upload(driver):
    """Teste l'upload d'une image et vérifie l'affichage d'un message de succès."""
    driver.get(APP_URL)
    time.sleep(2)
    file_inputs = driver.find_elements(By.CSS_SELECTOR, "input[type='file']")
    assert len(file_inputs) >= 2, "Nombre insuffisant de champs d'upload détectés."

    image_input = file_inputs[0]
    image_input.send_keys(IMAGE_FILE)
    time.sleep(1)

    success_msgs = driver.find_elements(By.XPATH, "//*[contains(text(), 'Téléchargement OK')]")
    assert len(success_msgs) > 0, "Message de succès introuvable après l'upload de l'image."

def test_audio_upload(driver):
    """Teste l'upload d'un audio et vérifie l'affichage d'un message de succès."""
    driver.get(APP_URL)
    time.sleep(2)
    file_inputs = driver.find_elements(By.CSS_SELECTOR, "input[type='file']")
    assert len(file_inputs) >= 2, "Nombre insuffisant de champs d'upload détectés."

    audio_input = file_inputs[1]
    audio_input.send_keys(AUDIO_FILE)
    time.sleep(1)

    success_msgs = driver.find_elements(By.XPATH, "//*[contains(text(), 'Téléchargement OK')]")
    assert len(success_msgs) > 0, "Message de succès introuvable après l'upload de l'audio."
