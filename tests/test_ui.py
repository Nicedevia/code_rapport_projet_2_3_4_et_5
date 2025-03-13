import time
import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# 📌 Configuration du navigateur Chrome pour Selenium
@pytest.fixture(scope="module")
def driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Exécution sans interface graphique
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    yield driver
    driver.quit()

# 📌 Test : Vérifier que la page Streamlit s'ouvre et que l'UI est fonctionnelle
def test_ui(driver):
    # 📌 Ouvrir l'application Streamlit
    driver.get("http://localhost:8501")  # Remplacez par l'URL de votre application si elle est déployée

    # 📌 Vérifier que le titre de la page s'affiche
    title = driver.find_element(By.TAG_NAME, "h1").text
    assert "Classification Chat/Chien" in title, "❌ Erreur : Le titre de la page ne correspond pas !"

    # 📌 Sélectionner le champ d'upload d'image et envoyer un fichier
    image_upload = driver.find_element(By.CSS_SELECTOR, "input[type='file']")
    image_upload.send_keys("data/images/cleaned/test_set/cats/cat.16.jpg")  # Remplacez par un chemin valide

    # 📌 Sélectionner le champ d'upload d'audio et envoyer un fichier
    audio_upload = driver.find_element(By.CSS_SELECTOR, "input[type='file']")
    audio_upload.send_keys("data/audio/cleaned/train/cats/cat_1.wav")  # Remplacez par un chemin valide

    # 📌 Cliquer sur le bouton "Prédire"
    predict_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Prédire')]")
    predict_button.click()

    # 📌 Attendre que la réponse apparaisse
    time.sleep(3)  # Augmentez si le modèle est lent à répondre

    # 📌 Vérifier que la prédiction s'affiche
    result_text = driver.find_element(By.CLASS_NAME, "stSuccess").text
    assert "Prédiction :" in result_text, "❌ Erreur : La prédiction ne s'affiche pas correctement !"

    print("✅ Test UI réussi : L'interface fonctionne correctement !")
