"""Build sirasinda EasyOCR modellerini indirir (her deploy'da ilk istekte beklememek icin)."""
import os
import easyocr

d = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".easyocr")
os.makedirs(d, exist_ok=True)
easyocr.Reader(["en"], gpu=False, model_storage_directory=d, user_network_directory=d)
print("EasyOCR modelleri indirildi:", d)
