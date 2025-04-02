
import os
import spacy
import subprocess
import urllib.request
from gensim.models import KeyedVectors

# Rutas de los modelos
SPACY_MODEL = "es_core_news_md"
FASTTEXT_MODEL_PATH = "cc.es.300.vec.gz"
FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.es.300.vec.gz"

def verificar_instalacion_spacy():
    """Verifica si el modelo de spaCy está instalado y lo instala si es necesario."""
    try:
        spacy.load(SPACY_MODEL)
        print(f"✅ Modelo {SPACY_MODEL} de spaCy ya está instalado.")
    except OSError:
        print(f"⚠️ Modelo {SPACY_MODEL} no encontrado. Instalando...")
        subprocess.run(["python", "-m", "spacy", "download", SPACY_MODEL])
        print(f"✅ Modelo {SPACY_MODEL} instalado correctamente.")

def verificar_instalacion_fasttext():
    """Verifica si el modelo de FastText está descargado y lo descarga si es necesario."""
    if os.path.exists(FASTTEXT_MODEL_PATH):
        print(f"✅ Modelo FastText encontrado en {FASTTEXT_MODEL_PATH}.")
    else:
        print(f"⚠️ Modelo FastText no encontrado. Descargando...")
        urllib.request.urlretrieve(FASTTEXT_URL, FASTTEXT_MODEL_PATH)
        print(f"✅ Modelo FastText descargado correctamente en {FASTTEXT_MODEL_PATH}.")

def cargar_modelo_fasttext():
    """Carga el modelo de FastText y verifica si es accesible."""
    try:
        modelo_fasttext = KeyedVectors.load_word2vec_format(FASTTEXT_MODEL_PATH, binary=False)
        print("✅ Modelo FastText cargado correctamente.")
        return modelo_fasttext
    except Exception as e:
        print(f"❌ Error al cargar el modelo FastText: {e}")
        return None

def cargar_modelos():
    """Verifica e instala los modelos necesarios."""
    verificar_instalacion_spacy()
    verificar_instalacion_fasttext()
    return spacy.load(SPACY_MODEL), cargar_modelo_fasttext()

# Si ejecutamos el script directamente, verifica e instala modelos
if __name__ == "__main__":
    nlp, fasttext_model = cargar_modelos()
    print("✅ Todos los modelos están listos para usarse.")
