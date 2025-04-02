import fasttext
import os
import requests
import gzip

FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.es.300.bin.gz"
FASTTEXT_MODEL = "cc.es.300.bin"
FASTTEXT_GZ = "cc.es.300.bin.gz"

# --- Descarga automática del modelo preentrenado si no existe ---
def verificar_o_descargar_modelo():
    if not os.path.exists(FASTTEXT_MODEL):
        print("\n📝 Modelo preentrenado no encontrado. Iniciando descarga...")
        response = requests.get(FASTTEXT_URL, stream=True)
        with open(FASTTEXT_GZ, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print("✅ Descarga completada. Descomprimiendo...")
        with gzip.open(FASTTEXT_GZ, "rb") as f_in, open(FASTTEXT_MODEL, "wb") as f_out:
            f_out.write(f_in.read())
        print("✅ Modelo preentrenado listo para usar.")
    else:
        print("✅ Modelo preentrenado ya disponible.")

# --- Crear archivo de entrenamiento basado en equipos y jugadores ---
def crear_archivo_entrenamiento():
    data = """
    __label__equipo Real Madrid es un club de fútbol español fundado en 1902.
    __label__jugador Vinícius Júnior es un delantero brasileño que juega en el Real Madrid.
    __label__equipo Manchester City es un equipo inglés de la Premier League.
    __label__jugador Erling Haaland es un delantero noruego que juega en el Manchester City.
    __label__equipo Bayern Múnich es un equipo alemán de la Bundesliga.
    __label__jugador Joshua Kimmich es un mediocampista alemán del Bayern Múnich.
    """
    with open("data.txt", "w", encoding="utf-8") as f:
        f.write(data)

# --- Entrenar un modelo de clasificación de texto ---
def entrenar_modelo():
    print("\n[Entrenando modelo de clasificación de texto...]")
    modelo = fasttext.train_supervised(input="data.txt", epoch=25, lr=0.5, wordNgrams=3)
    modelo.save_model("modelo_clasificacion.bin")
    print("✅ Modelo de clasificación entrenado y guardado.")

# --- Probar el modelo de clasificación ---
def probar_modelo():
    modelo = fasttext.load_model("modelo_clasificacion.bin")
    print("\n[Probando el modelo de clasificación de texto...]")
    textos = [
        "Barcelona es un club de fútbol en España.",
        "Lionel Messi es un jugador argentino muy famoso.",
        "Liverpool es un equipo de la Premier League.",
        "Kylian Mbappé es un delantero francés del PSG."
    ]
    for texto in textos:
        resultado = modelo.predict(texto)
        print(f"📝 Texto: {texto} → 🔍 Predicción: {resultado}")

# --- Probar el modelo preentrenado de FastText ---
def probar_modelo_preentrenado():
    verificar_o_descargar_modelo()
    modelo_pre = fasttext.load_model(FASTTEXT_MODEL)
    print("\n[Probando modelo preentrenado de FastText...]")
    
    palabras = ["fútbol", "gol", "delantero", "entrenador"]
    
    for palabra in palabras:
        vector = modelo_pre.get_word_vector(palabra)
        print(f"🔤 Palabra: {palabra} → 🔢 Vector (primeros 5 valores): {vector[:5]}")
        
        # Obtener las 5 palabras más similares
        similares = modelo_pre.get_nearest_neighbors(palabra)
        print(f"🔍 Palabras similares a '{palabra}':")
        for score, similar_word in similares:
            print(f" - {similar_word} (similitud: {score})")

# --- Ejecución del script ---
if __name__ == "__main__":
    crear_archivo_entrenamiento()
    entrenar_modelo()
    probar_modelo()
    probar_modelo_preentrenado()