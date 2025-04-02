from transformers import pipeline


# Cargar el pipeline de Question Answering
#qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

#qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Definir el contexto (puedes cambiarlo)
contexto = """ 
El aprendizaje profundo es una rama del aprendizaje automático que utiliza redes neuronales con muchas capas para modelar y extraer características complejas de los datos. Se ha aplicado con éxito en visión por computadora, procesamiento del lenguaje natural y otros campos. 
"""
from transformers import pipeline

# Cargar el modelo de Question Answering
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Contexto optimizado
contexto = ("El aprendizaje profundo es una rama del aprendizaje automático "
            "que utiliza redes neuronales con muchas capas para modelar y extraer "
            "características complejas de los datos. Se ha aplicado con éxito en "
            "visión por computadora, procesamiento del lenguaje natural y otros campos.")

# Preguntas de prueba
preguntas = [
    "Define aprendizaje profundo.",
    "Menciona algunos campos donde se aplica el aprendizaje profundo.",
    "¿Qué tipo de redes neuronales se usan en aprendizaje profundo?"
]

# 🔹 Ajustar umbral de confianza
umbral_confianza = 0.20  # Ajusta este valor según tu necesidad

print("\n📌 Contexto:\n", contexto, "\n")

# Procesar preguntas
for pregunta in preguntas:
    resultado = qa_pipeline(question=pregunta, context=contexto)
    print(f"❓ Pregunta: {pregunta}")
    print(f"✅ Respuesta: {resultado['answer']} (confianza: {resultado['score']:.2f})\n")

    # Verificar si la confianza supera el umbral
    #if resultado['score'] >= umbral_confianza:
    #   print(f"❓ Pregunta: {pregunta}")
    #   print(f"✅ Respuesta: {resultado['answer']} (confianza: {resultado['score']:.2f})\n")
    #else:
    #   print(f"❓ Pregunta: {pregunta}")
    #    print("⚠️ Respuesta descartada por baja confianza.\n")
