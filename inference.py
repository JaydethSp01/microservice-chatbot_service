from transformers import pipeline

class ModeloInferencia:
    """Clase para manejar inferencias con modelos de lenguaje."""

    def __init__(self):
        self.modelo = pipeline("question-answering", model="distilbert-base-uncased")

    def inferir_respuesta(self, pregunta, contexto):
        """Genera una respuesta basada en la pregunta y el contexto."""
        return self.modelo(question=pregunta, context=contexto)["answer"]
