import os
# Deshabilitar CUDA para forzar el uso de CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Desactiva oneDNN
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"  # Evita que TensorFlow intente usar GPU
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
import re
import logging
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from transformers import pipeline
from futbol_dao import EquipoDAO
from serpapi import GoogleSearch
from bson import ObjectId



# Configuración de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)

# Clave de SerpAPI (idealmente obtenida desde una variable de entorno en producción)
SERPAPI_KEY = os.environ.get("SERPAPI_KEY", "391feb88b916e9a5ae927a1145c538bd92ac85aa95afa0ec81907a6969f9692f")

# Inicialización del pipeline de QA forzando el uso de CPU (device=-1)
qa_pipeline = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad",
    max_answer_length=150,
    device=-1
)

# Cache para almacenar información de jugadores (con caducidad de 12 horas)
jugadores_cache = {}

logging.info("✅ Servicio de Chatbot iniciado con modelo de QA cargado.")

def obtener_info_jugador(jugador_id, equipo_id):
    """
    Obtiene la información completa de un jugador, consultando primero la cache,
    luego la base de datos y finalmente completando la información faltante mediante SerpAPI.
    """
    cache_key = f"{equipo_id}_{jugador_id}"
    
    # Verificar cache
    if cache_key in jugadores_cache:
        cache_entry = jugadores_cache[cache_key]
        if datetime.now() - cache_entry["timestamp"] < timedelta(hours=12):
            return cache_entry["data"]
    
    # Obtener datos base desde MongoDB
    jugador = EquipoDAO.obtener_jugador_por_id(jugador_id, equipo_id)
    if not jugador:
        return None

    # Comprobar y completar campos faltantes
    campos_requeridos = ['age', 'position', 'nationality', 'market_value']
    if not all(campo in jugador and jugador[campo] is not None for campo in campos_requeridos):
        datos_serpapi = buscar_info_serpapi(jugador['name'], jugador.get('team_name'))
        for key, value in datos_serpapi.items():
            if key not in jugador or jugador[key] is None:
                jugador[key] = value
        
        # Actualizar la base de datos con la información nueva
        EquipoDAO.actualizar_info_jugador(jugador_id, datos_serpapi)
    
    # Guardar en cache
    jugadores_cache[cache_key] = {
        "data": jugador,
        "timestamp": datetime.now()
    }
    
    return jugador

def buscar_info_serpapi(nombre_jugador, nombre_equipo=None):
    """
    Consulta SerpAPI para obtener información adicional de un jugador.
    """
    try:
        query = f"{nombre_jugador} {nombre_equipo if nombre_equipo else ''} perfil jugador estadísticas"
        params = {
            "engine": "google",
            "q": query,
            "api_key": SERPAPI_KEY,
            "gl": "es",
            "hl": "es",
            "num": 3  # Más resultados para aumentar la probabilidad de encontrar datos
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        logging.info(f"Resultados de SerpAPI: {results}")  # Depuración

        knowledge_graph = results.get('knowledge_graph', {})
        organic_results = results.get('organic_results', [])
        
        info = {
            "name": nombre_jugador,
            "age": extraer_edad(knowledge_graph) or buscar_edad_en_resultados(organic_results),
            "position": (extraer_posicion(knowledge_graph) or 
                         buscar_posicion_en_resultados(organic_results) or
                         determinar_posicion_por_nombre(nombre_jugador)),
            "nationality": (extraer_nacionalidad(knowledge_graph) or 
                            buscar_nacionalidad_en_resultados(organic_results)),
            "market_value": (extraer_valor_mercado(knowledge_graph) or 
                             buscar_valor_mercado_en_resultados(organic_results))
        }
        return {k: v for k, v in info.items() if v is not None}
        
    except Exception as e:
        logging.warning(f"Error al buscar en SerpAPI: {e}")
        return {}

def extraer_edad(knowledge_graph):
    """Extrae la edad del jugador desde el knowledge_graph, si está disponible."""
    edad = knowledge_graph.get('age')
    try:
        return int(edad) if edad else None
    except ValueError:
        return None

def extraer_posicion(knowledge_graph):
    """Extrae la posición del jugador desde el knowledge_graph."""
    return knowledge_graph.get('position')

def extraer_nacionalidad(knowledge_graph):
    """Extrae la nacionalidad del jugador desde el knowledge_graph."""
    return knowledge_graph.get('nationality')

def extraer_valor_mercado(knowledge_graph):
    """Extrae el valor de mercado del jugador desde el knowledge_graph."""
    valor = knowledge_graph.get('market_value')
    try:
        return float(valor) if valor else None
    except (ValueError, TypeError):
        return None

def buscar_edad_en_resultados(organic_results):
    """Busca la edad en los snippets de los resultados orgánicos."""
    for result in organic_results:
        snippet = result.get('snippet', '')
        # Busca un patrón que coincida con una coma seguida de 1-2 dígitos y otra coma o espacio
        match = re.search(r',\s*(\d{1,2})[, ]', snippet)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                continue
    return None

def buscar_posicion_en_resultados(organic_results):
    """Busca la posición en los snippets de los resultados orgánicos."""
    posiciones = ['delantero', 'mediocampista', 'centrocampista', 'mediocentro', 'defensa', 'portero']
    for result in organic_results:
        snippet = result.get('snippet', '').lower()
        for pos in posiciones:
            if pos in snippet:
                # Normalización: si se encuentra "mediocentro", se transforma a "Centrocampista"
                if pos == 'mediocentro':
                    return 'Centrocampista'
                return pos.capitalize()
    return None

def buscar_nacionalidad_en_resultados(organic_results):
    """Busca la nacionalidad en los snippets de los resultados orgánicos."""
    for result in organic_results:
        snippet = result.get('snippet', '')
        # Busca un patrón tipo: ", 22, España ➤ ..." o similar
        match = re.search(r',\s*\d{1,2},\s*([\w\s]+)[➤,]', snippet)
        if match:
            return match.group(1).strip()
    return None

def buscar_valor_mercado_en_resultados(organic_results):
    """Busca el valor de mercado en los snippets de los resultados orgánicos."""
    for result in organic_results:
        snippet = result.get('snippet', '')
        # Buscar "Valor de mercado: 120,00 mill" (captura números con coma)
        match = re.search(r'Valor de mercado:\s*([\d,]+)\s*mill', snippet, re.IGNORECASE)
        if match:
            try:
                valor_str = match.group(1).replace(',', '.')
                return float(valor_str)
            except ValueError:
                continue
    return None

def determinar_posicion_por_nombre(nombre_jugador):
    """Determina la posición del jugador basándose en un mapeo de nombres conocidos."""
    posiciones_conocidas = {
        'pedri': 'Centrocampista',
        'lamine yamal': 'Delantero',
        'robert lewandoski': 'Delantero',
        'gavi': 'Centrocampista',
        'ter stegen': 'Portero',
        'raphinha': 'Delantero',
        'dani olmo': 'Centrocampista',
        'pau cubarsi': 'Defensa'
    }
    return posiciones_conocidas.get(nombre_jugador.lower())

def generar_respuesta_alternativa(jugador_info):
    """Genera una respuesta básica en caso de que el modelo de QA no pueda responder la pregunta."""
    return (
        f"Información disponible: {jugador_info['name']} juega en "
        f"{jugador_info.get('team_name', 'un equipo desconocido')}. "
        f"Edad: {jugador_info.get('age', 'desconocida')}, "
        f"Posición: {jugador_info.get('position', 'desconocida')}."
    )

@app.route("/chat", methods=["POST"])
def chat():
    """
    Endpoint principal del chatbot.
    Recibe una pregunta, junto con el ID del jugador y del equipo, y devuelve la respuesta correspondiente.
    """
    try:
        datos = request.get_json()
        pregunta = datos.get("pregunta", "").strip().lower()
        jugador_id = datos.get("jugador_id")
        equipo_id = datos.get("equipo_id")
        
        logging.info(f"📩 Pregunta recibida: '{pregunta}' (Jugador: {jugador_id}, Equipo: {equipo_id})")
        
        # Validación de parámetros
        if not pregunta or not jugador_id or not equipo_id:
            return jsonify({
                "error": "Faltan parámetros requeridos",
                "ejemplo": {
                    "pregunta": "¿Cuál es la posición de este jugador?",
                    "jugador_id": "67ec8e3788568f4b9ad4227f",
                    "equipo_id": "67ec8e3788568f4b9ad4227e"
                }
            }), 400
        
        # Validar formato de IDs
        if not ObjectId.is_valid(jugador_id) or not ObjectId.is_valid(equipo_id):
            return jsonify({"error": "Los IDs proporcionados no tienen un formato válido"}), 400
        
        # Obtener información del jugador
        jugador_info = obtener_info_jugador(jugador_id, equipo_id)
        if not jugador_info:
            return jsonify({"error": "Jugador no encontrado"}), 404
        
        # Construir contexto para el modelo de QA a partir de la información del jugador
        contexto = (
            f"Nombre: {jugador_info['name']}. "
            f"Equipo: {jugador_info.get('team_name', 'Desconocido')}. "
            f"Edad: {jugador_info.get('age', 'Desconocida')}. "
            f"Posición: {jugador_info.get('position', 'Desconocida')}. "
            f"Nacionalidad: {jugador_info.get('nationality', 'Desconocida')}. "
            f"Valor de mercado: {jugador_info.get('market_value', 'Desconocido')}."
        )
        
        logging.info(f"📜 Contexto generado: {contexto}")
        
        # Caso especial: Si la pregunta menciona "messi" y el contexto no lo incluye,
        # se devuelve una respuesta indicando que esa consulta no está soportada actualmente.
        if "messi" in pregunta and "messi" not in contexto.lower():
            respuesta = (
                "Lo siento, actualmente no cuento con información para responder a esa pregunta en mi modelo. "
                "Sin embargo, dispongo de datos sobre la edad, posición, nacionalidad y valor de mercado del jugador."
            )
            return jsonify({"respuesta": respuesta, "fuente": "modelo_no_soporta"})
        
        # Respuestas directas para ciertas preguntas
        if "edad" in pregunta and jugador_info.get('age'):
            respuesta = f"{jugador_info['name']} tiene {jugador_info['age']} años."
            return jsonify({"respuesta": respuesta, "fuente": "datos_directos"})
        elif "posición" in pregunta and jugador_info.get('position'):
            respuesta = f"{jugador_info['name']} juega como {jugador_info['position']}."
            return jsonify({"respuesta": respuesta, "fuente": "datos_directos"})
        elif "nacionalidad" in pregunta and jugador_info.get('nationality'):
            respuesta = f"{jugador_info['name']} es de nacionalidad {jugador_info['nationality']}."
            return jsonify({"respuesta": respuesta, "fuente": "datos_directos"})
        elif "valor" in pregunta and jugador_info.get('market_value'):
            respuesta = (
                f"El valor de mercado de {jugador_info['name']} es de aproximadamente "
                f"{jugador_info['market_value']} millones de euros."
            )
            return jsonify({"respuesta": respuesta, "fuente": "datos_directos"})
        
        # Utilizar el modelo de QA para preguntas complejas si hay suficiente contexto
        if len(contexto.split()) > 10:
            try:
                qa_result = qa_pipeline({"question": pregunta, "context": contexto})
                if qa_result.get("score", 0) > 0.5:
                    return jsonify({
                        "respuesta": qa_result["answer"],
                        "contexto_utilizado": contexto,
                        "fuente": "modelo_qa"
                    })
            except Exception as e:
                logging.warning(f"Error en el modelo de QA: {e}")
        
        # Respuesta alternativa si no se obtuvo respuesta mediante QA
        return jsonify({
            "respuesta": generar_respuesta_alternativa(jugador_info),
            "fuente": "informacion_basica"
        })
            
    except Exception as e:
        logging.error(f"Error en el endpoint /chat: {e}")
        return jsonify({"error": "Error interno del servidor"}), 500

if __name__ == "__main__":
    # Obtener el puerto desde la variable de entorno PORT (necesario para Render) o usar 5001 por defecto
    port = int(os.environ.get("PORT", 5001))
    logging.info(f"🚀 Chatbot Service iniciado en http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
