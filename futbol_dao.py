from database import DatabaseConnection
from bson import ObjectId
from serpapi import GoogleSearch
import os
import re
from datetime import datetime, timedelta
# Deshabilitar CUDA para forzar el uso de CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

class EquipoDAO:
    """Clase de acceso a datos para equipos y jugadores en MongoDB."""

    # Configuración de SerpAPI
    SERPAPI_KEY = os.getenv("SERPAPI_KEY", "391feb88b916e9a5ae927a1145c538bd92ac85aa95afa0ec81907a6969f9692f")

    @staticmethod
    def obtener_equipos():
        """Consulta todos los equipos."""
        try:
            db = DatabaseConnection().get_connection()
            equipos = list(db.images_ligas.find({}, {"_id": 1, "name": 1, "logo": 1}))
            for equipo in equipos:
                equipo["id"] = str(equipo.pop("_id"))
            return equipos
        except Exception as e:
            print(f"❌ Error al obtener equipos: {e}")
            return []

    @staticmethod
    def obtener_equipo_por_id(equipo_id):
        """Obtiene un equipo específico por su ID."""
        try:
            db = DatabaseConnection().get_connection()
            equipo = db.images_ligas.find_one({"_id": ObjectId(equipo_id)})
            if equipo:
                equipo["id"] = str(equipo.pop("_id"))
                if "players" in equipo:
                    for jugador in equipo["players"]:
                        jugador["id"] = str(jugador.pop("_id"))
                return equipo
            return None
        except Exception as e:
            print(f"❌ Error al obtener equipo por ID: {e}")
            return None

    @staticmethod
    def obtener_jugadores_por_equipo(equipo_id):
        """Obtiene los jugadores de un equipo específico."""
        try:
            db = DatabaseConnection().get_connection()
            equipo = db.images_ligas.find_one(
                {"_id": ObjectId(equipo_id)},
                {"players": 1}
            )
            if equipo and "players" in equipo:
                for jugador in equipo["players"]:
                    jugador["id"] = str(jugador.pop("_id"))
                return equipo["players"]
            return []
        except Exception as e:
            print(f"❌ Error al obtener jugadores por equipo: {e}")
            return []

    @staticmethod
    def obtener_jugador_por_id(jugador_id, equipo_id=None):
        """Obtiene un jugador por ID, opcionalmente filtrado por equipo."""
        try:
            db = DatabaseConnection().get_connection()
            
            if equipo_id:
                # Búsqueda más eficiente dentro de un equipo específico
                equipo = db.images_ligas.find_one(
                    {"_id": ObjectId(equipo_id), "players._id": ObjectId(jugador_id)},
                    {"players.$": 1, "name": 1}
                )
            else:
                # Búsqueda global en todos los equipos
                equipo = db.images_ligas.find_one(
                    {"players._id": ObjectId(jugador_id)},
                    {"players.$": 1, "name": 1}
                )

            if equipo and "players" in equipo and len(equipo["players"]) > 0:
                jugador = equipo["players"][0]
                jugador["id"] = str(jugador.pop("_id"))
                jugador["team_id"] = str(equipo["_id"])
                jugador["team_name"] = equipo["name"]
                return jugador
            return None
        except Exception as e:
            print(f"❌ Error al obtener jugador por ID: {e}")
            return None

    @staticmethod
    def actualizar_info_jugador(jugador_id, nuevos_datos):
        """Actualiza información de un jugador en la base de datos."""
        try:
            db = DatabaseConnection().get_connection()
            resultado = db.images_ligas.update_one(
                {"players._id": ObjectId(jugador_id)},
                {"$set": {f"players.$.{k}": v for k, v in nuevos_datos.items()}}
            )
            return resultado.modified_count > 0
        except Exception as e:
            print(f"❌ Error al actualizar jugador: {e}")
            return False

    @staticmethod
    def buscar_info_serpapi(nombre_jugador, equipo=None):
        """Busca información del jugador en SerpAPI."""
        try:
            query = f"{nombre_jugador} {equipo if equipo else ''} footballer"
            params = {
                "engine": "google",
                "q": query,
                "api_key": EquipoDAO.SERPAPI_KEY,
                "gl": "es",
                "hl": "es"
            }
            
            search = GoogleSearch(params)
            results = search.get_dict()
            
            return {
                "age": EquipoDAO._extraer_edad(results),
                "position": EquipoDAO._extraer_posicion(results),
                "nationality": EquipoDAO._extraer_nacionalidad(results),
                "market_value": EquipoDAO._extraer_valor_mercado(results)
            }
        except Exception as e:
            print(f"⚠️ Error al buscar en SerpAPI: {e}")
            return {}

    @staticmethod
    def _extraer_edad(results):
        """Extrae edad de los resultados de SerpAPI."""
        knowledge_graph = results.get('knowledge_graph', {})
        if 'age' in knowledge_graph:
            age_str = knowledge_graph['age']
            if isinstance(age_str, str):
                match = re.search(r'(\d+)', age_str)
                if match:
                    return int(match.group(1))
        return None

    @staticmethod
    def _extraer_posicion(results):
        """Extrae posición del jugador."""
        knowledge_graph = results.get('knowledge_graph', {})
        return knowledge_graph.get('position')

    @staticmethod
    def _extraer_nacionalidad(results):
        """Extrae nacionalidad del jugador."""
        knowledge_graph = results.get('knowledge_graph', {})
        return knowledge_graph.get('nationality')

    @staticmethod
    def _extraer_valor_mercado(results):
        """Extrae valor de mercado del jugador."""
        knowledge_graph = results.get('knowledge_graph', {})
        if 'market_value' in knowledge_graph:
            value_str = knowledge_graph['market_value']
            if isinstance(value_str, str):
                clean_value = re.sub(r'[^\d.]', '', value_str)
                try:
                    return float(clean_value)
                except ValueError:
                    return None
        return None