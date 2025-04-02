import base64

class Equipo:
    def __init__(self, id, name, logo=None, players=None):
        self.id = id
        self.name = name
        self.logo = logo  # URL del logo del equipo
        self.players = players or []  # Lista de objetos Jugador

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "logo": self.logo,
            "players": [player.to_dict() for player in self.players]
        }

    def add_player(self, player):
        """Añade un jugador al equipo"""
        self.players.append(player)

    def get_player(self, player_name):
        """Obtiene un jugador por nombre (case insensitive)"""
        lower_name = player_name.lower()
        for player in self.players:
            if player.name.lower() == lower_name:
                return player
        return None

class Jugador:
    def __init__(self, id, name, photo=None):
        self.id = id
        self.name = name
        self.photo = photo  # URL de la foto del jugador
        # Estos atributos se completarán después con datos de SerpAPI
        self.age = None
        self.position = None
        self.nationality = None
        self.market_value = None
        self.stats = {}  # Diccionario para estadísticas adicionales

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "photo": self.photo,
            "age": self.age,
            "position": self.position,
            "nationality": self.nationality,
            "market_value": float(self.market_value) if self.market_value else None,
            "stats": self.stats
        }

    def update_info(self, age=None, position=None, nationality=None, 
                   market_value=None, stats=None):
        """Actualiza la información del jugador con datos adicionales"""
        if age is not None:
            self.age = age
        if position is not None:
            self.position = position
        if nationality is not None:
            self.nationality = nationality
        if market_value is not None:
            self.market_value = market_value
        if stats is not None:
            self.stats.update(stats)