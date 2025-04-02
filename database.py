from pymongo import MongoClient

class DatabaseConnection:
    """Clase Singleton para manejar la conexión a MongoDB."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseConnection, cls).__new__(cls)
            # Conexión a MongoDB Atlas
            connection_string = "mongodb+srv://jsimarrapolo:8wRVNDMWkC.6GYu@taskcluster.hixyz.mongodb.net/futnexus?retryWrites=true&w=majority"
            cls._instance.client = MongoClient(connection_string)
            cls._instance.db = cls._instance.client["futnexus"]
        return cls._instance

    def get_connection(self):
        """Devuelve la conexión a la base de datos."""
        return self.db

    def get_client(self):
        """Devuelve el cliente de MongoDB."""
        return self.client

    def close_connection(self):
        """Cierra la conexión con MongoDB."""
        if self._instance and self._instance.client:
            self._instance.client.close()
            self._instance = None