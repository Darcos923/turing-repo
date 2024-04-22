# API DETECCION DE OBJETOS

El objetivo es disponer de un servicio que tenga como entrada una imagen y que como salida proporcione un JSON con detecciones de coches y personas.

El servicio ha sido desarrollado con [FastAPI](https://fastapi.tiangolo.com/) y poteriormente se ha creado un [Docker](https://www.docker.com/) para facilitar el despligue del servicio.

## Uso del servicio

1. Clonar el repositorio

```bash
git clone url
```

2. Instalar requerimientos de la **section-3-practical**

```bash
pip install -r requirements.txt
```

3. Lanzar el contenedor y usar el servicio mediante postman

```bash
docker compose up --build
```

En caso de querer lanzar la solicitud para probar el servicio, el script `request.py` ejecuta una solicitud que devuelve la salida con el resultado.
