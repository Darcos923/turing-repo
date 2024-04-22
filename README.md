## Reto Data Scientist

El reto estará compuesto de 3 apartados. Un apartado técnico, un apartado teórico y un apartado opcional.

El objetivo del apartado técnico es valorar la capacidad de buscar una solución a un problema en un tiempo limitado y que ésta de la salida especificada. No se plantea evaluar que la solución sea optima en ninguna métrica, es decir, ni tiempo de ejecución, ni precisión de la inferencia.

El objetivo del apartado teórico es realizar un pequeño guion de los pasos necesarios a realizar para resolver el reto, así como detectar y comentar los posibles problemas que podrían aparecer y plantear contramedidas para minimizar los riesgos.

El objetivo del apartado opcional es simplemente una oportunidad para demostrar el conocimiento en otras tecnologías relevantes.

> **Apartado 1:** _Crear un chatbot que tenga las siguientes funcionalidades:_

1. Una interfaz, por ejemplo la interfaz de chatbot de Gradio.
2. Se han de ingestar varios documentos PDF largos para usarlos como base de conocimiento de un RAG. Se ha de usar una base de datos vectorial a elección.
3. Se ha de implementar una memoria dinámica que mantenga la conversación y que cuendo esta pase de X tokens se resuma de forma automática.
4. La implementación ha de estar basada en langchain.
5. Si se detecta una pregunta que necesite de exactitud en la respuesta el modelo ha ser capaz de implementar y ejecutar código python.

> **Apartado 2:** _Dar respuesta a los siguientes puntos de forma teórica, sin necesidad de desarrollarlos, que guardan relación con las tecnologías utilizadas en el primer apartado:_

1. Diferencias entre 'completion' y 'chat' models
2. ¿Cómo forzar a que el chatbot responda 'si' o 'no'?¿Cómo parsear la salida para que siga un formato determinado?
3. Ventajas e inconvenientes de RAG vs fine-tunning
4. ¿Cómo evaluar el desempaño de un bot de Q&A? ¿Cómo evaluar el desempeño de un RAG?

> **Apartado 3 (Opcional):** _Servicio local para detección de objetos. El objetivo es disponer de un servicio que tenga como entrada una imagen y que como salida proporcione un JSON con detecciones de coches y personas. Se han de cumplir los siguientes puntos:_

1. No hay necesidad de entrenar un modelo. Se pueden usar preentrenados.
2. El servicio ha de estar conteinerizado. Es decir, una imagen docker que al arrancar exponga el servicio.
3. La petición al servicio se puede hacer desde Postman o herramienta similar o desde código Python.
4. La solución ha de estar implementada en Python.

Además, plantear cuales serían los pasos necesarios para entrenar un modelo de detección con categorías no existentes en los modelos preentrenados. Los puntos en los que centrar la explicación son:

1. Pasos necesarios a seguir.
2. Descripción de posibles problemas que puedan surgir y medidas para reducir el riesgo.
3. Estimación de cantidad de datos necesarios así como de los resultados, métricas, esperadas.
4. Enumeración y pequeña descripción (2-3 frases) de técnicas que se pueden utilizar para mejorar el desempeño, las métricas del modelo en tiempo de entrenamiento y las métricas del modelo en tiempo de inferencia.
