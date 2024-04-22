# Apartados teóricos

## Índice

- [Apartado 2](#apartado-2)
- [Apartado 3](#apartado-3)

### _Apartado 2_

> _Diferencias entre 'completion' y 'chat' models_

Los modelos tipo `completion`se utiliza principalmente para generar salidas de texto basadas en una solicitud de entrada. Funciona mejor con instruciones directas y se recomienta su uso cuando no se requiere mantener un contexto de conversación a lo largo de multiples intercambios.

Sin embargo, los modelos tipo `chat` están diseñados especificamente para mantener conversaciones con el usuario creando una interacción bidireccional. El problema principal a tratar con estos modelos son las posibles alucinaciones.

> _¿Cómo forzar a que el chatbot responda 'si' o 'no'?¿Cómo parsear la salida para que siga un formato determinado?_

La manera más sencilla para que un chatbot entienda que únicamente tiene que responder 'si' o 'no' a las entradas o preguntas que se le realicen, es mediante técnicas de prompting, indicandole especificamente al modelo que de esa salida.

Cuando se requiere que la respuesta de un modelo salga en un determinado formato (json, csv, etc.) se puede utilizar estos "parseadores" o analizadores clase, los cuales tienen dos metodos a implementar: obtener la instrucciones de formato, es decir, el cómo realizar ese formateo y un metodo 'parse' que toma el string de entrada y lo formatea.

> _Ventajas e inconvenientes de RAG vs fine-tunning_

El método `RAG` combina un mecanismo de recuperación (retriever) con un modelo de generación para proporcionar respuestas. El retriever busca información relevante en un conjunto de datos o corpus, y esta información es utilizada por el modelo generativo para formular una respuesta. Ahora comentamos las principales ventajas y desventajas.

- **Ventajas:**

  - _Precisión en información específica:_ al recuperar información de documentos relevantes antes de generar una respuesta, RAG puede proporcionar respuestas más precisas y detalladas basadas en los datos disponibles. Evita las alucinaciones.
  - _Adaptable a datos actualizados:_ RAG puede aprovechar un corpus actualizado regularmente sin necesidad de un nuevo entrenamiento intensivo del modelo generativo.

- **Desventajas:**
  - _Dependencia de la calidad de los datos:_ la calidad de las respuestas generadas depende fuertemente de la relevancia y calidad del corpus utilizado por el sistema de recuperación.
  - _Complejidad de implementación:_ integrar y optimizar los sistemas de recuperación y generación puede ser técnicamente complejo.
  - _Lantencia:_ el tiempo de respuesta puede ser mayor y por ello un inconveniente en aplicaciones de tiempo real.

Si hablamos de `fine-tuning` implica el entrenamiento de un modelo preexistente en un conjunto de datos específico para adaptarlo a una tarea particular, ajustando los pesos del modelo para optimizar su rendimiento en la tarea deseada.

- **Ventajas:**

  - _Personalización específica en una tarea:_ permite adaptar el modelo a necesidades muy específicas, mejorando su rendimiento en tareas particulares.
  - _Eficiencia en tiempo de respuesta:_ el modelo puede generar respuestas rápidamente sin la necesidad de buscar en un corpus externo.
  - _Control sobre el aprendizaje:_ los desarrolladores pueden influir directamente en lo que el modelo aprende y cómo responde a través del diseño del conjunto de datos de entrenamiento, dentro de unos cierto limites claro esta.

- **Desventajas:**
  - _Costo en recursos:_ el fine-tuning puede requerir una cantidad significativa de recursos computacionales y de datos.
  - _Riesgo de sobreajuste:_ si el conjunto de datos de refinamiento no es lo suficientemente diverso o es demasiado pequeño, el modelo puede sobreajustarse a esos datos y no generalizar bien en situaciones reales.
  - _Mantenimiento continuo:_ los modelos refinados pueden necesitar actualizaciones regulares a medida que cambian las condiciones o aparecen nuevos datos.

La eleccion entre un sistema u otro dependerá pues de la disponibiliada de datos que se tenga, el coste de recursos y tiempo disponible para llevarlo a cabo, las necesidades de actualización para el caso de que los datos sean cambiantes o más constantes, el tiempo de respuesta real, etc.

> _¿Cómo evaluar el desempaño de un bot de Q&A? ¿Cómo evaluar el desempeño de un RAG?_

Existen diferentes sistemas de evaluación o benchmarks que son capaces de evaluar el rendimiento de estos chatbots, con diferentes metricas (Accuracy, BLEU, ROUGE, etc.) específicas, utilizando una serie de datasets en concreto. Un repositorio comunmente utilizado es el [llm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

Para los sistemas RAG también existen esas metricas de evalución como pueden ser la precisión, relevancia de la respuesta, utilización del contexto y su precisión, etc. [Ragas](https://docs.ragas.io/en/stable/index.html) es un ejemplo de framework que combina todas las herramientas para evaluar este tipo de sistemas.

### _Apartado 3_

> _Cuales serían los pasos necesarios para entrenar un modelo de detección con categorías no existentes en los modelos preentrenados. Los puntos en los que centrar la explicación son:_
>
> 1. _Pasos necesarios a seguir._
> 2. _Descripción de posibles problemas que puedan surgir y medidas para reducir el riesgo._
> 3. _Estimación de cantidad de datos necesarios así como de los resultados, métricas, esperadas._
> 4. _Enumeración y pequeña descripción (2-3 frases) de técnicas que se pueden utilizar para mejorar el desempeño, las métricas del modelo en tiempo de entrenamiento y las métricas del modelo en tiempo de inferencia_

1. Es necesario comenzar con una recopilación amplia de datos/imagenes, las cuales sean un conjunto amplio en cuanto a diversidad y real para el caso de uso.

   Posteriormente mediante un proceso de _labeling_ manual deben añadirse a las imagenes las _bounding box_ o mascaras dependiendo de si el entrenamiento es para la tarea de detección de objetos o de segmentación, respectivamente.

   Acto seguido se divide el conjunto de datos en entrenamiento y validación (se puede incluir un conjunto también de test), se normalizan las imagenes y se selecciona el modelo preentrenado que consideremos que sea optimo para la tarea.

   Tras el entrenamiento y evaluación del model entrenado, sólo queda llevarlo o implementarlo en un endpoint para su servicio.

2. Dentro del mundo de _computer vision_, destacan problemas principales a abordar como la oclusión, confusión, cambios en la iluminación, objetos pequeños y cambios de escala, deformación, cambios de pose, etc.

   Este tipo de problemas se pueden abordar aumentando la cantidad y variabilidad en las imagenes, esto conlleva el aumento y cambio de escala, fondos distintos tras el objeto, rotaciones, etc.

3. Dependiendo de la complejidad de las catergorías a clasificar se pueden necesitar más o menos imagenes, pero generalmente para obtener un buen rendimiento suele hablarse de miles de ellas.

   Las métricas utilizas son las algunas también utilizadas para tareas de clasificación: precisión, recall, F1-score. Además el _Average Precision_ (AP) para clasificacion por categoría y el _Mean Average Precision_ (mAP) para la media de todas las categorias, también son estándares utilizados.

4. Para mejorar el desempeño del modelo se pueden utilizar técnicas como el _Data Augmentation_ aplicando transformaciones aleatorias a las imagenes.

   En cuanto a la mejora del tiempo de entrenamiento el _Transfer Learning_ es ideal, utilizando un modelo que ya ha sido preentrenado para esa tarea.

   Si hablamos de mejorar el tiempo de inferencia, la cuantificación puede ayudar a optimizar nuestro modelo al reducir el tamaño y la complejidad de este.
