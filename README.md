# Taller 1 -INTELIGENCIA ARTIFICIAL
### *ADULTS INCOMES*  

En base a los requerimientos del problema, en este caso, predecir si una persona tiene ingresos mayores o menores a $50.000 USD, se plantean las siguientes preguntas con el fin de otorgar solución al mismo.

#####1. ¿Qué contienen los datos?

Se analiza el conjunto de datos, en este se evidencian tres archivos principales

- *adult.data*: Representa el conjunto de datos de entrenamiento.
- *adult.names*: Describen el conjunto de los datos.
- *adult.test*: Es el conjunto de datos de prueba.

##### ANALISIS DE DATOS
Inicialmente se observa la cantidad de datos disponibles, tanto para el entrenamiento como para las pruebas.

- ***Caracteristicas y tipos de datos***

Para el dataset se tiene tipos de datos cualitatívos (categóricos) y cuantitatívos (numéricos).

Caracteristica | Categórico | Numérico |
--- | --- | --- | 
Age|  | X | 
Workclass| X |  |
Fnlwgt|  | X |  
Education| X |  |
Education-num| X |  |
Marital-status| X |  |
Occupation| X |  |
Relationship| X |  |
Race| X |  |
Sex| X |  |
Capital-gain|  | X |
Capital-loss|  | X |
Hours per week|  | X |
Native country| X |  |

- ***Cantidad de datos y manejo de datos perdidos***

Descripción | Cantidad | Caracteristicas | Datos perdidos | Porcentaje datos perdidos
--- | --- | --- | --- | --- |
Datos de entrenamiento | 32561 | 14 | 2399 | 7.6
Datos de prueba | 16281 | 14 | 1221 | 7.5

Las caracteristicas: ***workClass, occupation*** y ***native-country*** tienen valores perdidos.
Para manejar datos perdidos se contemplaron dos opciones: rellenar esos datos con algún valor aleatorio o descartarlos, decidimos descartar los datos perdidos, pues constituyen aproximadamente el 7.6% de la cantidad de datos en total para los dos conjuntos de datos. 

- Cantidad de datos

Los datos estan constituidos por 32561 muestras y 14 caracteristicas. Los datos contienen una series de datos de tipo categóricos y numéricos.  

Mientras que,los datos correspondientes al conjunto de prueba estan constituidos por 16281 muestras y no existe ninguna perdida en los datos.

2. ¿Qué métodos se proponen utilizar y por qué?

El primer modelo utilizado fué, ***Máquinas de Vector Soporte*** ya que rara vez se encuentran casos en los que las clases sean perfectas y linealmente separable, se prefiere implementar un clasificador que se base en un hiperplano que, aunque no separe perfectamente las dos clases, tenga una mayor capacidad de prediccion  presentando menos problemas de overfitting. Pues, en vez de buscar que todos los datos de la clase se encuentren en el lado correcto del hiperplano, da la posibilidad que ciertas muestras estén en el lado incorrecto del hiperplano. Luego, el hiperplano depende únicamente de una pequeña proporción de muestras o vectores soporte, es su robustez frente a observaciones muy alejadas del hiperplano.

Como segundo modelo, se tomó el ***Perceptron***, pues este siempre encuentra un hiperplano de separación, siempre y cuando los datos sean linealmente separables.



```python
print('This is a example')
````





