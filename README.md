# Taller1_IA

En base a los requerimientos del problema, en este caso, predecir si una persona tiene ingresos mayores o menores a $50.000, se plantean las siguientes preguntas con el fin de otorgar solución al mismo.

1. ¿Qué contienen los datos?

Se analizan el conjunto de datos, en este se evidencian tres archivos principales

- adult.data

Representa el conjunto de datos de entrenamiento.

- adult.names

Describen el conjunto de los datos.

- adult.test

Es el conjunto de datos de prueba.

Los datos estan constituidos por 32561 muestras y 14 parametros. Los datos contienen una serie de valores categóricos, numéricos y perdidos. Las columnas correspondientes a ***workClass, occupation*** y ***native-country*** tienen valores perdidos. 

Mientras que,los datos correspondientes al conjunto de prueba estan constituidos por 16281 muestras y no existe ninguna perdida en los datos.

2. ¿Qué métodos se proponen utilizar y por qué?

El primer modelo utilizado fué, ***Máquinas de Vector Soporte*** ya que rara vez se encuentran casos en los que las clases sean perfectas y linealmente separable, se prefiere implementar un clasificador que se base en un hiperplano que, aunque no separe perfectamente las dos clases, tenga una mayor capacidad de prediccion  presentando menos problemas de overfitting. Pues, en vez de buscar que todos los datos de la clase se encuentren en el lado correcto del hiperplano, da la posibilidad que ciertas muestras estén en el lado incorrecto del hiperplano. Luego, el hiperplano depende únicamente de una pequeña proporción de muestras o vectores soporte, perfilandose a ser mas robusto frente a muestras muy lejanas.

Como segundo modelo, se tomó el ***Perceptron***, pues este siempre encuentra un hiperplano de separación, siempre y cuando los datos sean linealmente separables. Siendo bastante simple e ideal para clasificaciones binarias. Este se puede entrenar en tiempo real haciendo uso de la actualización del modelo con una única iteración con los datos que se incluyan como argumentos.

## Desarrollo Métodos

En el codigo adjunto de Python se ve representada la implementación de los modelos, de acuerdo a los parametros especificados en el enunciado del taller.

## Resultados

En en desarollo del codigo, se realizan la precisión media de los dos modelos especificados, obteniendo como resultado:

- SMV = 0.847
-PERCEPTRON = 0.795

De tal modo que, el porcentaje correspondiente a cada modelo es:

- SMV = 84.7%
-PERCEPTRON = 79.5%



## Conclusiones

La forma que describe el perceptron es bastante sencilla, de tal manera que permite una muy buena flexibilidad al momento de la actualización de los pesos, permitiendo que sea facil de adaptar al problema. 

La cantidad de los parametros es primordial al momento de la predicción. Pues muchos de ellos pueden llegar a hacer muy complejo el modelo y su desarrollo, para ello es necesario el uso de herramientas como PCA.


```python
print('This is a example')
````


