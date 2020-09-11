# Taller 1 -INTELIGENCIA ARTIFICIAL
### *ADULTS INCOMES*  

En base a los requerimientos del problema, en este caso, predecir si una persona tiene ingresos mayores o menores a $50.000 USD, se plantean las siguientes preguntas con el fin de otorgar solución al mismo.

#### 1. ¿Qué contienen los datos?

Se analiza el conjunto de datos, en este se evidencian tres archivos principales

- *adult.data*: Representa el conjunto de datos de entrenamiento.
- *adult.names*: Describen el conjunto de los datos.
- *adult.test*: Es el conjunto de datos de prueba.

##### *ANALISIS DE DATOS* 

- ***Caracteristicas y tipos de datos***

Para el dataset se tiene tipos de datos cualitatívos (categóricos) y cuantitatívos (numéricos).


| Caracteristica | Categórico | Numérico |
|:--------------:|:----------:|:--------:| 
|      Age       |            |    X     | 
|   Workclass    |      X     |          |
|     Fnlwgt     |            |    X     |  
|   Education    |      X     |          |
| Education-num  |      X     |          |
| Marital-status |      X     |          |
|  Occupation    |      X     |          |
|  Relationship  |      X     |          |
|      Race      |      X     |          |
|      Sex       |      X     |          |
|  Capital-gain  |            |    X     |
|  Capital-loss  |            |    X     |
| Hours per week |            |    X     |
| Native country |      X     |          |

- ***Cantidad de datos y manejo de datos perdidos***

Se observa la cantidad de datos disponibles, tanto para el entrenamiento como para las pruebas.

|       Descripción      | Cantidad | Caracteristicas | Datos perdidos | Porcentaje datos perdidos |
|:----------------------:|:--------:|:---------------:|:--------------:|:-------------------------:|
| Datos de entrenamiento |   32561  |        14       |      2399      |            7.6            |
|    Datos de prueba     |   16281  |        14       |      1221      |            7.5            |

Las caracteristicas: ***workClass, occupation*** y ***native-country*** tienen valores perdidos.
Para manejar datos perdidos se contemplaron dos opciones: rellenar esos datos con algún valor aleatorio o descartarlos, decidimos descartar los datos perdidos, pues constituyen aproximadamente el 7.6% de la cantidad de datos en total para los dos conjuntos de datos. 

- ***Balance de datos***

Se puede observar que el conjunto de datos de entrenamiento y de prueba presentan un desbalance, lo cual llegará a afectar en el entrenamiento del módelo.

|     Datos      |     <=50K      |     >50K      |     Total    |  
|:--------------:|:--------------:|:-------------:|:------------:| 
| Entrenamiento  | 24720 (75.92%) | 7841 (24.08%) | 32561 (100%) |
|     Prueba     | 12435 (76.38%) | 3846 (23.62%) | 16281 (100%) |
|                |        


## PROCESAMIENTO DE DATOS

Para las caracteristicas de tipo categóricas (exceptuando `'Sex'`) se realizó una codificación ***One Hot*** la cual aumenta la dimensionalidad del problema, pues si una caracteristica tiene 3 opciones, esta caracteristica será reemplazada por 3 nuevas caracteristicas, cada una correspondiente a las opciones que tenía la caracteristica inicial.

|    Datos     |Caracteristicas iniciales|Caracteristicas despues de codificar|
|:------------:|:-----------------------:|:----------------------------------:| 
|Entrenamiento |            14           |                 103                |
|    Prueba    |            14           |                 102                |

Se observa que despues de realizar la codificación a ambos conjuntos de datos no tienen la misma cantidad de caracteristicas.
Lo anterior se debe a que en la caracteristica inicial `'Native country'` el set de entrenamiento cuenta con la opción `'Holand-Netherlands'`, es decir, alguna persona nació en este pais, mientras que en el conjunto de pruebas ninguna persona nació en este pais, por ello la diferencia en esta caracteristica (despues de realizar la codificación)

Al observar que en el conjunto de entrenamiento, sólo una persona tenía como `'Native country'` a `'Holand-Netherlands'` se decidió descartar esta caracteristica, pues no sería muy relevante.

Lo anterior se debe a qué los dos conjuntos de datos deben tener la misma cantidad de caracteristicas.  

#### 2.¿Qué métodos se proponen utilizar y por qué?

El primer modelo utilizado fue, Máquinas de Vector Soporte ya que rara vez se encuentran casos en los que las clases sean perfectas y linealmente separable, se prefiere implementar un clasificador que se base en un hiperplano que, aunque no separe perfectamente las dos clases, tenga una mayor capacidad de prediccion presentando menos problemas de overfitting. Pues, en vez de buscar que todos los datos de la clase se encuentren en el lado correcto del hiperplano, da la posibilidad que ciertas muestras estén en el lado incorrecto del hiperplano. Luego, el hiperplano depende únicamente de una pequeña proporción de muestras o vectores soporte, perfilandose a ser mas robusto frente a muestras muy lejanas.

Como segundo módelo, se tomó el Perceptron, pues este siempre encuentra un hiperplano de separación, siempre y cuando los datos sean linealmente separables. Siendo bastante simple e ideal para clasificaciones binarias. Este se puede entrenar en tiempo real haciendo uso de la actualización del modelo con una única iteración con los datos que se incluyan como argumentos.

## DESARROLLO DE MÉTODOS


En el codigo adjunto de Python se ve representada la implementación de los modelos, de acuerdo a los parametros especificados en el enunciado del taller.

### RESULTADOS

En en desarollo del código, se realizan la precisión media de los dos modelos especificados, obteniendo como resultado:

|    Módelo    | Precisión media | 
|:------------:|:---------------:| 
|     SVM      |      0.848      |
|   Fischer    |      0.839      |
|  Perceptron  |      0.795      |


### COMPARACIONES 

|    Resultados SVM    | 
|:--------------------:| 
|![SVM](https://user-images.githubusercontent.com/61461128/92958993-c80b1b00-f430-11ea-895b-e779d8ef5d88.PNG)|

|    Resultados Perceptron    | 
|:---------------------------:| 
|![Perceptron](https://user-images.githubusercontent.com/61461128/92958772-7367a000-f430-11ea-8754-32355fe14de0.PNG)|

|    Resultados Fischer    | 
|:------------------------:| 
|![Fischer](https://user-images.githubusercontent.com/61461128/92958981-c4779400-f430-11ea-9ee4-e6c17adc6882.PNG)|

### CONCLUSIONES

La forma que describe el perceptron es bastante sencilla, de tal manera que permite una muy buena flexibilidad al momento de la actualización de los pesos, permitiendo que sea facil de adaptar al problema.

La cantidad de los parametros es primordial al momento de la predicción. Pues muchos de ellos pueden llegar a hacer muy complejo el modelo y su desarrollo, para ello es necesario el uso de herramientas como PCA.
