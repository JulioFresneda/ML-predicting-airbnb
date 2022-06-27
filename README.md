# ML-predicting-airbnb
En este ejercicio simulado se van a obtener ciertos datos, y usarlos para poner a prueba varios algoritmos de aprendizaje supervisado de clasificación.  Los datos provienen de la web InsideAirBnB, dedicada al estudio de los alquileres vacacionales ofrecidos en la plataforma AirBnB. Este fichero es una versión editada, del listado original de información sobre las ofertas existentes, para la ciudad de Madrid, en abril de 2017.  La tarea de clasificación consistirá en clasificar los datos según el tipo de alojamiento, definido en el campo room_type, a partir del resto de características.

![procesoAA](Proceso_de_AA.svg)

# Visión general y enmarcar el problema
Los objetivos son, a partir de ciertos datos, poner a prueba ciertos algoritmos enfocados al aprendizaje supervisado. Estos algoritmos serán:
- Naive Bayes
- K-Nearest-Neighbors (KNN)
- Árboles de decisión

Dentro de "poner a prueba", se incluye la selección óptima de hiperparámetros, así como el modelado de datos (encoding, features que no usaremos, etc.). 

# Recolección de datos

Como se ha mencionado anteriormente, los datos provienen de la web InsideAirBnB, dedicada al estudio de los alquileres vacacionales ofrecidos en la plataforma AirBnB. Este fichero es una versión editada, del listado original de información sobre las ofertas existentes, para la ciudad de Madrid, en abril de 2017.

Estos datos son públicos y accesibles para cualquiera.

# Preparación y análisis de los datos

En este apartado vamos a intentar entender un poco a qué datos nos enfrentamos. Dado que este es un problema de clasificación, la primera pregunta que surge es, ¿cuántas clases hay?
Vemos que hay tres clases: Casa/apartamento completo, habitación privada, y habitación compartida. Vemos además, que tenemos un problema con esta última clase: Tenemos muy pocas entradas. 
La anterior observación ha sido la primera y última que vamos a hacer antes de dividir el dataset en sets de entrenamiento y test (el cual no analizaremos nunca). Esto es para evitar lo que se conoce como data snooping. Sin embargo, personalmente he visto conveniente, al menos, analizar el número de entradas por clase antes de dividir. Ésto es porque, si existiese una clase con muy poca representación (como es el caso), existe la remota posibilidad de dejarla completamente fuera del set de entrenamiento, lo cual nos llevaría a que nuestros modelos ni si quiera conozcan una de las tres clases.
Para elminar esa (muy remota) posibilidad de que nuestro set de entrenamiento no sea representativo del total, vamos a usar la clase StratifiedSuffleSplit. Como bien dice su nombre, esta clase mezcla las entradas de un dataset, y devuelve dos subconjuntos con una proporción por clase similar a la proporción original (estratificada).

## Análisis y exploración de datos
Una vez tenemos nuestro set de entrenamiento, podemos investigar un poco para saber más sobre a qué nos podemos enfrentar. Ya que para explorar los datos vamos a hacer algunos cambios en el set, vamos a crear un set de entrenamiento.
Tenemos dos atributos categóricos: vecindario y grupo de vecindarios, los cuales podemos pensar que tienen relación. Tenemos además ocho atributos contínuos, de los cuales dos (longitud y latitud) representan información a priori muy relacionada a vecindario. Esto está bien saberlo por si en un futuro nos conviene prescindir de información redundante.
Vamos a ver algo de información estadística.
* La latitud y longitud son atributos con mucha precisión en sus decimales. Esto habrá que tenerlo en cuenta porque si se redondea podemos perder mucha información. 
* El precio tiene un rango muy amplio, parece que hay outliers. La desviación es notable y el máximo difiere por mucho de la media. Si escalamos, dependiendo de la técnica para escalar, estos outliers pueden influir en la escala de forma que se distorsione la información representada.
* El resto de atributos son contínuos y parece que se mueven en rangos acotados

## Preprocesamiento de datos
Este es un apartado clave de este ejercicio, y personalmente creo que es el más importante. En mi opinión, es muy fácil lanzar un algoritmo en python, la dificultad radica en cómo preprocesar los datos con lo que lo alimentamos.

Antes de nada, vamos a repasar algunas características del dataset que vamos a usar como entrenamiento:
- Está muy desbalanceada, apenas tenemos unas 150 entradas de la clase 2, de un total de más de 10 mil.
- Tenemos atributos numéricos, con unas escalas muy dispares y algunos casos de outliers.
- Tenemos dos atributos categóricos, por lo que habrá que usar algún tipo de enconding.

Teniendo en cuenta que vamos a usar el mismo set de entrenamiento ya procesado para todos los algoritmos, debemos procesarlo de forma que se adapte correctamente a las necesidades de todos ellos, en la medida de lo posible.
### Balanceado
En nuestro set de entrenamiento, claramente una de las clases está desbalanceada. Es por ello que no es mala idea corregir un poco este desequilibrio, a riesgo de que la clase no tenga suficiente peso para que el modelo pueda aprender correctamente de ella. Si bien es cierto que para árboles de decisión ésto se puede corregir desde sus parámetros, para Naive Bayes y KNN debemos usar otras técnicas.

Por ello usaremos una función genérica que balancee datasets desbalanceados, duplicando las entradas menos comunes (oversampling).
### Escalado
sklearn nos da un abanico de métodos de escalado, cada uno con distintas propiedades. Antes de elegir uno, vamos a listarlos con una breve descripción:
- MinMaxScaler: Escala cada atributo dado un rango concreto
- MaxAbsScaler: Escala cada atributo entre 0 y 1, en base a su máximo absoluto
- StandardScaler: Sigue la distribución normal estándar (SND). Por lo tanto, hace media = 0 y escala los datos a la varianza unitaria.
- RobustScaler: Elimina la mediana y escala de acuerdo con el rango de cuantiles (el valor predeterminado el rango entre el primer cuartil (cuantil 25) y el tercer cuartil (cuantil 75).

Tanto MinMaxScaler como MaxAbsScaler son muy sensibles a outliers. Por ejemplo, con rango 0-1, si tenemos un atributo cuyos valores van entre 0 y 1, excepto un outlier con un valor de 100, el escalado los transformará en valores entre 0 y 0.01, excepto 1 para el outlier. Claramente esto puede confundir al algoritmo. 
Dado que tenemos varios outliers en nuestros datos, como hemos visto antes, quizás deberíamos elegir otra alternativa.

Quizás en otros problemas podríamos eliminar los outliers asumiendo que son ruido y usar cualquier tipo de escalado, pero aquí los outliers quizás sí aportan información relevante al modelo. Por tanto, vamos a usar RobustScaler().

Hay que tener en cuenta que hay algoritmos como Multinomial Naive Bayes o Compliment Naive Bayes no aceptan valores negativos, por tanto StandardScaler y RobustScaler nos puede dar problemas si centramos en el 0. Teniendo en cuenta que son algoritmos enfocados a problemas sobre textos y conteo de palabras, quizás nos convendría descartarlos, pero en vez de eso, vamos a probar a re-escalar con MinMax al final, sólo para estos algoritmos. 

Vamos a crear un primer Pipeline (que encadenaremos con otros) con el escalado elegido. Además añado como buena práctica, un transformador que rellena los datos null con la media. Digo como buena práctica porque en este caso no hay datos nulos.
También creamos otro pipeline específico para los casos antes mencionados.

### Encoding
Igual que con el escalado, existen diversas técnicas de encoding. Las que más he visto usar son OrdinalEncoding y OneHotEncoding. Ninguna me termina de convencer:
- OrdinalEncoding: Asocia un ordinal a cada valor único del atributo. Tiene como problema que algunos algoritmos asumen que dos valores cercanos tienen más en común que dos valores lejanos: El barrio 2 estará más relacionado con el barrio 3 que con el 12, cuando en realidad no es así.
- OneHotEncoding: Para solucionar el problema anterior, OneHotEncoding crea n columnas para cada atributo, siendo n el número de valores únicos. Pone a 1 la columna del valor en cuestión, y 0 el resto. Como problema, se crean demasiados atributos nuevos. Esto, puede lastrar el desempeño de los algoritmos teniendo un dataset pequeño (10k).

Una posible codificación podría ser la Binary Encoding. Es prácticamente igual que OneHotEncoding, excepto que usa notación binaria. Esto reduce drásticamente el número de columnas necesarias.

Por tanto, BinaryEncoding va a ser nuestra selección.

#### Pipeline final, y preprocesamiento de datos
Crearemos un pipeline final que aplique el escalado a los atributos numéricos y la codificación a los atributos categóricos, los cuales no escalaremos pues sus valores son 1 o 0.
Ídem con su alternativa con MinMax.
## Elegir algoritmo
Para elegir un modelo, necesitamos saber qué algoritmo es más probable que funcione en nuestro problema. Para ello, usaremos validación cruzada.

En la validación cruzada, se divide el set de entrenamiento en n partes, y se entrena el algoritmo n veces con n-1 partes, probándose contra la parte restante.

Una vez tengamos medias de todos los algoritmos, elegiremos el mejor y lo entrenaremos con todo el set de entrenamiento, validandolo conta el test original.

### Naive Bayes
Los métodos de Naive Bayes son un conjunto de algoritmos de aprendizaje supervisados basados en la aplicación del teorema de Bayes con el supuesto "ingenuo" de independencia condicional entre cada par de características dado el valor de la variable de clase.

Hay diversas versiones, cada una con sus ventajas y desventajas. En principio, no aceptan ningún tipo de parámetro relacionado con el balanceo.
#### Gaussian Naive Bayes
Supone que la probabilidad de las características sigue una distribución gaussiana.
Una media de 0.47 de acierto, menos acierto que jugar a cara o cruz. Parece que Gaussian Naive Bayes no funciona demasiado bien.
#### Multinomial Naive Bayes
Implementa el NB para datos distribuidos de forma multinomial. Funciona especialmente bien con problemas de textos, pero quizás en este caso nos sorprenda. Como requisito, no admite valores negativos, por lo que usaremos la versión MinMax del pipeline.
Una media de 0.56 de acierto. Algo mejor que el gaussiano.
#### Complement Naive Bayes
El CNB es una adaptación del Multinomial, funciona especialmente bien con datasets poco balanceados. Nuestro set sigue sin estar del todo balanceado, quizás mejore algo la media.
Media de 0.53. Parece que no ha funcionado como esperábamos.
### K-Nearest Neighbors
El KNN es un algoritmo simple pero potente, que clasifica en función de los K vecinos más cercanos. Una tarea importante es ajustar el hiperparámetro K. Pero primero, vamos a probar con K por defecto, a ver cómo se comporta.
Una media de 0.83. Bastante mejor que los Naive Bayes, parece ser, y aún no hemos ajustado K. Vamos a ajustarla, teniendo en cuenta que lo ideal es que sea impar, y no sea múltiplo del número de clases, para evitar fronteras.

Para estimar los hiperparámetros ideales, podemos usar la clase GridSearchCV

Parece que como mejor resultados obtenemos es con K igual a 1, es decir, estimar en función de su vecino más cercano. Tiene cierto sentido, teniendo en cuenta que hablamos de pisos y habitaciones.

GridSearchCV ya hace validación cruzada por nosotros, por lo que no tenemos que repetirla.

### Decission Trees
Por último, vamos a probar con árboles de decisiones. Aunque hemos balanceado un poco nuestro set de entrenamiento, vamos a buscar el balance también usando el hiperparámetro disponible. Al igual que antes, probamos primero con los parámetros por defecto, para ver el comportamiento.
Media de 0.88! Hasta ahora, el mejor algoritmo que hemos probado. Vamos a calcular los mejores hiperparámetros disponibles.
Un 83% de acierto. Sin duda, parece un modelo prometedor. Puede sorprender que con los valores por defecto se obtenga más puntuación (88), pero probablemente sea causa de overfitting. Si nos fijamos en los parámetros usados con GridSearch, uno limita el número de hojas y otro la profundidad. Es muy probable que ésto nos evite el overfitting, y contraintuitivamente lo haga funcionar mejor en el mundo real.

## Entrenar con el algoritmo elegido
Parece que, por poco, el KNN ha resultado ser el mejor algoritmo, o al menos, el que mejor resultados ha dado en el set de entrenamiento. Vamos a entrenarlo con todo el set, para probarlo en nuestro set de test original, el cual no hemos mirado en absoluto para evitar data snooping. Quizás haya mala suerte y no funcione como se espera, vamos a averiguarlo. Usamos el hiperparametro de k=1.

Una precisión de casi un 80%! No está nada mal. En la matriz de confusión podemos obtener más información. Vemos que clasifica estupendamente los apartamentos completos, ha habido pocos casos donde los ha confundido con habitaciones. Ídem para habitaciones unipersonales. Con habitaciones compartidas en cambio, el modelo no es demasiado bueno. Si bien no los confunde prácticamente con pisos completos (solo faltaría!), casi todas las interpreta como habitaciones unipersonales.

En cuanto al recall, precision y F1, todas van en la misma linea y ninguna destaca de sobremanera.

En general es un modelo equilibrado en el sentido de que hay numeros parecidos entre falsos positivos y falsos negativos.

## Conclusiones finales
Respecto al problema, está claro que la parte difícil es clasificar las habitaciones compartidas. Era de esperar, estaba muy desbalanceada en ese caso, y quizás al ser casos poco comunes, sus características no siguen un patrón tan fácil de encontrar como con habitaciones o pisos.

Respecto a los algoritmos, creo que cuadra lo que hemos visto con el funcionamiento de cada algoritmo. 
* Naive Bayes asume independencia entre cada una de los atributos, es decir, ninguno influye sobre otro. Esto es difícil que sea así, por ejemplo, hay barrios mucho más caros que otros, y a mayor precio se espera que haya menos valoraciones, pues menos gente puede permitírselo.
* Los árboles de decisión quizás sí hubieran funcionado mejor con habitaciones compartidas, por su modelo de funcionamiento. Pero tienden al overfitting, por tanto una puntuación alta puede engañar, incluso con cross validation.
* KNN en mi opinión es el algoritmo con más sentido, en este problema en particular, y así lo han demostrado las puntuaciones. Digo esto porque es muy raro que una habitación o piso de alquiler tenga un precio muy dispar con la competencia de la zona. Sí que se pueden dar casos por ejemplo en los que estén en la frontera con otro barrio más rico, o haya alguna excepción, pero por lo general suele estar bastante equilibrado en la vida real. La parte negativa del KNN, es que es muy dificil que prediga bien excepciones: Puede haber un piso muy caro porque el edificio sea de renombre, por ejemplo, en un barrio normal, con pisos vecions a un precio normal. Esto KNN no lo puede controlar.

## Bibliografia

* Medidas: https://towardsdatascience.com/how-to-evaluate-your-machine-learning-models-with-python-code-5f8d2d8d945b
* Naive Bayes: https://scikit-learn.org/stable/modules/naive_bayes.html
* KNN: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
* Decision trees: https://scikit-learn.org/stable/modules/tree.html
* Distintos tipos de encoding: https://www.analyticsvidhya.com/blog/2020/08/types-of-categorical-data-encoding/
* Clase para balancear (No se ha acabado usando, pero es interesante): https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/
