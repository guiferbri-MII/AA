# ==========================================================
# Aprendizaje automático 
# Máster en Ingeniería Informática - Universidad de Sevilla
# Curso 2021-22
# Primer trabajo práctico
# ===========================================================

# --------------------------------------------------------------------------
# APELLIDOS: Fernández de Bobadilla Brioso
# NOMBRE: Guiomar
# ----------------------------------------------------------------------------


# *****************************************************************************
# HONESTIDAD ACADÉMICA Y COPIAS: un trabajo práctico es un examen, por lo que
# debe realizarse de manera individual. La discusión y el intercambio de
# información de carácter general con los compañeros se permite (e incluso se
# recomienda), pero NO AL NIVEL DE CÓDIGO. Igualmente el remitir código de
# terceros, OBTENIDO A TRAVÉS DE LA RED o cualquier otro medio, se considerará
# plagio. 

# Cualquier plagio o compartición de código que se detecte significará
# automáticamente la calificación de CERO EN LA ASIGNATURA para TODOS los
# estudiantes involucrados. Por tanto, NO se les conservará, para
# futuras convocatorias, ninguna nota que hubiesen obtenido hasta el
# momento. SIN PERJUICIO DE OTRAS MEDIDAS DE CARÁCTER DISCIPLINARIO QUE SE
# PUDIERAN TOMAR.  
# *****************************************************************************


# IMPORTANTE: NO CAMBIAR EL NOMBRE NI A ESTE ARCHIVO NI A LAS CLASES Y MÉTODOS
# QUE SE PIDEN





# ========================
# IMPORTANTE: USO DE NUMPY
# ========================

# SE PIDE USAR NUMPY EN LA MEDIDA DE LO POSIBLE. 

import numpy as np
import carga_datos
X_votos = carga_datos.X_votos
y_votos = carga_datos.y_votos
X_credito = carga_datos.X_credito
y_credito = carga_datos.y_credito
X_cancer = carga_datos.X_cancer
y_cancer = carga_datos.y_cancer
X_train_imdb = carga_datos.X_train_imdb
X_test_imdb = carga_datos.X_test_imdb
y_train_imdb = carga_datos.y_train_imdb
y_test_imdb = carga_datos.y_test_imdb
X_iris = carga_datos.X_iris
y_iris = carga_datos.y_iris
# SE PENALIZARÁ el uso de bucles convencionales si la misma tarea se puede
# hacer más eficiente con operaciones entre arrays que proporciona numpy. 

# PARTICULARMENTE IMPORTANTE es el uso del método numpy.dot. 
# Con numpy.dot podemos hacer productos escalares de pesos por características,
# y extender esta operación de manera compacta a dos dimensiones, cuando tenemos 
# varias filas (ejemplos) e incluso varios varios vectores de pesos.  

# En lo que sigue, los términos "array" o "vector" se refieren a "arrays de numpy".  

# NOTA: En este trabajo NO se permite usar scikit-learn (salvo en el código que
# se proporciona para cargar los datos).

# -----------------------------------------------------------------------------

# *****************************************
# CONJUNTOS DE DATOS A USAR EN ESTE TRABAJO
# *****************************************

# Para aplicar las implementaciones que se piden en este trabajo, vamos a usar
# los siguientes conjuntos de datos. Para cargar todos los conjuntos de datos,
# basta con descomprimir el archivo datos-trabajo-aa.zip y ejecutar el
# archivo carga_datos.py (algunos de estos conjuntos de datos se cargan usando
# utilidades de Scikit Learn). Todos los datos se cargan en arrays de numpy.

# * Datos sobre concesión de prestamos en una entidad bancaria. En el propio
#   archivo datos/credito.py se describe con más detalle. Se carga en las
#   variables X_credito, y_credito.   

# * Conjunto de datos de la planta del iris. Se carga en las variables X_iris,
#   y_iris.  

# * Datos sobre votos de cada uno de los 435 congresitas de Estados Unidos en
#   17 votaciones realizadas durante 1984. Se trata de clasificar el partido al
#   que pertenece un congresista (republicano o demócrata) en función de lo
#   votado durante ese año. Se carga en las variables X_votos, y_votos. 

# * Datos de la Universidad de Wisconsin sobre posible imágenes de cáncer de
#   mama, en función de una serie de características calculadas a partir de la
#   imagen del tumor. Se carga en las variables X_cancer, y_cancer.
  
# * Críticas de cine en IMDB, clasificadas como positivas o negativas. El
#   conjunto de datos que usaremos es sólo una parte de los textos. Los textos
#   se han vectorizado usando CountVectorizer de Scikit Learn. Como vocabulario, 
#   se han usado las 609 palabras que ocurren más frecuentemente en las distintas 
#   críticas. Los datos se cargan finalmente en las variables X_train_imdb, 
#   X_test_imdb, y_train_imdb,y_test_imdb.    

# * Un conjunto de imágenes (en formato texto), con una gran cantidad de
#   dígitos (de 0 a 9) escritos a mano por diferentes personas, tomado de la
#   base de datos MNIST. En digitdata.zip están todos los datos en formato
#   comprimido. Para preparar estos datos habrá que escribir funciones que los
#   extraigan de los ficheros de texto (más adelante se dan más detalles). 




# ===========================================================
# EJERCICIO 1: SEPARACIÓN EN ENTRENAMIENTO Y PRUEBA (HOLDOUT)
# ===========================================================

# Definir una función 

#           particion_entr_prueba(X,y,test=0.20)

# que recibiendo un conjunto de datos X, y sus correspondientes valores de
# clasificación y, divide ambos en datos de entrenamiento y prueba, en la
# proporción marcada por el argumento test, y conservando la correspondencia 
# original entre los ejemplos y sus valores de clasificación.
# La división ha de ser ALEATORIA y ESTRATIFICADA respecto del valor de clasificación.

def particion_entr_prueba(X,y,test=0.20):
    X_train, X_test = np.empty((0,X.shape[1])), np.empty((0,X.shape[1]))
    y_train, y_test = np.empty((0,1)), np.empty((0,1))
    y_values = np.unique(y)
    for y_value in y_values:
        y_value_indexes = np.where(y == y_value)[0]
        y_value_indexes_random = np.random.permutation(y_value_indexes)
        num_train_prop = int((1-test)*len(y_value_indexes))
        index_train, index_test = y_value_indexes_random[:num_train_prop], y_value_indexes_random[num_train_prop:]

        X_train = np.append(X_train, X[index_train,:], axis = 0)
        X_test = np.append(X_test, X[index_test,:], axis = 0)
        y_train = np.append(y_train, y[index_train])        
        y_test = np.append(y_test, y[index_test])
    return X_train, X_test, y_train, y_test


#Xe_votos,Xp_votos,ye_votos,yp_votos = particion_entr_prueba(X_votos,y_votos,test=1/3)
# BORRAR
'''#print(X_votos[0])
#print(y_votos[0])
#print(particion_entr_prueba(X_votos, y_votos))
print('ye_votos[0]', ye_votos[0])
print('Xe_votos[0]', Xe_votos[0])
print(X_votos.shape,Xe_votos.shape,Xp_votos.shape)
print(y_votos.shape,ye_votos.shape,yp_votos.shape)
print(y_votos.shape[0],ye_votos.shape[0],yp_votos.shape[0])
print(np.unique(y_votos,return_counts=True))
print(np.unique(ye_votos,return_counts=True))
print(np.unique(yp_votos,return_counts=True))'''


#Xe_credito,Xp_credito,ye_credito,yp_credito =particion_entr_prueba(X_credito,y_credito,test=0.4)
# BORRAR
'''print('CREDITO')
print(np.unique(y_credito,return_counts=True))
print(np.unique(ye_credito,return_counts=True))
print(np.unique(yp_credito,return_counts=True))
print(X_credito.shape,Xe_credito.shape,Xp_credito.shape)
print(y_credito.shape,ye_credito.shape,yp_credito.shape)
print('ye_credito[0]', ye_credito[0])
print('Xe_credito[0]', Xe_credito[0])'''
# ------------------------------------------------------------------------------
# Ejemplos:
# =========

# En votos:

# In[1]: Xe_votos,Xp_votos,ye_votos,yp_votos          
#            =particion_entr_prueba(X_votos,y_votos,test=1/3)

# Como se observa, se han separado 2/3 para entrenamiento y 1/3 para prueba:
# In[2]: y_votos.shape[0],ye_votos.shape[0],yp_votos.shape[0]
# Out[2]: (435, 290, 145)

# Las proporciones entre las clases son (aprox) las mismas en los dos conjuntos de
# datos, y la misma que en el total: 267/168=178/112=89/56

# In[3]: np.unique(y_votos,return_counts=True)
# Out[3]: (array(['democrata', 'republicano'], dtype='<U11'), array([267, 168]))
# In[4]: np.unique(ye_votos,return_counts=True)
# Out[4]: (array(['democrata', 'republicano'], dtype='<U11'), array([178, 112]))
# In[5]: np.unique(yp_votos,return_counts=True)
# Out[5]: (array(['democrata', 'republicano'], dtype='<U11'), array([89, 56]))

# La división en trozos es aleatoria y, por supuesto, en el orden en el que
# aparecen los datos en Xe_votos,ye_votos y en Xp_votos,yp_votos, se preserva
# la correspondencia original que hay en X_votos,y_votos.


# Otro ejemplo con más de dos clases:

# In[6]: Xe_credito,Xp_credito,ye_credito,yp_credito               
#              =particion_entr_prueba(X_credito,y_credito,test=0.4)

# In[7]: np.unique(y_credito,return_counts=True)
# Out[7]: (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#          array([202, 228, 220]))

# In[8]: np.unique(ye_credito,return_counts=True)
# Out[8]: (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#          array([121, 137, 132]))

# In[9]: np.unique(yp_credito,return_counts=True)
# Out[9]: (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#          array([81, 91, 88]))
# ------------------------------------------------------------------





# ===========================================
# EJERCICIO 2: REGRESIÓN LOGÍSTICA MINI-BATCH
# ===========================================


# Se pide implementar el clasificador de regresión logística mini-batch 
# a través de una clase python, que ha de tener la siguiente estructura:

# class RegresionLogisticaMiniBatch():

#    def __init__(self,normalizacion=False,
#                 rate=0.1,rate_decay=False,batch_tam=64,
#                 pesos_iniciales=None):

#          .....
         
#    def entrena(self,entr,clas_entr,n_epochs=1000,
#                reiniciar_pesos=False):

#         ......

#     def clasifica_prob(self,E):


#         ......

#     def clasifica(self,E):


#         ......
        

# Explicamos a continuación cada uno de los métodos:


# * Constructor de la clase:
# --------------------------

#  El constructor debe tener los siguientes argumentos de entrada:


#  - El parámetro normalizacion, que puede ser True o False (False por
#    defecto). Indica si los datos se tienen que normalizar, tanto para el
#    entrenamiento como para la clasificación de nuevas instancias.  La
#    normalización es una técnica que suele ser útil cuando los distintos
#    atributos reflejan cantidades numéricas de muy distinta magnitud.
#    En ese caso, antes de entrenar se calcula la media m_i y la desviación
#    típica d_i en CADA COLUMNA i (es decir, en cada atributo) de los
#    datos del conjunto de entrenamiento.  A continuación, y antes del
#    entrenamiento, esos datos se transforman de manera que cada componente
#    x_i se cambia por (x_i - m_i)/d_i. Esta MISMA transformación se realiza
#    sobre las nuevas instancias que se quieran clasificar.

#  - rate: si rate_decay es False, rate es la tasa de aprendizaje fija usada
#    durante todo el aprendizaje. Si rate_decay es True, rate es la
#    tasa de aprendizaje inicial. Su valor por defecto es 0.1.

#  - rate_decay, indica si la tasa de aprendizaje debe disminuir en
#    cada epoch. En concreto, si rate_decay es True, la tasa de
#    aprendizaje que se usa en el n-ésimo epoch se debe de calcular
#    con la siguiente fórmula: 
#       rate_n= (rate_0)*(1/(1+n)) 
#    donde n es el número de epoch, y rate_0 es la cantidad
#    introducida en el parámetro rate anterior.   

#  - batch_tam: indica el tamaño de los mini batches (por defecto 64)
#    que se usan para calcular cada actualización de pesos.
    
#  - pesos_iniciales: Si es None, los pesos iniciales se inician 
#    aleatoriamente. Si no, debe proporcionar un array de pesos que se 
#    tomarán como pesos iniciales.     

# 

# * Método entrena:
# -----------------

#  Este método es el que realiza el entrenamiento del clasificador. 
#  Debe calcular un vector de pesos, mediante el correspondiente
#  algoritmo de entrenamiento basado en ascenso por el gradiente mini-batch, 
#  para maximizar la log verosimilitud. Describimos a continuación los parámetros de
#  entrada:  

#  - entr y clas_entr, son los datos del conjunto de entrenamiento y su
#    clasificación, respectivamente. El primero es un array (bidimensional)  
#    con los ejemplos, y el segundo un array (unidimensional) con las clasificaciones 
#    de esos ejemplos, en el mismo orden. 

#  - n_epochs: número de pasadas que se realizan sobre todo el conjunto de
#    entrenamiento.

#  - reiniciar_pesos: si es True, se reinicia al comienzo del 
#    entrenamiento el vector de pesos de manera aleatoria 
#    (típicamente, valores aleatorios entre -1 y 1).
#    Si es False, solo se inician los pesos la primera vez que se
#    llama a entrena. En posteriores veces, se parte del vector de
#    pesos calculado en el entrenamiento anterior. Esto puede ser útil
#    para continuar el aprendizaje a partir de un aprendizaje
#    anterior, si por ejemplo se dispone de nuevos datos.     

#  NOTA: El entrenamiento en mini-batch supone que en cada epoch se
#  recorren todos los ejemplos del conjunto de entrenamiento,
#  agrupados en grupos del tamaño indicado. Por cada uno de estos
#  grupos de ejemplos se produce una actualización de los pesos. 
#  Se pide una VERSIÓN ESTOCÁSTICA, en la que en cada epoch se asegura que 
#  se recorren todos los ejemplos del conjunto de entrenamiento, 
#  en un orden ALEATORIO, aunque agrupados en grupos del tamaño indicado. 


# * Método clasifica_prob:
# ------------------------

#  Método que devuelve el array de correspondientes probabilidades de pertenecer 
#  a la clase positiva (la que se ha tomado como clase 1), para cada ejemplo de un 
#  array E de nuevos ejemplos.


        
# * Método clasifica:
# -------------------
    
#  Método que devuelve un array con las correspondientes clases que se predicen
#  para cada ejemplo de un array E de nuevos ejemplos. La clase debe ser una de las 
#  clases originales del problema (por ejemplo, "republicano" o "democrata" en el 
#  problema de los votos).  


# Si el clasificador aún no ha sido entrenado, tanto "clasifica" como
# "clasifica_prob" deben devolver una excepción del siguiente tipo:

class ClasificadorNoEntrenado(Exception): pass

class RegresionLogisticaMiniBatch():
    
    def __init__(self,normalizacion=False,vrate=0.1,rate_decay=False,batch_tam=64,vpesos_iniciales=None):
        self.normalizacion = normalizacion
        self.vrate = vrate
        self.rate_decay = rate_decay
        self.batch_tam = batch_tam
        self.vpesos_iniciales = vpesos_iniciales
        self.weights = vpesos_iniciales
        self.isTrain = False

    def normalize(self,data):
        deviation = np.std(data, axis=0)
        mean = np.mean(data, axis=0)
        normalizeData = (data - mean)/deviation
        self.deviation = deviation
        self.mean = mean
        return normalizeData
    
    def pred(self, X, weights):
        return 1/(1+np.e**(-np.dot(X, weights)))

    def entrena(self,entr,clas_entr,n_epochs=1000, reiniciar_pesos=False):
        self.y_class = np.unique(clas_entr)
        clas_entr = np.where(clas_entr == self.y_class[1], 1, 0)
        X_train = entr
        if self.normalizacion:
            X_train = self.normalize(X_train)

        if reiniciar_pesos or self.vpesos_iniciales == None:
            self.weights = np.random.uniform(-1, 1, (X_train.shape[1]+1,))
        
        num_batches = int(X_train.shape[0] / self.batch_tam)
        weights = self.weights
        rate_0 = self.vrate
        
        for epoch in range(n_epochs):
            indexes_random = np.random.permutation(X_train.shape[0])
            X_train = X_train[indexes_random,:]
            clas_entr = clas_entr[indexes_random]

            rate = self.vrate

            #ToDo: Unificar for y el if
            for num_batch in range(num_batches):
                X_mini_batch = X_train[num_batch*self.batch_tam : (num_batch+1)*self.batch_tam, :]
                Y_mini_batch = clas_entr[num_batch*self.batch_tam : (num_batch+1)*self.batch_tam]
                X_mini_batch = np.insert(X_mini_batch, 0, 1, axis = 1)
                
                o = self.pred(X_mini_batch, weights)
                weights = weights + rate * (np.dot(X_mini_batch.T, (Y_mini_batch - o)))

            if X_train.shape[0] % self.batch_tam != 0:
                X_mini_batch = X_train[num_batches*self.batch_tam : X_train.shape[0], :]
                Y_mini_batch = clas_entr[num_batches*self.batch_tam : X_train.shape[0]]
                X_mini_batch = np.insert(X_mini_batch, 0, 1, axis = 1)

                o = self.pred(X_mini_batch, weights)
                weights = weights + (rate * (np.dot(X_mini_batch.T, (Y_mini_batch - o))))
                
            if self.rate_decay:
                rate = (rate_0)*(1/(1+epoch))

        self.weights = weights
        self.isTrain = True            

    def clasifica_prob(self,E):
        #print('clasifica_prob')
        if not self.isTrain:
            raise ClasificadorNoEntrenado("Primero debe entrenar el modelo")
        
        if self.normalizacion:
            E = (E - self.mean)/self.deviation
        
        E_x = np.copy(E)
        E_x = np.insert(E_x, 0, 1, axis = 1)
        return self.pred(E_x, self.weights)

    def clasifica(self,E):
        if not self.isTrain:
            raise ClasificadorNoEntrenado("Primero debe entrenar el modelo")
            
        E_x = np.copy(E)
        y_proba = self.clasifica_prob(E_x)
        classification = np.where(y_proba > 0.5,self.y_class[1],self.y_class[0])
        return classification

Xe_votos,Xp_votos,ye_votos,yp_votos = particion_entr_prueba(X_votos,y_votos)
# ToDo: DESCOMENTAR!!!
'''RLMB_votos=RegresionLogisticaMiniBatch(normalizacion=True)
#weigths = RLMB_votos.entrena(Xe_votos, ye_votos)
#print('RESULT entrena:', weigths)
RLMB_votos.entrena(Xe_votos, ye_votos)
probabilidad = RLMB_votos.clasifica_prob(Xp_votos)
#print('probabilidad')
#print(probabilidad)
clasificacion = RLMB_votos.clasifica(Xp_votos)
#print('clasificacion')
#print(clasificacion)'''



# Ejemplos de uso:
# ----------------



# CON LOS DATOS VOTOS:
        
#   

# En primer lugar, separamos los datos en entrenamiento y prueba (los resultados pueden
# cambiar, ya que esta partición es aleatoria)

        
# In [1]: Xe_votos,Xp_votos,ye_votos,yp_votos            
#            =particion_entr_prueba(X_votos,y_votos)

# Creamos el clasificador:
        
# In [2]: RLMB_votos=RegresionLogisticaMiniBatch()

# Lo entrenamos sobre los datos de entrenamiento:

# In [3]: RLMB_votos.entrena(Xe_votos,ye_votos)

# Con el clasificador aprendido, realizamos la predicción de las clases
# de los datos que estan en test:

# In [4]: RLMB_votos.clasifica_prob(Xp_votos)
# array([3.90234132e-04, 1.48717603e-11, 3.90234132e-04, 9.99994374e-01, 9.99347533e-01,...]) 
        
# In [5]: RLMB_votos.clasifica(Xp_votos)
# Out[5]: array(['democrata', 'democrata', 'democrata','republicano',... ], dtype='<U11')

# Calculamos la proporción de aciertos en la predicción, usando la siguiente 
# función que llamaremos "rendimiento".

def rendimiento(clasif,X,y):
    return sum(clasif.clasifica(X)==y)/y.shape[0]
        
# In [6]: rendimiento(RLMB_votos,Xp_votos,yp_votos)
# Out[6]: 0.9080459770114943    
# ToDo: DESCOMENTAR!!!
'''score_votes = rendimiento(RLMB_votos,Xp_votos,yp_votos)
print(score_votes)'''
# ---------------------------------------------------------------------

# CON LOS DATOS DEL CÀNCER
        
# Hacemos un experimento similar al anterior, pero ahora con los datos del 
# cáncer de mama, y usando normalización y disminución de la tasa         

# In[7]: Xe_cancer,Xp_cancer,ye_cancer,yp_cancer           
#           =particion_entr_prueba(X_cancer,y_cancer)


# In[8]: RLMB_cancer=RegresionLogisticaMiniBatch(normalizacion=True,rate_decay=True)

# In[9]: RLMB_cancer.entrena(Xe_cancer,ye_cancer)

# In[9]: RLMB_cancer.clasifica_prob(Xp_cancer)
# Out[9]: array([9.85046885e-01, 8.77579844e-01, 7.81826115e-07,..])

# In[10]: RLMB_cancer.clasifica(Xp_cancer)
# Out[10]: array([1, 1, 0,...])

# In[11]: rendimiento(RLMB_cancer,Xp_cancer,yp_cancer)
# Out[11]: 0.9557522123893806

Xe_cancer,Xp_cancer,ye_cancer,yp_cancer = particion_entr_prueba(X_cancer,y_cancer)
# ToDo: DESCOMENTAR!!!
'''RLMB_cancer=RegresionLogisticaMiniBatch(batch_tam=16,rate_decay=True)
RLMB_cancer.entrena(Xe_cancer,ye_cancer)
RLMB_cancer.clasifica_prob(Xp_cancer)
#print(RLMB_cancer.clasifica(Xp_cancer))
score_cancer = rendimiento(RLMB_cancer,Xp_cancer,yp_cancer)
print('score_cancer')
print(score_cancer)'''

# =================================================
# EJERCICIO 3: IMPLEMENTACIÓN DE VALIDACIÓN CRUZADA
# =================================================

# Este ejercicio vale 2 PUNTOS (SOBRE 10) pero se puede saltar, sin afectar 
# al resto del trabajo. Puede servir para el ajuste de parámetros en los ejercicios 
# posteriores, pero si no se realiza, se podrían ajustar siguiendo el método "holdout" 
# implementado en el ejercicio 1. 

# La técnica de validación cruzada que se pide en este ejercicio se explica
# en el tema "Evaluación de modelos".     

# Definir una función: 

#  rendimiento_validacion_cruzada(clase_clasificador,params,X,y,n=5)

# que devuelve el rendimiento medio de un clasificador, mediante la técnica de
# validación cruzada con n particiones. Los arrays X e y son los datos y la
# clasificación esperada, respectivamente. El argumento clase_clasificador es
# el nombre de la clase que implementa el clasificador. El argumento params es
# un diccionario cuyas claves son nombres de parámetros del constructor del
# clasificador y los valores asociados a esas claves son los valores de esos
# parámetros para llamar al constructor.

# INDICACIÓN: para usar params al llamar al constructor del clasificador, usar
# clase_clasificador(**params)
def rendimiento_validacion_cruzada(clase_clasificador,params,X,y,n=5):
    num_fold = int(X.shape[0] / n)
    y_values = np.unique(y)
    scores = []
    for i in range(n):

        X_train_data, X_test_data = np.empty((0,X.shape[1])), np.empty((0,X.shape[1]))
        y_train_data, y_test_data = np.empty((0,1)), np.empty((0,1))
        for y_value in y_values:
            y_value_indexes = np.where(y == y_value)[0]
            y_value_indexes_random = np.random.permutation(y_value_indexes)
            y_prop = round(len(y_value_indexes)/len(y),2)
            num_train_prop = int(round(y_prop*num_fold,0))
            
            index_train, index_test = y_value_indexes_random[:num_train_prop], y_value_indexes_random[num_train_prop:]
            X_train_data = np.append(X_train_data, X[index_train,:], axis = 0)
            X_test_data = np.append(X_test_data, X[index_test,:], axis = 0)
            y_train_data = np.append(y_train_data, y[index_train])        
            y_test_data = np.append(y_test_data, y[index_test])

        classifier = clase_clasificador(**params)  
        classifier.entrena(X_train_data, y_train_data)
        score_fold = rendimiento(classifier,X_test_data,y_test_data)
        scores.append(score_fold)

    scores_np = np.array(scores)
    return np.mean(scores_np)

X_prueba = np.array([11, 12, 13,14,15,16,17,18,19,20])
Y_prueba = np.array([0,1,1,1,0,1,0,1,1,0])  
# ToDo: DESCOMENTAR !!  
'''cancer_score_cross_val = rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch, {"batch_tam":16,"rate_decay":True},Xe_cancer,ye_cancer,n=5)
print(cancer_score_cross_val)'''

# ------------------------------------------------------------------------------
# Ejemplo:
# --------
# Lo que sigue es un ejemplo de cómo podríamos usar esta función para
# ajustar el valor de algún parámetro. En este caso aplicamos validación
# cruzada, con n=5, en el conjunto de datos del cáncer, para estimar cómo de
# bueno es el valor batch_tam=16 con rate_decay en regresión logística mini_batch.
# Usando la función que se pide sería (nótese que debido a la aleatoriedad, 
# no tiene por qué coincidir exactamente el resultado):

# >>> rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,         
#             {"batch_tam":16,"rate_decay":True},Xe_cancer,ye_cancer,n=5)
# 0.9121095227289917


# El resultado es la media de rendimientos obtenidos entrenando cada vez con
# todas las particiones menos una, y probando el rendimiento con la parte que
# se ha dejado fuera. Las particiones deben ser aleatorias y estratificadas. 
 
# Si decidimos que es es un buen rendimiento (comparando con lo obtenido para
# otros valores de esos parámetros), finalmente entrenaríamos con el conjunto de
# entrenamiento completo:

# >>> LR16=RegresionLogisticaMiniBatch(batch_tam=16,rate_decay=True)
# >>> LR16.entrena(Xe_cancer,ye_cancer)

# Y daríamos como estimación final el rendimiento en el conjunto de prueba, que
# hasta ahora no hemos usado:
# >>> rendimiento(LR16,Xp_cancer,yp_cancer)
# 0.9203539823008849

#------------------------------------------------------------------------------





# ===================================================
# EJERCICIO 4: APLICANDO LOS CLASIFICADORES BINARIOS
# ===================================================



# Usando los dos modelos implementados en el ejercicio 3, obtener clasificadores 
# con el mejor rendimiento posible para los siguientes conjunto de datos:

# - Votos de congresistas US
# - Cáncer de mama 
# - Críticas de películas en IMDB

# Ajustar los parámetros para mejorar el rendimiento. Si se ha hecho el ejercicio 3, 
# usar validación cruzada para el ajuste (si no, usar el "holdout" del ejercicio 1). 

# Mostrar el proceso realizado en cada caso, y los rendimientos finales obtenidos. 

# Votos
votes_score_cross_val_1 = rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch, {"batch_tam":16,"rate_decay":True},Xe_votos,ye_votos,n=5)
votes_score_cross_val_2 = rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch, {"batch_tam":20},Xe_votos,ye_votos,n=10)
print('VOTOS')
print('Rendimiento obtenido con los parámetros: batch_tam = 16, rate_decay = True, 5 particiones: ', votes_score_cross_val_1) #0.9345323741007194
print('Rendimiento obtenido con los parámetros: batch_tam = 20, 10 particiones: ', votes_score_cross_val_2) #0.9207667731629392
RLMN_votes_16=RegresionLogisticaMiniBatch(batch_tam=16,rate_decay=True)
RLMN_votes_16.entrena(Xe_votos,ye_votos)
votes_score = rendimiento(RLMN_votes_16,Xp_votos,yp_votos)
print('Rendimiento sobre el conjunto de pruebas: ', votes_score)

# Cancer
print('\nCancer')
cancer_score_cross_val_1 = rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch, {"batch_tam":16,"rate_decay":True},Xe_cancer,ye_cancer,n=5)
cancer_score_cross_val_2 = rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch, {"batch_tam":25,"rate_decay":True},Xe_cancer,ye_cancer,n=10)
print('Rendimiento obtenido con los parámetros: batch_tam = 16, rate_decay = True, 5 particiones: ', cancer_score_cross_val_1) #0.8983516483516484
print('Rendimiento obtenido con los parámetros: batch_tam = 25, rate_decay = True, 10 particiones ', cancer_score_cross_val_2) #0.880195599022005
RLMN_cancer_16=RegresionLogisticaMiniBatch(batch_tam=16,rate_decay=True)
RLMN_cancer_16.entrena(Xe_cancer,ye_cancer)
cancer_score = rendimiento(RLMN_cancer_16,Xp_cancer,yp_cancer)
print('Rendimiento sobre el conjunto de pruebas: ', cancer_score)

# IMDB
print('\nIMDB')
imdb_score_cross_val_1 = rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch, {"batch_tam":16,"rate_decay":True},X_train_imdb,y_train_imdb,n=5)
imdb_score_cross_val_2 = rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch, {"batch_tam":25},X_train_imdb,y_train_imdb,n=10)
print('Rendimiento obtenido con los parámetros: batch_tam = 16, rate_decay = True, 5 particiones: ', imdb_score_cross_val_1) #0.7185
print('Rendimiento obtenido con los parámetros: batch_tam = 25, 10 particiones: ', imdb_score_cross_val_2) #0.6927222222222221
RLMN_imdb_16=RegresionLogisticaMiniBatch(batch_tam=16,rate_decay=True)
RLMN_imdb_16.entrena(X_train_imdb,y_train_imdb)
imdb_score = rendimiento(RLMN_imdb_16,X_test_imdb,y_test_imdb)
print('Rendimiento sobre el conjunto de pruebas: ', imdb_score)

# =====================================
# EJERCICIO 5: CLASIFICACIÓN MULTICLASE
# =====================================

# Técnica "One vs Rest" (Uno frente al Resto)
# -------------------------------------------


# Se pide implementar la técnica "One vs Rest" (Uno frente al Resto),
# para obtener un clasificador multiclase a partir del clasificador
# binario definido en el apartado anterior.


#  En concreto, se pide implementar una clase python
#  RegresionLogisticaOvR con la siguiente estructura, y que implemente
#  el entrenamiento y la clasificación siguiendo el método "One vs
#  Rest" tal y como se ha explicado en las diapositivas del módulo.

 

# class RegresionLogisticaOvR():

#    def __init__(self,normalizacion=False,rate=0.1,rate_decay=False,
#                 batch_tam=64):

#          .....
         
#    def entrena(self,entr,clas_entr,n_epochs=1000):

#         ......

#    def clasifica(self,E):


#         ......

class RegresionLogisticaOvR():
    def __init__(self,normalizacion=False,rate=0.1,rate_decay=False, batch_tam=64):
        self.normalizacion = normalizacion
        self.rate = rate
        self.rate_decay = rate_decay
        self.batch_tam = batch_tam
        self.models = []
         
    def entrena(self,entr,clas_entr,n_epochs=1000):
        self.y_class = np.unique(clas_entr)
        for class_value in self.y_class:
            clas_entr_trans = np.where(clas_entr == class_value, 1, 0)
            classifier = RegresionLogisticaMiniBatch(normalizacion=self.normalizacion, vrate=self.rate, rate_decay=self.rate_decay, batch_tam=self.batch_tam)
            classifier.entrena(entr, clas_entr_trans, n_epochs=n_epochs)
            self.models.append(classifier)
        self.isTrain = True

    def clasifica(self,E):
        if not self.isTrain:
            raise ClasificadorNoEntrenado("Primero debe entrenar el modelo")
        E_x = np.copy(E)
        scores = []
        for model in self.models:
            model_proba = model.clasifica_prob(E_x)
            scores.append(model_proba)
        np_scores = np.asarray(scores)
        pred = np.argmax(np_scores, axis=0)
        return self.y_class[pred]


#  Los parámetros de los métodos significan lo mismo que en el
#  apartado anterior.

#  Un ejemplo de sesión, con el problema del iris:


# --------------------------------------------------------------------
# ToDo: DESCOMENTAR!!!
'''Xe_iris,Xp_iris,ye_iris,yp_iris = particion_entr_prueba(X_iris,y_iris,test=1/3)
rl_iris=RegresionLogisticaOvR(rate=0.001,batch_tam=20)
rl_iris.entrena(Xe_iris,ye_iris)
score_iris_e = rendimiento(rl_iris,Xe_iris,ye_iris)
print('score_iris_e: ', score_iris_e)
score_iris_p = rendimiento(rl_iris,Xp_iris,yp_iris)
print('score_iris_p: ', score_iris_p)'''

# In[1] Xe_iris,Xp_iris,ye_iris,yp_iris          
#            =particion_entr_prueba(X_iris,y_iris,test=1/3)

# >>> rl_iris=RL_OvR(rate=0.001,batch_tam=20)

# >>> rl_iris.entrena(Xe_iris,ye_iris)

# >>> rendimiento(rl_iris,Xe_iris,ye_iris)
# 0.9797979797979798

# >>> rendimiento(rl_iris,Xp_iris,yp_iris)
# >>> 0.9607843137254902
# --------------------------------------------------------------------




# ==============================================
# EJERCICIO 6: APLICACION A PROBLEMAS MULTICLASE
# ==============================================


# ---------------------------------------------------------
# 6.1) Conjunto de datos de la concesión de crédito
# ---------------------------------------------------------

# Aplicar la implementación del apartado anterior, para obtener un
# clasificador que aconseje la concesión, estudio o no concesión de un préstamo,
# basado en los datos X_credito, y_credito. Ajustar adecuadamente los parámetros. 

# NOTA IMPORTANTE: En este caso concreto, los datos han de ser transformados, 
# ya que los atributos de este conjunto de datos no son numéricos. Para ello, usar la llamada 
# "codificación one-hot", descrita en el tema "Preprocesado e ingeniería de características".
# Se pide implementar esta transformación (directamete, SIN USAR Scikt Learn ni Pandas). 

def oneHot(X):
    X_code = np.copy(X)
    oneHotEncode = np.zeros((X.shape[0],0))
    for i in range(X.shape[1]):
        column_values = np.unique(X[:,i])
        X_column = np.copy(X_code[:,i])
        for j in range(len(column_values)):
            X_column = np.where(X_column == column_values[j], j, X_column)
        X_column = X_column.astype(np.int32)
        X_column_aux = np.zeros((X.shape[0],len(column_values)), dtype=np.int32)
        X_column_aux[np.arange(X_column.size),X_column] = 1
        oneHotEncode = np.append(oneHotEncode, X_column_aux, axis = 1)
    return oneHotEncode

'''print(X_credito.shape)
X_reshape = np.reshape(X_credito,-1)
nb_classes = X_credito.shape[1]
prueba = np.eye(nb_classes)[X_reshape]
print(prueba[0])
#print(np.unique(X_credito, axis = 0))'''

X_credito_code = oneHot(X_credito)
# ToDo: DESCOMENTAR!!!
'''
Xe_credito,Xp_credito,ye_credito,yp_credito = particion_entr_prueba(X_credito_code,y_credito,test=1/3)
rl_credito=RegresionLogisticaOvR(rate=0.001,batch_tam=20)
rl_credito.entrena(Xe_credito,ye_credito)
#print(X_credito_code.shape)
score_credito_e = rendimiento(rl_credito,Xe_credito,ye_credito)
print('score_credito_e: ', score_credito_e)
score_credito_p = rendimiento(rl_credito,Xp_credito,yp_credito)
print('score_credito_p: ', score_credito_p)'''

# ---------------------------------------------------------
# 6.2) Clasificación de imágenes de dígitos escritos a mano
# ---------------------------------------------------------


#  Aplicar la implementación o implementaciones del apartado anterior, para obtener un
#  clasificador que prediga el dígito que se ha escrito a mano y que se
#  dispone en forma de imagen pixelada, a partir de los datos que están en el
#  archivo digidata.zip que se suministra.  Cada imagen viene dada por 28x28
#  píxeles, y cada pixel vendrá representado por un caracter "espacio en
#  blanco" (pixel blanco) o los caracteres "+" (borde del dígito) o "#"
#  (interior del dígito). En nuestro caso trataremos ambos como un pixel negro
#  (es decir, no distinguiremos entre el borde y el interior). En cada
#  conjunto las imágenes vienen todas seguidas en un fichero de texto, y las
#  clasificaciones de cada imagen (es decir, el número que representan) vienen
#  en un fichero aparte, en el mismo orden. Será necesario, por tanto, definir
#  funciones python que lean esos ficheros y obtengan los datos en el mismo
#  formato numpy en el que los necesita el clasificador. 

#  Los datos están ya separados en entrenamiento, validación y prueba. En este
#  caso concreto, NO USAR VALIDACIÓN CRUZADA para ajustar, ya que podría
#  tardar bastante (basta con ajustar comparando el rendimiento en
#  validación). Si el tiempo de cómputo en el entrenamiento no permite
#  terminar en un tiempo razonable, usar menos ejemplos de cada conjunto.

# Ajustar los parámetros de tamaño de batch, tasa de aprendizaje y
# rate_decay para tratar de obtener un rendimiento aceptable (por encima del
# 75% de aciertos sobre test). 

def readDigitFile(X_data_file_path, y_data_file_path):
    size = 28
    X_data_file = open(X_data_file_path)
    lines = X_data_file.readlines()
    X_data = np.zeros((len(lines),size)).astype(np.int32)
    for i in range(len(lines)):
        line = lines[i]
        line = line.replace('\n', '')
        np_line = np.array(list(line)).reshape(1,size)
        np_line = np.where((np_line == '+') | (np_line == '#'), 1, 0)
        X_data[i] = np_line
    np_numbers = X_data.reshape(int(len(X_data)/28),size*size)
    y_data_file = open(y_data_file_path)
    y_lines = y_data_file.readlines()
    y_lines = [line.replace('\n', '') for line in y_lines]
    y_data = np.array(y_lines)
    return np_numbers, y_data

import os
root = os.getcwd() + '/datos/digitdata/'
X_train, y_train = readDigitFile(root + 'trainingimages', root + 'traininglabels')
X_val, y_val = readDigitFile(root + 'validationimages', root + 'validationlabels')
X_test, y_test = readDigitFile(root + 'testimages', root + 'testlabels')

# Con el conjunto de validacion para ver los parámetros
'''rl_digit_val_1 = RegresionLogisticaOvR(rate=0.001,batch_tam=20)
rl_digit_val_1.entrena(X_val,y_val)
score_digit_p_val_1 = rendimiento(rl_digit_val_1,X_test,y_test)
print('score_digit_p_val_1: ', score_digit_p_val_1)

rl_digit_val_2 = RegresionLogisticaOvR(rate=0.015,batch_tam=80)
rl_digit_val_2.entrena(X_val,y_val)
score_digit_p_val_2 = rendimiento(rl_digit_val_2,X_test,y_test)
print('score_digit_p_val_2: ', score_digit_p_val_2)

rl_digit_val_3 = RegresionLogisticaOvR(rate=0.001,rate_decay=True,batch_tam=20)
rl_digit_val_3.entrena(X_val,y_val)
score_digit_p_val_3 = rendimiento(rl_digit_val_3,X_test,y_test)
print('score_digit_p_val_3: ', score_digit_p_val_3)

# Con el conjunto de entrenamiento
rl_digit=RegresionLogisticaOvR(rate=0.001,rate_decay=True,batch_tam=20)
rl_digit.entrena(X_train,y_train)
score_digit_e = rendimiento(rl_digit,X_train,y_train)
print('score_digit_e: ', score_digit_e)
score_digit_p = rendimiento(rl_digit,X_test,y_test)
print('score_digit_p: ', score_digit_p)'''