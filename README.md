# **Ingeniero de datos en NEQUI: Prueba Técnica**

## Alcance del proyecto y captura de datos

Para la realización de este proyecto se implementaron los datos generados por la simulación de PaySim [[1]](https://www.kaggle.com/datasets/ealaxi/paysim1). Este proyecto es el desarrollo de una aplicación para simular datos para la detección de fraude. 

Los valores de simulación de fraude se utilizarían en diversos contextos para evaluar y prevenir el fraude en diferentes sistemas, industrias y transacciones. La aplicación para la cual se implementa los valores se puede definir como un proceso para evaluación de sistemas de detección de fraude. En este caso el objetivo es construir un servicio en donde cada vez que se detecta una transacción, se puede enviar una solicitud para generar una detección de si la transacción actual es un fraude o no y guardarla como información historica para reentrenamiento del modelo.

Para este tipo de proyecto, los tipos de datos que se manejan, como se vera mas adelante, tienen un desbalance de información en cuanto a si la transacción es un fraude o no. Para este proposito, se implementa un modelo de XGBoost para obtener un valor de [AUC](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc?hl=es-419#:~:text=AUC%20represents%20the%20probability%20that,has%20an%20AUC%20of%201.0.) de aproximadamente 0.934 desde el notebook de calibración N_Model_calibration.ipynb.


## Proyecto de detección de Fraude

Generalmente, la información real de transacción de este tipo es resguardad y por eso se recurre a implementar valores de simulación para realizar modelos y pruebas de conceptos.

En este caso, la información que nos entrega PaySim [[1]](https://www.kaggle.com/datasets/ealaxi/paysim1) puede ser resumida en las siguientes columnas:

* Type - CASH-IN, CASH-OUT, DEBIT, PAYMENT y TRANSFER.

* amount - cantidad de la transacción en moneda local.

* nameOrig - cliente que inició la transacción.

* oldbalanceOrg - saldo inicial antes de la transacción.

* newbalanceOrig - nuevo saldo después de la transacción.

* nameDest - cliente que recibe la transacción.

* oldbalanceDest - saldo inicial del destinatario antes de la transacción. Tenga en cuenta que no hay información para los clientes que comienzan con M (Comerciantes).

* newbalanceDest - nuevo saldo del destinatario después de la transacción. Tenga en cuenta que no hay información para los clientes que comienzan con M (Comerciantes).

* isFraud - Estas son las transacciones realizadas por agentes fraudulentos dentro de la simulación. En este conjunto de datos específico, el comportamiento fraudulento de los agentes tiene como objetivo obtener beneficios tomando el control de las cuentas de los clientes y tratar de vaciar los fondos transfiriéndolos a otra cuenta y luego retirándolos del sistema.

* isFlaggedFraud - El modelo de negocio tiene como objetivo controlar las transferencias masivas de una cuenta a otra y detectar intentos ilegales. Un intento ilegal en este conjunto de datos es un intento de transferir más de 200.000 en una sola transacción.


## Explorar y evaluar los datos, el EDA.

Para el analisis de exploración de datos se implementa el notebook N_EDA.ipynb. La información de los datos se puede ver en la siguiente imagen:


![datahead](https://github.com/jmpt97/PruebaN/blob/main/Imagenes/datahead.png?raw=true)

El proceso para realizar la exploración y limpieza de datos sigue los siguientes pasos:

    1. Identificación y manejo de valores faltantes.
    2. Eliminación de valores duplicados
    3. Corrección de errores de formato.    
    4. Filtrado de datos irrelevantes.    
    5. Estandarización y normalización de datos.


Para el caso de los valores de este proyecto, se realiza las verificaciones necesarias y se hacen los pasos a seguir.

Para el manejo de valores faltantes primero se realiza la verificación de estos mismos: 

![datanull](https://github.com/jmpt97/PruebaN/blob/main/Imagenes/datanull.png?raw=true)

Esto se puede explicar debido a que los valores que se implementan son provenientes de una simulación. Normalmente, dependiendo del tipo de datos se puede hacer una eliminación de la fila completa de datos.


Adicional a esto se realiza la verificación de valores duplicados y no se encuentran. Esto explicado por el origen de los valores.

Sabiendo la descripción de las columnas y revisando la siguiente imagen:

![datainfo](https://github.com/jmpt97/PruebaN/blob/main/Imagenes/datainfo.png?raw=true)

Se puede encontrar el formato de los valores de las columnas. Esto nos puede confirmar para saber el tipo de formato y si existen errores de formato.


Para el filtrado de datos irrelevantes o incorrectos se realiza analisis de la cantidad de datos que puedan existir en el dataframe. 


En este caso se realiza analisis de varías columnas de los datos:

Para la columna de nameOrig se encuentra la siguiente información:

![namecountnameorg](https://github.com/jmpt97/PruebaN/blob/main/Imagenes/namecountnameorg.png?raw=true)

Para la columna de nameDest se encuentra la siguiente información:

![datacountnamedest](https://github.com/jmpt97/PruebaN/blob/main/Imagenes/datacountnamedest.png?raw=true)

Debido a que la columna objetiva para entrenar el modelo es isFraid se puede sacar información de la siguiente imagen:


![datacountfraud](https://github.com/jmpt97/PruebaN/blob/main/Imagenes/datacountfraud.png?raw=true)

Con esta información se puede concluir que con este tipo de información se presenta un desbalance de información.

Para la columna de tipo de dato se puede encontrar la siguiente información:

![datacounttype](https://github.com/jmpt97/PruebaN/blob/main/Imagenes/datacounttype.png?raw=true)

Debido al desbalance de la información, se realiza la siguiente gráfica para entender el porcentaje de información con respecto a la cantidad total de los valores:

![graph_type_transaction](https://github.com/jmpt97/PruebaN/blob/main/Imagenes/graph_type_transaction.png?raw=true)

Junto con esta información y la variable objetivo, se puede obtener la cantidad de valores por tipo	de transacción que son Fraude.

![graph_type_transactionFraud](https://github.com/jmpt97/PruebaN/blob/main/Imagenes/graph_type_transactionFraud.png?raw=true)

Junto con esta información y la variable objetivo y con la variable de isFlaggedFraud, se puede obtener la cantidad de valores por tipo	de transacción que son Fraude y las variables isFlaggedFraud.

![graph_type_transactionFraudisFlagged](https://github.com/jmpt97/PruebaN/blob/main/Imagenes/graph_type_transactionFraudisFlagged.png?raw=true)

Ahora, con la gráfica anterior se puede obtener la matriz de confusión para los valores de Fraud y isFlaggedFraud:

![confMatrix](https://github.com/jmpt97/PruebaN/blob/main/Imagenes/confMatrix.png?raw=true)

En este caso, con esta información se podría verificar el funcionamiento del modelo para detectar isFlaggedFraud. Esta variable se implementa dentro de la simulación como una detección previa de fraude cuando el proceso detecta ciertos parametros dentro de la transaction.


Para saber la cantida maxima de transacción cuando se detecta un fraude se puede observar la siguiente imagen:

![fraud_max_amount](https://github.com/jmpt97/PruebaN/blob/main/Imagenes/fraud_max_amount.png?raw=true)

Ahora, con el objetivo de poder construir un modelo de datos se decide agregar las siguientes columnas:

![adicion_col](https://github.com/jmpt97/PruebaN/blob/main/Imagenes/adicion_col.png?raw=true)
## Definir el modelo de datos

En la siguiente imagen se puede observar el modelo de base de datos con el objetivo de poder entender las relaciones entre las diferentes tablas, las llaves primarias y las llaves foraneas.

![db_class](https://github.com/jmpt97/PruebaN/blob/main/Arquitectura/db_class.png?raw=true)
En este caso se puede observar el tipo de relacion que existe entre las columnas y las tablas para las llaves foraneas. 

## diccionario de datos para cada tabla

Diccionario de datos para las tablas involucradas en el sistema:

##### **dbo.Transactions**

| Nombre de columna  | Tamaño | Tipo de dato  | Descripción                                                                                                           |
|--------------------|--------|---------------|-----------------------------------------------------------------------------------------------------------------------|
| TransactionID      | INT    | Identidad     | Identificador único e incremental para cada transacción.                                                               |
| HoursPast          | INT    | Entero        | Horas transcurridas desde algún punto de referencia hasta el momento de la transacción.                                |
| OperationTypeID    | INT    | Entero        | Identificador del tipo de operación realizada en la transacción.                                                       |
| amount             | DECIMAL(10,2) | Decimal   | Monto de la transacción.                                                                                              |
| nameOrig           | VARCHAR(100)  | Cadena     | Nombre o identificador del origen de la transacción.                                                                   |
| oldbalanceOrg      | DECIMAL(10,2) | Decimal   | Saldo inicial del origen antes de la transacción.                                                                      |
| newbalanceOrig     | DECIMAL(10,2) | Decimal   | Saldo resultante del origen después de la transacción.                                                                 |
| nameDest           | VARCHAR(100)  | Cadena     | Nombre o identificador del destino de la transacción.                                                                  |
| oldbalanceDest     | DECIMAL(10,2) | Decimal   | Saldo inicial del destino antes de la transacción.                                                                     |
| newbalanceDest     | DECIMAL(10,2) | Decimal   | Saldo resultante del destino después de la transacción.                                                                |
| currencyOrigID     | INT    | Entero        | Identificador de la moneda utilizada en el origen de la transacción.                                                  |
| currencyDestID     | INT    | Entero        | Identificador de la moneda utilizada en el destino de la transacción.                                                 |
| CreationDate       | DATETIME2(0)  | Fecha y hora | Fecha y hora de creación de la transacción.                                                                           |


##### **dbo.Currency**

| Nombre de columna | Tamaño       | Tipo de dato  | Descripción                                                          |
|-------------------|--------------|---------------|----------------------------------------------------------------------|
| CurrencyID        | INT          | Identidad     | Identificador único e incremental para cada moneda.                    |
| Description       | VARCHAR(1000)| Cadena        | Descripción de la moneda.                                             |
| Value             | VARCHAR(1000)| Cadena        | Valor de la moneda.                                                   |
| CreationDate      | DATETIME2(0) | Fecha y hora  | Fecha y hora de creación de la moneda.                                 |


##### **dbo.OperationType**


| Nombre de columna | Tamaño       | Tipo de dato  | Descripción                                                          |
|-------------------|--------------|---------------|----------------------------------------------------------------------|
| OperationTypeID   | INT          | Identidad     | Identificador único e incremental para cada tipo de operación.        |
| Description       | VARCHAR(1000)| Cadena        | Descripción del tipo de operación.                                    |
| Value             | VARCHAR(1000)| Cadena        | Valor del tipo de operación.                                          |
| CreationDate      | DATETIME2(0) | Fecha y hora  | Fecha y hora de creación del tipo de operación.                       |


##### **dbo.ProcessedTransactions**

| Nombre de columna      | Tamaño        | Tipo de dato  | Descripción                                                          |
|------------------------|---------------|---------------|----------------------------------------------------------------------|
| ProcessedTransactionID | INT           | Identidad     | Identificador único e incremental para cada transacción procesada.    |
| TransactionID          | INT           | Entero        | Identificador de la transacción asociada al registro procesado.       |
| isFraud                | BIT           | Booleano      | Indica si la transacción es fraudulenta (1) o no (0).                  |
| CreationDate           | DATETIME2(0)  | Fecha y hora  | Fecha y hora de creación del registro procesado.                      |


## Arquitectura de solución

Para la arquitectura de la solución se construye la siguiente imagen:

![azure_arch_solution](https://github.com/jmpt97/PruebaN/blob/main/Arquitectura/azure_arch_solution.png?raw=true)
Para esta solución se implementa diferentes herramientas de Microsoft Azure. 

Cabe aclarar que los elementos que estan rodeados por una linea puntada son procesos que existen pero que son necesarios para la construcción y analisis inicial de la solución propuesta. Los elementos que estan en un rectangulo gris, son elementos que no existen pero se plantean. Y los demas elementos existen construidos como medios para la ejecución de los otros elementos.

Estas herramientas son:

1. LogicApp: Debido a que el objetivo es recibir información y poder orquestar los demas procesos, se implementa una logicaApp que se puede ejecutar cada vez que se reciba una solicitud de transacción.
2. MSSQL Server: Servidor de base de datos para guardar información de inteligencia de negocios y transacciones durante un tiempo estipulado.
2. Datafactory: Herramienta para realizar la ejecución de proceos de pipelines.
3. DataStorage: Solución para guardar información de valores historicos para entrenamiento y poder aliviar el procesamiento de la base de datos, guardar información de modelos e información de configuración.
4. Databricks: Plataforma de ejecución para procesos de python con el objetivo de poder realizar los entrenamientos y pronosticos de los modelos.
5. DataFlow: Proceso de flujo de datos para limpieza, tratamiento de variables y reemplazo de información para llaves foraneas.


## Consideraciones

Debido la cantidad de valores que se estan procesando y la misma arquitectura del proyecto, la solución permitiria hacer modificaciones para escalar de manera horizontal y vertical de manera efectiva. 

Si la cantidad de valores aumentaran consideralmente, la solución se construyo de tal manera que las calibraciones guardaran la información en el DataStorage en formato parquet para optimizar el procesamiento y escritura de los datos de la base de datos. Adicionalmente, se tendría que construir un proceso de guardado y borrado de información para el datalake. Para estos procesos se tendría que construir un pipeline adicional que este realizando este proceso periodicamente.

La solución, es construida para recibir información constantemente desde una solicitud http pero para verificación de procesos se hizo las modificaciones para leer dataframes y poder simular el proceso de una solicitud. Revisando el diagrama de la arquitectura, el pipeline de DataQuality y ForecastFraud se podría modificar para tener un trigger periodico con la resolución minima desde lo que permite la creación de Datafactory.

Dependiendo de la cantidad de usuarios que van a acceder a la base de datos, es necesario mantener una optimización adecuada de la base de datos. Para esto se hace la limpieza constante de los valores de la tabla de dbo.Transactions y dbo.ProcessedTransactions. Aun asi, sea la necesidad de acceder a los datos, se propone crear vistas con índices agrupados de la vista de dbo.Transactions y dbo,ProcessedTransactions que puedan ser de utilidad para generar reportes.

Finalmente, si el objetivo es realizar una analítica en tiempo real, se puede modificar el proceso de ejecuión de Databricks para incluir dentro del dataflow una ejecución del Azure synapse para permitir la ejecuión del pronosticos al momento de hacer la solicitud de http a la LogicApp y poder devovler el resultado de tal valor. Esto se consideraría como la unica modifcación debido a la forma en la que se construyo la solución.



## References
[1]
E. A. Lopez-Rojas , A. Elmir, and S. Axelsson. "PaySim: A financial mobile money simulator for fraud detection". In: The 28th European Modeling and Simulation Symposium-EMSS, Larnaca, Cyprus. 2016
