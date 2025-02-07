
<h1 align="center">Análisis Estadistico y Comportamiento SNR  de una Señal EMG</h1>
<p align="center"> </p>
<p align="center"><img src="https://www.geriatricarea.com/wp-content/uploads/2022/11/geriatricarea-proyecto-SOLFIS-IBV.jpg"/></p> 

## Introducción
El siguiente proyecto consiste en el analisis fisiológico correspondiente a una señal de electromiografía (EMG) asociada a una neuropatía, Este código consta con el cálculo de los estadisticos fundamentales, tanto descriptivos como con funciones predeterminadas y también una breve contaminación de ruido sobre la señal fisiológica mediante la relación señal-ruido (SNR) lo cual permitió comprender la calidad de la señal y su posible optimización para aplicaciones biomédicas.


## Contexto 
La señal fisiológica electromiográfica (EMG)  corresponde a un registro en el músculo tibial anterior de un paciente de 62 años, sexo masculino, con diagnóstico de dolor lumbar crónico y neuropatía por radiculopatía derecha en la raíz nerviosa L5.
La señal fue capturada mediante un electrodo concéntrico de 25 mm, mientras el paciente realizaba una dorsiflexión suave del pie contra resistencia, seguida de relajación. 
<p align="center"><img src="https://www.unipamplona.edu.co/unipamplona/portalIG/home_17/recursos/01_general/14092010/daqsenalesemg.png"/></p>

## Guía de usuario
En el encabezado del código encontramos la inicialización de las librerias correspondientes para el correcto funcionamiento del código
 ```pyton 
import os  # ubicacion archivo
import wfdb  # señal
import matplotlib.pyplot as plt  # graficas
import numpy as np # operaciones matemáticas 
from scipy.stats import norm, gaussian_kde # estadísticas avanzadas y Función de Probabilidad
import statistics # cálculos estadísticos básicos
```
Es importante modificar la ruta de ubicación, se aconceja tener los archivos plt.plot, plt.xlabel,plt.ylabel,plt.titleemg_neuropathy.dat" y "emg_neuropathy.hea" junto al archivo del código en una misma carpeta para su correcta compilación
 ```pyton
ruta = r'C:\Users\Esteban\Pictures\Analisis estadistico señal EMG\emg_neuropathy'
datos, info = wfdb.rdsamp(ruta, sampfrom=50, sampto=1000)  #ubicacion
```
Para que nuestra señal sea gráficada correctamente es necesario usar los comandos " np.array"  para que los datos se puedan expresar gráficamente y comandos "plt. " para crear cuadricula, nombrar ejes y agregar un titulo para el gráfico.
```pyton
datos = np.array(datos).flatten()  # Convertir a 1D si es necesario
plt.figure(figsize=(10, 5))
plt.plot(datos, label="Señal EMG", color='c')
plt.xlabel("Tiempo (ms)")
plt.ylabel("Amplitud (mV)")  
plt.title("Señal EMG Neuropatía")
plt.legend()
plt.grid()
```
Continuamos con los estadisticos descriptivos correspondientes para la media, desviación estandar y coeficiente de variación, los cuales consisten en la operación básica de estas medidas
```pyton
#Media
sumatoriadatos = 0
for i in datos:
    sumatoriadatos += i
media=sumatoriadatos/info['sig_len']
print(f"Media: {media}")
```
<p align="center"><img src="https://www.masscience.com/wp-content/uploads/2019/12/formula-para-calcular-la-media.png"/></p>

```pyton
#DESVIACION ESTANDAR 
resta=datos-media
#print(resta)
resta2=resta**2
#print(resta2)
sumatoriaresta=0
for i in resta2:
    sumatoriaresta += i    
#print(sumatoriaresta)
S=np.sqrt(sumatoriaresta/(info['sig_len']-1)) 
print(f"Desviacion estandar: {S}")
```
<p align="center"><img src="https://economipedia.com/wp-content/uploads/Formula-Desviacion-Tipica.jpg"/></p>

```pyton
#COEFICIENTE DE VARIACIÓN
CV =(S/media)*100
print(f"Coeficiente de Variación: {CV}%")
```
<p align="center"><img src="https://economipedia.com/wp-content/uploads/coeficiente-de-variacion-formula.png"/></p>

Continuamos con los estadisticos realizados con funciones predefinidas correspondientes para la media, desviación estandar y coeficiente de variación, donde usamos la biblioteca numpy la cual nos proporciona calculos eficientes para ciertas operaciones matemáticas
```pyton
#MEDIA
mean=np.mean(datos)
print(f"Media Numpy: {mean}")
```

```pyton
#DESVIACION ESTANDAR
desviacion_muestral = np.std(datos, ddof=1)  # ddof=1 para muestra
print(f"Desviación estándar Numpy: {desviacion_muestral:.4f}")
```

```pyton
#COEFICIENTE DE VARIACION
cv = (desviacion_muestral / mean) * 100
print(f"Coeficiente de Variación Numpy: {cv:.2f}%")
```
Para el histograma se uso la herramienta graficadora plt. donde los datos de la señal son guardados por intervalos y renombrados "bins" en este caso se tienen 50 intervalos. Por otra parte se importa la biblioteca "gaussian_kde" para realizar una estimación del comportamiento de los intervalos bins y graficar una tendencía los mas cercana posible(Función de probabilidad), como se muestra en la imagen
```pyton
# Histograma
plt.figure()
plt.hist(datos, bins=50, edgecolor='black', alpha=1.0, color='orange', density=True)  # Normalizado para densidad
plt.grid()

# Estimación de la densidad mediante gaussian_kde
kde = gaussian_kde(datos.flatten())
# Ajustar los valores de KDE para que alcancen hasta 2.5 en el eje y
scaling_factor = 2.5 / max(kde(datos.flatten()))  # scaling_factor para calcular el comportamiento de los datos  
x_vals = np.linspace(datos.min(), datos.max(), 1000) # genera 1000 puntos entre los espacios de los datos para lograr graficar Kde
plt.plot(x_vals, kde(x_vals) * scaling_factor, color='blue', lw=2, label='Densidad KDE (escalada)') #Grafica y ajusta la curva Kde
```
<p align="center"><img src="https://github.com/1ebarry/Analisis-estadistico-se-al-EMG/blob/main/Figure%202025-02-06%20212910.png?raw=true"/></p>

Por ultimo tenemos el codigo respectivo para la contaminación de nuestra señal EMG, se realiza con tres tipos de ruido y por tipo dos pruebas, las cuales consisten en primer lugar el SNR con ruido unicamente y en segundo lugar el SNR con ruido y amplitúd aplicada

## Prueba 1
# Ruido Gaussiano
```pyton
# --- GENERAR RUIDO GAUSSIANO ---
ruido_std = np.std(datos) * 0.3  # 30% de la desviación estándar de la señal
ruido = np.random.normal(0, ruido_std, size=len(datos))  # Ruido gaussiano

# --- SEÑAL CONTAMINADA ---
señal_ruidosa = datos + ruido
```
# Ruido de Red 

```pyton
frecuencia_red = 60  # Frecuencia del ruido (60 Hz)
amplitud_ruido = 0.8  # Amplitud del ruido de red
ruido_red = amplitud_ruido * np.sin(2 * np.pi * frecuencia_red * t)

# Contaminación con ruido de red
datos_contaminados_red = datos + ruido_red
```
# Ruido de Pulso
```pyton
# Parámetros del ruido de pulso
amplitud_ruido_min = -2.5  # Valor mínimo del impulso
amplitud_ruido_max = 2.5   # Valor máximo del impulso
ruido_pulso = np.zeros_like(datos)
num_impulsos = int(0.05 * len(datos))  # 5% de la longitud total de la señal
indices_impulso = np.random.choice(len(datos), size=num_impulsos, replace=False)
ruido_pulso[indices_impulso] = np.random.uniform(amplitud_ruido_min, amplitud_ruido_max, size=num_impulsos)

# Contaminación con ruido de pulso
datos_contaminados_pulso = datos + ruido_pulso
```
# Cálculo SNR
```pyton
# --- CÁLCULO DEL SNR ---
def calcular_snr(señal, ruido):
    potencia_señal = np.mean(señal_ruidosa**2)  # Potencia de la señal
    potencia_ruido = np.mean(ruido**2)  # Potencia del ruido
    snr = 10 * np.log10(potencia_señal / potencia_ruido)  # SNR en dB
```
<p align="center"><img src="https://image.slidesharecdn.com/sesion05-estadisticaensenales-130528192513-phpapp02/85/Sesion-05-Estadistica-en-senales-19-320.jpg"/></p>

## Prueba 2
Los datos de la señal EMG son multiplicados por el factor de amplificación = 2 
# Ruido Gaussiano amplitud aplicada
```pyton
plt.subplot(2, 1, 2)
plt.plot(señal_ruidosa, color='red', label="Señal con Ruido Gaussiano")
plt.title("Señal EMG Contaminada con Ruido Gaussiano Amplificada")
plt.xlabel("Tiempo (ms)")
plt.ylabel("Amplitud (mV)")
plt.legend()
plt.grid()
```
# Ruido de red amplitud aplicada
```pyton
plt.subplot(3, 1, 2)
plt.plot(t * 1000, datos_con_red, label="Señal con Ruido de Red (60 Hz)", color='orange')
plt.title("Señal EMG Contaminada con Ruido de Red Amplificada")
plt.xlabel("Tiempo (ms)")
plt.ylabel("Amplitud (mV)")
plt.legend()
plt.grid()
```
# Ruido de pulso amplitud aplicada
```pyton
plt.subplot(3, 1, 3)
plt.plot(t * 1000, datos_con_pulso, label="Señal con Ruido de Pulso", color='purple')
plt.title("Señal EMG Contaminada con Ruido de Pulso Amplificada")
plt.xlabel("Tiempo (ms)")
plt.ylabel("Amplitud (mV)")
plt.legend()
plt.grid()
```
## Resultados 
Al compilar este código usted deberá obtener : 
# Señal EMG 
<p align="center"><img src="https://github.com/1ebarry/Analisis-estadistico-se-al-EMG/blob/main/SE%C3%91AL%20EMG.png?raw=true"/></p>

<p align="center"><img src=""/></p>

# Histograma con función de probabilidad
<p align="center"><img src="https://github.com/1ebarry/Analisis-estadistico-se-al-EMG/blob/main/HISTOGRAMA%20CON%20FUNCION%20DE%20PROBABILIDAD.png?raw=true"/></p>

# Señal EMG contaminada por ruido Gaussiano, red y pulso
<p align="center"><img src="https://github.com/1ebarry/Analisis-estadistico-se-al-EMG/blob/main/SE%C3%91AL%20EMG%20CON%20RUIDO%20GAUSSIANO.png?raw=true"/></p>

<p align="center"><img src="https://github.com/1ebarry/Analisis-estadistico-se-al-EMG/blob/main/SE%C3%91AL%20EMG%20CON%20RUIDO%20DE%20RED%20Y%20PULSO.png?raw=true"/></p>

# Señal EMG con ruidos (Amplificada)
<p align="center"><img src="https://github.com/1ebarry/Analisis-estadistico-se-al-EMG/blob/main/SE%C3%91AL%20EMG%20AMPLIFICADA%20Y%20SE%C3%91AL%20RUIDO%20GAUSSIANO%20AMPLIFICADA.png?raw=true"/></p>

<p align="center"><img src="https://github.com/1ebarry/Analisis-estadistico-se-al-EMG/blob/main/SE%C3%91AL%20RUIDO%20DE%20RED%20Y%20PULSO%20AMPLIFICADAS.png?raw=true"/></p>

## Información adicional
Para finalizar es bueno que tenga presente  optimizar el SNR en las señales biomédicas, teniendo en cuenta que el rango ideal según la literatura para una señal EMG de calidad aceptable debe estar entre 15 dB y 30 dB  para garantizar mediciones precisas y representaciones gráficas fieles de la señal capturada, puesto que un valor elevado de SNR implica que la señal sea más clara en comparación con el ruido y un SNR bajo dificulta la identificación de características relevantes debido a la presencia dominante del ruido
## Bibliografía
[1] Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220.
## Licencia 
DOI (version 1.0.0):
https://doi.org/10.13026/C24S3D

Temas:
neuropatía /
electromiografía

## Literatura Sugerida
1.Kimura J. Electrodiagnóstico en enfermedades de los nervios y músculos: principios y práctica , 3.ª edición. Nueva York, Oxford 
  University Press, 2001.
2.Reaz MBI, Hussain MS y Mohd-Yasin F. Técnicas de análisis de señales EMG: detección, procesamiento, clasificación y aplicaciones. 
  Biol. Proced. Online 2006; 8 (1): 11-35.

