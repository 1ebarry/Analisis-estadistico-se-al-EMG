
import os ##ubicacion archivo
import wfdb #señal
import matplotlib.pyplot as plt #graficas
import numpy as np 
from scipy.stats import norm, gaussian_kde
import statistics

ruta = r'C:\Users\Esteban\Pictures\Analisis estadistico señal EMG\emg_neuropathy'
datos, info = wfdb.rdsamp(ruta, sampfrom=50, sampto=1000)  #ubicacion
#print(os.getcwd())
datos = np.array(datos).flatten()  # Convertir a 1D si es necesario
plt.figure(figsize=(10, 5))
plt.plot(datos, label="Señal EMG", color='c')
plt.xlabel("Tiempo (ms)")
plt.ylabel("Amplitud (mV)")  
plt.title("Señal EMG Neuropatía")
plt.legend()
plt.grid()
##grafica la señal

#Media
sumatoriadatos = 0
for i in datos:
    sumatoriadatos += i
#datos sumados
media=sumatoriadatos/info['sig_len']
print(f"Media: {media}")

mean=np.mean(datos)
print(f"Media Numpy: {mean}")



############
#DESVIACION ESTANDAR 
resta=datos-media
#print(resta)
resta2=resta**2
#print(resta2)
sumatoriaresta=0
for i in resta2:
    sumatoriaresta += i    
#print(sumatoriaresta)
S=np.sqrt(sumatoriaresta/(info['sig_len']-1)) ###nan
print(f"Desviacion estandar: {S}")

desviacion_muestral = np.std(datos, ddof=1)  # ddof=1 para muestra
print(f"Desviación estándar Numpy: {desviacion_muestral:.4f}")

############
#COEFICIENTE DE VARIACIÓN
CV =(S/media)*100
print(f"Coeficiente de Variación: {CV}%")

# Calcular coeficiente de variación (en porcentaje)
cv = (desviacion_muestral / mean) * 100
print(f"Coeficiente de Variación Numpy: {cv:.2f}%")
######


# Histograma
plt.figure()
plt.hist(datos, bins=50, edgecolor='black', alpha=1.0, color='orange', density=True)  # Normalizado para densidad
plt.grid()

# Estimación de la densidad mediante gaussian_kde
kde = gaussian_kde(datos.flatten())

# Ajustar los valores de KDE para que alcancen hasta 2.5 en el eje y
scaling_factor = 2.5 / max(kde(datos.flatten()))  # Factor de escalamiento
x_vals = np.linspace(datos.min(), datos.max(), 1000)
plt.plot(x_vals, kde(x_vals) * scaling_factor, color='blue', lw=2, label='Densidad KDE (escalada)')

# Limitar el eje Y
plt.ylim(0, 2.5)

plt.title("Histograma con Función de Probabilidad (KDE)")
plt.xlabel("Amplitud")
plt.ylabel("Densidad de Probabilidad")
plt.legend()
plt.show()

# --- GENERAR RUIDO GAUSSIANO ---
ruido_std = np.std(datos) * 0.3  # 30% de la desviación estándar de la señal
ruido = np.random.normal(0, ruido_std, size=len(datos))  # Ruido gaussiano

# --- SEÑAL CONTAMINADA ---
señal_ruidosa = datos + ruido

# --- GRAFICAR SEÑALES ---
plt.figure(figsize=(12, 5))

plt.subplot(2, 1, 1)
plt.plot(datos, color='c', label="Señal Original")
plt.title("Señal EMG Neuropatia ")
plt.xlabel("Tiempo (ms)")
plt.ylabel("Amplitud (mV)")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(señal_ruidosa, color='red', label="Señal con Ruido Gaussiano")
plt.title("Señal EMG Contaminada con Ruido Gaussiano")
plt.xlabel("Tiempo (ms)")
plt.ylabel("Amplitud (mV)")
plt.legend()
plt.grid()


plt.tight_layout()
plt.show()

# --- CÁLCULO DEL SNR ---
def calcular_snr(señal, ruido):
    potencia_señal = np.mean(señal_ruidosa**2)  # Potencia de la señal
    potencia_ruido = np.mean(ruido**2)  # Potencia del ruido
    snr = 10 * np.log10(potencia_señal / potencia_ruido)  # SNR en dB
    return snr

snr_original = calcular_snr(datos, np.zeros_like(datos))  # SNR sin ruido
snr_ruidoso = calcular_snr(datos, ruido)  # SNR con ruido

print(f"SNR con ruido Gaussiano: {snr_ruidoso:.2f} dB")



datos = datos.flatten()  # Convertir a una dimensión

# Frecuencia de muestreo (obtenida del archivo de información)
fs = info['fs']  # Frecuencia de muestreo en Hz

# Tiempo asociado a la señal
t = np.arange(0, len(datos)) / fs

# Parámetros del ruido de red
frecuencia_red = 60  # Frecuencia del ruido (60 Hz)
amplitud_ruido = 0.8  # Amplitud del ruido de red
ruido_red = amplitud_ruido * np.sin(2 * np.pi * frecuencia_red * t)

# Contaminación con ruido de red
datos_contaminados_red = datos + ruido_red

# Parámetros del ruido de pulso
amplitud_ruido_min = -2.5  # Valor mínimo del impulso
amplitud_ruido_max = 2.5   # Valor máximo del impulso
ruido_pulso = np.zeros_like(datos)
num_impulsos = int(0.05 * len(datos))  # 5% de la longitud total de la señal
indices_impulso = np.random.choice(len(datos), size=num_impulsos, replace=False)
ruido_pulso[indices_impulso] = np.random.uniform(amplitud_ruido_min, amplitud_ruido_max, size=num_impulsos)

# Contaminación con ruido de pulso
datos_contaminados_pulso = datos + ruido_pulso

# Gráficas
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t * 1000, datos, label="Señal EMG Original", color='c')
plt.title("Señal EMG Original")
plt.xlabel("Tiempo (ms)")
plt.ylabel("Amplitud (mV)")
plt.legend()
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(t * 1000, datos_contaminados_red, label="Señal EMG Contaminada (60 Hz)", color='orange')
plt.title("Señal EMG Contaminada con Ruido de Red (60 Hz)")
plt.xlabel("Tiempo (ms)")
plt.ylabel("Amplitud (mV)")
plt.legend()
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(t * 1000, datos_contaminados_pulso, label="Señal EMG Contaminada con Ruido de Pulso", color='purple')
plt.title("Señal EMG Contaminada con Ruido de Pulso")
plt.xlabel("Tiempo (ms)")
plt.ylabel("Amplitud (mV)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
# Cálculo del SNR para ruido de red
P_signal_red = np.mean(datos_contaminados_red ** 2)  # Potencia de la señal
P_noise_red = np.mean(ruido_red ** 2)  # Potencia del ruido de red
SNR_red = 10 * np.log10(P_signal_red / P_noise_red)
print(f"SNR con Ruido de Red: {SNR_red:.2f} dB")

# Cálculo del SNR para ruido de pulso
P_signal_pulso = np.mean(datos_contaminados_pulso ** 2)
P_noise_pulso = np.mean(ruido_pulso ** 2)
SNR_pulso = 10 * np.log10(P_signal_pulso / P_noise_pulso)
print(f"SNR con Ruido de Pulso: {SNR_pulso:.2f} dB")

###############Evaluacion de amplitud

A = 2  # Factor de amplificación
datos_amplificados = A * datos

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(datos_amplificados, color='c', label="Señal Amplificada")
plt.title("Señal EMG Neuropatía (Amplificada)")
plt.xlabel("Tiempo (ms)")
plt.ylabel("Amplitud (mV)")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(señal_ruidosa, color='red', label="Señal con Ruido Gaussiano")
plt.title("Señal EMG Contaminada con Ruido Gaussiano Amplificada")
plt.xlabel("Tiempo (ms)")
plt.ylabel("Amplitud (mV)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# --- CÁLCULO DEL SNR ---
def calcular_snr(señal, ruido):
    potencia_señal = np.mean(señal**2)  # Potencia de la señal original
    potencia_ruido = np.mean(ruido**2)  # Potencia del ruido
    snr = 10 * np.log10(potencia_señal / potencia_ruido)  # SNR en dB
    return snr

snr_nuevo = calcular_snr(datos_amplificados, ruido)
print(f"SNR ruido Gaussiano con Amplitud Aplicada: {snr_nuevo:.2f} dB")


datos_con_red = datos_amplificados + ruido_red
datos_con_pulso = datos_amplificados + ruido_pulso

def calcular_snr(señal_original, ruido):
    P_signal = np.mean(señal_original ** 2)  # Potencia de la señal original
    P_noise = np.mean(ruido ** 2)  # Potencia del ruido
    return 10 * np.log10(P_signal / P_noise)

SNR_red = calcular_snr(datos_amplificados, ruido_red)
SNR_pulso = calcular_snr(datos_amplificados, ruido_pulso)

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t * 1000, datos_amplificados, label="Señal EMG Amplificada", color='c')
plt.title("Señal EMG Amplificada")
plt.xlabel("Tiempo (ms)")
plt.ylabel("Amplitud (mV)")
plt.legend()
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(t * 1000, datos_con_red, label="Señal con Ruido de Red (60 Hz)", color='orange')
plt.title("Señal EMG Contaminada con Ruido de Red Amplificada")
plt.xlabel("Tiempo (ms)")
plt.ylabel("Amplitud (mV)")
plt.legend()
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(t * 1000, datos_con_pulso, label="Señal con Ruido de Pulso", color='purple')
plt.title("Señal EMG Contaminada con Ruido de Pulso Amplificada")
plt.xlabel("Tiempo (ms)")
plt.ylabel("Amplitud (mV)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

print(f"SNR con Ruido de Red Amplitud aplicada: {SNR_red:.2f} dB")
print(f"SNR con Ruido de Pulso Amplitud Aplicada: {SNR_pulso:.2f} dB")