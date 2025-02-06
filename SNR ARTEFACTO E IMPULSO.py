import os  # Ubicación archivo
import wfdb  # Señal
import matplotlib.pyplot as plt  # Gráficas
import numpy as np
from scipy.stats import gaussian_kde

# Leer la señal EMG desde archivo
datos, info = wfdb.rdsamp('C:/Users/Esteban/Pictures/emg/emg_neuropathy', sampfrom=50, sampto=1000)
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
amplitud_ruido_min = -1.5  # Valor mínimo del impulso
amplitud_ruido_max = 1.5   # Valor máximo del impulso
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

# Calcular estadísticas
# Media
sumatoriadatos = 0
for i in datos:
    sumatoriadatos += i
# Datos sumados
media = sumatoriadatos / info['sig_len']
print(f"Media: {media}")

mean = np.mean(datos)
print(f"Media Numpy: {mean}")

# Desviación estándar
resta = datos - media
resta2 = resta ** 2
sumatoriaresta = 0
for i in resta2:
    sumatoriaresta += i    
S = np.sqrt(sumatoriaresta / (info['sig_len'] - 1))  # nan
print(f"Desviación estándar: {S}")

desviacion_muestral = np.std(datos, ddof=1)  # ddof=1 para muestra
print(f"Desviación estándar Numpy: {desviacion_muestral:.4f}")

# Coeficiente de variación
CV = (S / media) * 100
print(f"Coeficiente de Variación: {CV}%")

cv = (desviacion_muestral / mean) * 100
print(f"Coeficiente de Variación Numpy: {cv:.2f}%")

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

# Histograma de la señal original
plt.figure()
plt.hist(datos, bins=50, edgecolor='black', alpha=1.0, color='orange', density=True)
plt.grid()

# Estimación de la densidad mediante gaussian_kde
kde = gaussian_kde(datos)
scaling_factor = 2.5 / max(kde(datos))
x_vals = np.linspace(datos.min(), datos.max(), 1000)
plt.plot(x_vals, kde(x_vals) * scaling_factor, color='blue', lw=2, label='Densidad KDE (escalada)')
plt.ylim(0, 2.5)
plt.title("Histograma con Función de Probabilidad (KDE)")
plt.xlabel("Amplitud")
plt.ylabel("Densidad de Probabilidad")
plt.legend()
plt.show()
