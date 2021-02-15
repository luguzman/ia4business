# Inteligencia Artifical aplicada a Negocios y Empresas
# Maximización de beneficios de una empresa de venta online con Muestreo de Thompson

# Importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import random

# Configuración de los parámetros
N = 10000   # Número total de rondas(clientes) que se convierte o no en "premium"
d = 9       # Número total de estrategias

# Creación de la simulación
# conversion_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
conversion_rates = [0.05, 0.13, 0.09, 0.16, 0.11, 0.04, 0.20, 0.08, 0.01]
X = np.array(np.zeros([N, d]))
# Iteramos sobre cada una de las celdas del array que creamos (X) y denotaremos que la celda cambiara de 
# de valor si el usuario i se convirtió en premium
for i in range(N):
    for j in range(d):
        if np.random.rand() <= conversion_rates[j]:
            X[i,j] = 1
# El resultado obtenido nos devuelve una matriz diciendonos ante que estrategias el usuario i se 
# transformo a prime    
            
# Implementación de la Selección Aleatoria y el Muestreo de Thompson
strategies_selected_rs = []
strategies_selected_ts = []
total_reward_rs = 0
total_reward_ts = 0
number_of_rewards_1 = [0] * d
number_of_rewards_0 = [0] * d
rewards_strategies = [0] * d
regret_rs = []
regret_ts = []
for n in range(0, N):
    # Paso1 con Selección Aleatoria
    strategy_rs = random.randrange(d)
    strategies_selected_rs.append(strategy_rs)
    reward_rs = X[n, strategy_rs]
    total_reward_rs += reward_rs
    # Paso1 con Muestreo de Thompson
    strategy_ts = 0
    max_random = 0
    for i in range(0, d):
        random_beta = random.betavariate(number_of_rewards_1[i]+1, 
                                         number_of_rewards_0[i]+1)
        if random_beta > max_random: 
            max_random = random_beta
            strategy_ts = i
    reward_ts = X[n, strategy_ts]
    if reward_ts == 1:
        number_of_rewards_1[strategy_ts] += 1
    else:
        number_of_rewards_0[strategy_ts] += 1
    strategies_selected_ts.append(strategy_ts)
    total_reward_ts += reward_ts
    # Best Strategy
    for i in range(0, d):
        rewards_strategies[i] = rewards_strategies[i] + X[n, i]
    total_reward_bs = max(rewards_strategies)
    # Regret
    regret_rs.append(total_reward_bs - total_reward_rs)
    regret_ts.append(total_reward_bs - total_reward_ts)
    
# Calcular le retorno relativo y absoluto
absolute_return = (total_reward_ts - total_reward_rs)*100   # Suponemos que la empresa vende la suscribción a 100 dólares
relative_return = (total_reward_ts - total_reward_rs) / total_reward_rs * 100   # Multiplicamos por 100 para uqe nos quede en %
print("Rendimiento Absoluto: {:.0f} $".format(absolute_return))
print("Rendimiento Relativo: {:.0f} %".format(relative_return))
    
# Representación del histograma de selecciones
plt.hist(strategies_selected_ts)
plt.title("Histograma de Selecciones")
plt.xlabel("Estrategia")
plt.ylabel("Numero de veces que se ha seleccionado la estrategia")
plt.show()

# Y finalmente, por supuesto, trazamos el arrepentimiento sobre las rondas con este simple código 
# (no tenemos que especificar las coordenadas x en la función plt.plot() porque las rondas ya son 
# índices de 0 a N):
# Plotting the Regret Curve
plt.plot(regret_rs)
plt.title('Regret Curve')
plt.xlabel('Round')
plt.ylabel('Regret')
plt.show()