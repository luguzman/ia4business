# Inteligencia Artifical aplicada a Negocios y Empresas - Caso Práctico 2
# Creación de la red Q profunda

# Importar las librerías
import numpy as np

# IMPLEMENTAR EL ALGORITMO DE DEEP Q-LEARNING CON REPETICIÓN DE EXPERIENCIA

class DQN(object):
    
    # INTRODUCCIÓN E INICIALIZACIÓN DE LOS PARÁMETROS Y VARIABLES DEL DQN
    def __init__(self, max_memory = 100, discount_factor = 0.9):
        self.memory = list()
        self.max_memory = max_memory
        self.discount_factor = discount_factor
        
    # CREACIÓN DE UN MÉTODO QUE CONSTRUYA LA MEMORIA DE LA REPETICIÓN DE EXPERIENCIA
    def remember(self, transition, game_over):
        self.memory.append([transition, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]
        
    # CREACIÓN DE UN MÉTODO QUE CONSTRUYA DOS BLOQUES DE ENTRADAS Y TARGETS EXTRATENDO TRANSICIONES
    def get_batch(self, model, batch_size = 10): # El batch_size puede ser otro
        len_memory = len(self.memory)
        num_inputs = self.memory[0][0][0].shape[1]
        num_outputs = model.output_shape[-1] # output_shape: es una función de tf.keras [-1] es para denotar que es de la ultima capa de salida
        inputs = np.zeros((min(batch_size, len_memory), num_inputs))  # nos quedamos con el minimo entre batch_size, len_memory ya que en la primera iteración la memoria no tendra 10 valores       
        targets = np.zeros((min(batch_size, len_memory), num_outputs))  # sucede de la misma manera
        for i, idx in enumerate(np.random.randint(0, len_memory, size=min(len_memory, batch_size))):    # extraeremos min(len_memory, batch_size) valores de la memoria 
            current_state, action, reward, next_state = self.memory[idx][0]
            game_over = self.memory[idx][1]
            inputs[i] = current_state
            targets[i] = model.predict(current_state)[0]
            Q_sa = np.max(model.predict(next_state)[0])
            if game_over:
                targets[i, action] = reward # hacemos que no sume nada sino que solo se quede en el valor super negativo que se obtendra de la energia que se tendría que usar
            else:
                targets[i, action] = reward + self.discount_factor*Q_sa # + el factor de descuento * el maximo de los Q(s,a)
        return inputs, targets
            
            
            
            