import tensorflow as tf
import matplotlib.pyplot as plt

# Criando tensor rank-2 com todos valores 1. 
rank2 = tf.ones(shape=(2,1))

# Criando tensor rank-3 com todos valores 0. 
rank3 = tf.zeros(shape=(3,3,9))

# Criando tensor rank-3 com todos valores aleatórios. 
# mean = Média onde os valores irão se concentrar.
# stddev = Desvio padrão dos valores
random_rank_3 = tf.random.normal(shape=(9,2,2), mean=2, stddev=2)

# Criando tensor rank-3 com todos valores aleatórios uniforme. 
uniform_rank_3 = tf.random.uniform(shape=(9,2,2), minval=0., maxval=1.)

# Como tensors são constantes, temos o .Variable que nos permite armazenar estado de tensors
tensor_variable = tf.Variable(initial_value=uniform_rank_3)

# Assinalando valor a um .Variable
# tensor_variable.assign(random_rank_3)

# Acessando indexes específicos de um tensor
# print(tensor_variable[0, 1, 1])

# Operações com tensors, no exemplo abaixo pegando o produto
product = tf.matmul(uniform_rank_3, random_rank_3)

print(product)




