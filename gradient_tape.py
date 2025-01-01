import tensorflow as tf

input_var = tf.Variable(initial_value=3.)

# Iniciando escopo para coletar a "tape".
with tf.GradientTape() as tape:
    result = tf.square(input_var)

gradient = tape.gradient(result, input_var)

# No código acima, iniciamos uma tape que irá observar as operações aplicada na variável pré definida. 
# Dentro do escopo foi aplicado a operação de extrair o quadrado dela, sendo $x^2$, como sabemos, a derivada para essa operação é $f'(x) = 2.x$, 
# a mágica ocorre pois a tape registra as derivadas geradas através dos cálculos aplicados na variável, 
# dessa forma,quando apontamos para a tape e chamamos o método gradient, passando o result que representa a função que estamos derivando e o input,
# como resultado final temos a taxa de variação.

# Extraindo gradientes de segunda ordem.
time = tf.Variable(0.)

with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as inner_tape:
        position = 4.9 * time ** 2
    speed = inner_tape.gradient(position, time)

    print(speed)

acceleration = outer_tape.gradient(speed, time)

print(acceleration)