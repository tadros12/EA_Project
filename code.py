import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import tensorflow as tf
from tensorflow import keras

INPUT_SIZE = 784
HIDDEN_SIZE = 128
OUTPUT_SIZE = 10

from keras.datasets import mnist


(x_train,y_train),(x_test, y_test) = mnist.load_data()
print("images train before normalize and processing",x_train.shape)

x_train = tf.reshape(x_train, [-1, 28*28])
x_test = tf.reshape(x_test, [-1, 28*28])

print("images train after reshaping",x_train.shape)
# x_train[1]

# handling pixels to be only black and white from gray scale
x_train = tf.cast(x_train, tf.float32) / 255.0
x_test = tf.cast(x_test, tf.float32) / 255.0


y_train_hot = keras.utils.to_categorical(y_train, num_classes=10)
y_test_hot = keras.utils.to_categorical(y_test, num_classes=10)
print(y_train_hot.shape)
print(y_test_hot.shape)

y_train_hot = tf.convert_to_tensor(y_train_hot, dtype=tf.float32)
y_test_hot = tf.convert_to_tensor(y_test_hot, dtype=tf.float32)



def sigmoid(x):
    return tf.nn.sigmoid(x)

def softmax(x):
    return tf.nn.softmax(x)

w1_size = INPUT_SIZE * HIDDEN_SIZE
b1_size = HIDDEN_SIZE
w2_size = HIDDEN_SIZE * OUTPUT_SIZE
b2_size = OUTPUT_SIZE
D = w1_size + b1_size + w2_size + b2_size


def decode_weights(vector):
    
    w1_size = INPUT_SIZE * HIDDEN_SIZE
    b1_size = HIDDEN_SIZE
    w2_size = HIDDEN_SIZE * OUTPUT_SIZE
    b2_size = OUTPUT_SIZE
    
    idx = 0
    w1 = tf.reshape(vector[idx:idx + w1_size], [INPUT_SIZE, HIDDEN_SIZE])
    idx += w1_size
    
    b1 = vector[idx:idx + b1_size]
    idx += b1_size
    
    w2 = tf.reshape(vector[idx:idx + w2_size], [HIDDEN_SIZE, OUTPUT_SIZE])
    idx += w2_size
    
    b2 = vector[idx:idx + b2_size]
    
    return w1, b1, w2, b2


def forward_pass(X, vector):
    w1, b1, w2, b2 = decode_weights(vector)
    
    hidden = sigmoid(tf.matmul(X, w1) + b1)
    output = softmax(tf.matmul(hidden, w2) + b2)
    
    return output

def fitness(vector, X, y_true, batch_size=1000):
    
    indices = np.random.choice(X.shape[0], batch_size, replace=False)
    X_batch = tf.gather(X, indices)
    y_batch = tf.gather(y_true, indices)
    
    y_pred = forward_pass(X_batch, vector)
    loss = tf.keras.losses.categorical_crossentropy(y_batch, y_pred)
    return tf.reduce_mean(loss)
    
# def fitness(vector, X, y_true):
#     y_pred = forward_pass(X, vector)
    
#     loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    
#     return tf.reduce_mean(loss)




total_weights = INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE * OUTPUT_SIZE + OUTPUT_SIZE
vector_np = np.random.randn(total_weights) * 0.01
vector_tf = tf.convert_to_tensor(vector_np, dtype=tf.float32)

loss = fitness(vector_tf, x_train, y_train_hot)
print(f"Initial loss: {loss.numpy()}")




def initialize_population(population_size, D, input_dim=784, output_dim=10):
    
    xavier_scaling = tf.sqrt(6.0 / tf.cast(input_dim + output_dim, tf.float32))
    population = tf.random.uniform(
        shape=(population_size, D),
        minval=-xavier_scaling,
        maxval=xavier_scaling,
        dtype=tf.float32
    )
    
    return population



    

def random_vectors(population, current_idx):
    indices = tf.range(NP)
    mask = indices != current_idx
    valid_indices = tf.boolean_mask(indices, mask)
    selected_indices = tf.random.shuffle(valid_indices)[:3]
    return [population[idx] for idx in selected_indices]

    
def mutate(x1, x2, x3, F, D, L=-1, H=1):

v = x3 + F * (x1 - x2)

out_of_bounds = tf.logical_or(v < L, v > H) # returns tensor fo booleans where true is erorr

random_values = L + tf.random.uniform(shape=(D,), dtype=tf.float32) * (H - L)

v = tf.where(out_of_bounds, random_values, v) # checks where true and takes value from the random value if false takes from v

return v

def crossover(target_vector, mutant_vector, CR, D):

    # instead of loop and check we make the whole vecotor once and check with mask
    r = tf.random.uniform(shape=(D,), dtype=tf.float32)
    
    
    mask = r < CR  # mask is tensor flow boolean vecotr
    
    trial_vector = tf.where(mask, mutant_vector, target_vector)
    
    return trial_vector



def evolve(population, GEN, NP, D, F, CR, x_train, y_train, fitness_func, L=-0.5, H=0.5):

    # best solution
    fitness_scores = [fitness_func(ind, x_train, y_train) for ind in population]
    best_idx = tf.argmin(fitness_scores)
    best_solution = tf.identity(population[best_idx])  # tf.identity is copy to set it as the best solution 
    best_fitness = fitness_scores[best_idx]
    
    fitness_history = []

    for g in range(GEN):
                
        new_population = [tf.identity(ind) for ind in population] #copy of the population for this generation

        new_fitness_scores = list(fitness_scores) 
        
        for j in range(NP):
            
            v1, v2, v3 = random_vectors(population, j)
            mutant_vector = mutate(v1, v2, v3, F, D, L, H)
            
            target_vector = tf.identity(population[j])         
            trial_vector = crossover(target_vector, mutant_vector, CR, D)
            
            trial_fitness = fitness_func(trial_vector, x_train, y_train)
            target_fitness = new_fitness_scores[j]
            
            if trial_fitness < target_fitness:
                new_population[j] = trial_vector
                new_fitness_scores[j] = trial_fitness
                    
                
                if trial_fitness < best_fitness:
                    best_solution = tf.identity(trial_vector)
                    best_fitness = trial_fitness

        
        
        # each genration swap worst for our best soultion   and send it to next population 
        worst_idx = tf.argmax(new_fitness_scores).numpy()  #get highest fitness / worst indfivsual 
        
        if new_fitness_scores[worst_idx] > best_fitness:
            new_population[worst_idx] = tf.identity(best_solution)
            new_fitness_scores[worst_idx] = best_fitness
            print(f"Generation {g+1}/{GEN}: Best fitness (loss) = {best_fitness:.6f}")

            
            
        # upate population with new poulation for next genration 
        for i in range(NP):
            population[i].assign(new_population[i])
        fitness_scores = new_fitness_scores

        if g % 10 == 0:  # for memory
            import gc
            gc.collect()


        fitness_history.append(best_fitness)

        
        #print(f"Generation {g+1}/{GEN}: Best fitness (loss) = {best_fitness:.6f}")
    
    return best_solution ,fitness_history,population


def tournament_selection(population, fitness_scores, tournament_size):

    population_size = len(population)
    
    candidates = np.random.choice(population_size, tournament_size, replace=False)
    
    candidate_fitness = [fitness_scores[i] for i in candidates]
    
    winner_idx = candidates[tf.argmin(candidate_fitness)]
    return winner_idx


def genetic_crossover(parent1, parent2):

    mask = tf.cast(tf.random.uniform(shape=parent1.shape) < 0.5, tf.float32)  # to summarize this just a vecotr of zeros and ones and the .5 control the bian to each parent 
    
    child = parent1 * mask + parent2 * (1 - mask)
    
    return child


def genetic_mutation(individual, mutation_rate):  #non uniform using gaussian distrubtion


    mutation_mask = tf.cast(tf.random.uniform(shape=individual.shape) < mutation_rate, tf.float32) # again some zeros and ones
    
    mutation = tf.random.normal(shape=individual.shape, mean=0.0, stddev=0.1)
    
    # Apply mutation
    mutated = individual + mutation * mutation_mask  #  𝑥'  = 𝑥  + N(0,std)   the mutation mask to make sure only %mtation rate genes change 
    
    return mutated
def genetic_algorithm(de_population, x_train, y_train, fitness_func,generations=500, mutation_rate=0.05, tournament_size=3):


    population = [tf.Variable(ind) for ind in de_population]
    population_size = len(population)
    
    fitness_scores = [fitness_func(ind, x_train, y_train) for ind in population]
    
    best_idx = tf.argmin(fitness_scores)
    best_weights = tf.identity(population[best_idx])
    best_fitness = fitness_scores[best_idx]

    
    fitness_history = [best_fitness.numpy()] # for plot later
    
    print(f"Starting GA with initial fitness (loss): {best_fitness:.6f}")
    
    for gen in range(generations):

        new_population = []
        
        for indivusal in range(population_size):

            parent1_idx = tournament_selection(population, fitness_scores, tournament_size)
            parent2_idx = tournament_selection(population, fitness_scores, tournament_size)
            
            # ensure parents are different
            while parent2_idx == parent1_idx:
                parent2_idx = tournament_selection(population, fitness_scores, tournament_size)
            



            #  crossover
            child = genetic_crossover(population[parent1_idx], population[parent2_idx])
            



            # mutate
            child = genetic_mutation(child, mutation_rate)
            
            new_population.append(child)
        
        population = new_population  # update old population with new poulation 
        
        fitness_scores = [fitness_func(ind, x_train, y_train) for ind in population]
        
        current_best_idx = tf.argmin(fitness_scores)
        current_best_fitness = fitness_scores[current_best_idx]
        
        if current_best_fitness < best_fitness:
            best_weights = tf.identity(population[current_best_idx])
            best_fitness = current_best_fitness


        fitness_history.append(best_fitness.numpy())
        

        print(f"Generation {gen+1}/{generations}, Best fitness: {best_fitness:.6f}")
    
    print(f"GA best fitness (loss): {best_fitness:.6f}")
    
    return best_weights, fitness_history,population

def de_ga_hybrid(population, GEN_DE, NP, D, F, CR, x_train, y_train, fitness_func,ga_generations=500, L=-0.5, H=0.5):

    best_de_solution, de_history, final_de_population = evolve(population, GEN_DE, NP, D, F, CR, x_train, y_train, fitness_func, L, H)

    
    print("DE complete starting GA ")
    

    ga_solution, ga_history,gen_poulation = genetic_algorithm(final_de_population,x_train,y_train,fitness_func,generations=ga_generations,mutation_rate=0.05,tournament_size=3)
    
    # final solutions
    de_fitness = fitness_func(best_de_solution, x_train, y_train)
    ga_fitness = fitness_func(ga_solution, x_train, y_train)


    
    print(f"DE best fitness: {de_fitness:.6f}")
    print(f"GA best fitness: {ga_fitness:.6f}")
    
    # Return the best overall solution
    if ga_fitness < de_fitness:
        print("GA had the best overall fitness")
        best_solution = ga_solution
    else:
        print("DE solution was better")
        best_solution = best_de_solution
    
    return best_solution, de_history, ga_history, final_de_population

NP = 100
GEN = 1000
seed = 42
np.random.seed(seed)

population = initialize_population(population_size=NP, D=D)
print("population shape is: ", population.shape)
population = tf.Variable(population)

res, de_history, ga_history, pop = de_ga_hybrid(population, GEN, NP, D, 0.8,  0.7,  x_train, y_train_hot, fitness, ga_generations=800, L=-1, H=1)

def evaluate_model(best_solution, x_test, y_test):

    # Get predictions
    probabilities = forward_pass(x_test, best_solution)
    predictions = tf.argmax(probabilities, axis=1)
    true_labels = tf.argmax(y_test, axis=1)
    
    # Calculate accuracy
    correct = tf.equal(predictions, true_labels)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
    # Calculate loss
    loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_test, probabilities))
    
    print(f"Test accuracy: {accuracy.numpy() * 100:.2f}%")
    print(f"Test loss: {loss.numpy():.4f}")
    
    return accuracy.numpy(), predictions.numpy(), probabilities.numpy()

import plotly.graph_objects as go

# Convert TensorFlow tensors to NumPy arrays for JSON serialization
de_history_np = [float(loss.numpy()) if hasattr(loss, 'numpy') else float(loss) for loss in de_history]
ga_history_np = [float(loss) if hasattr(loss, 'numpy') else float(loss) for loss in ga_history]

# Create generation numbers for x-axis
de_generations = np.arange(len(de_history_np))
ga_generations = np.arange(len(de_history_np), len(de_history_np) + len(ga_history_np))

# Find min and max loss values
min_loss = min(min(de_history_np), min(ga_history_np))
max_loss = max(max(de_history_np), max(ga_history_np))

# Create figure
fig = go.Figure()

# Add DE history trace (blue)
fig.add_trace(
    go.Scatter(
        x=de_generations,
        y=de_history_np,
        mode='lines',
        name='DE Optimization',
        line=dict(color='blue')
    )
)

# Add GA history trace (red)
fig.add_trace(
    go.Scatter(
        x=ga_generations,
        y=ga_history_np,
        mode='lines',
        name='GA Refinement',
        line=dict(color='red')
    )
)

# Add vertical line at transition
fig.add_shape(
    type="line",
    x0=len(de_history_np)-1,
    y0=min_loss,
    x1=len(de_history_np)-1,
    y1=max_loss,
    line=dict(
        color="gray",
        width=1,
        dash="dash",
    )
)

# Add transition annotation
fig.add_annotation(
    x=len(de_history_np)-1,
    y=float(de_history_np[-1]),
    text="DE → GA Transition",
    showarrow=True,
    arrowhead=1,
    ax=0,
    ay=-40
)

# Update layout
fig.update_layout(
    title='Hybrid DE-GA Optimization Progress',
    xaxis_title='Generation',
    yaxis_title='Loss',
    height=600,
    width=900,
    legend=dict(
        x=0.01,
        y=0.99,
        bgcolor='rgba(255, 255, 255, 0.5)'
    ),
    hovermode='x unified'
)

# Customize hover information
fig.update_traces(
    hovertemplate='Generation: %{x}<br>Loss: %{y:.6f}'
)

# Display the figure
fig.show()

# Save as interactive HTML
fig.write_html("hybrid_optimization_progress.html")