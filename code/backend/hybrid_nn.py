import numpy as np
import tensorflow as tf
from tensorflow import keras
import gc # Keep garbage collection

# --- Constants ---
INPUT_SIZE = 784
HIDDEN_SIZE = 32
OUTPUT_SIZE = 10
W1_SIZE = INPUT_SIZE * HIDDEN_SIZE
B1_SIZE = HIDDEN_SIZE
W2_SIZE = HIDDEN_SIZE * OUTPUT_SIZE
B2_SIZE = OUTPUT_SIZE
D = W1_SIZE + B1_SIZE + W2_SIZE + B2_SIZE

# --- Data Loading ---
# (Keep the function to load data within the script context)
def load_and_prepare_data():
    print("Loading MNIST data...")
    try:
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        print("MNIST data loaded successfully.")
        print("Original x_train shape:", x_train.shape)
        x_train = tf.reshape(x_train, [-1, INPUT_SIZE])
        x_test = tf.reshape(x_test, [-1, INPUT_SIZE])
        print("Reshaped x_train shape:", x_train.shape)
        x_train = tf.cast(x_train, tf.float32) / 255.0
        x_test = tf.cast(x_test, tf.float32) / 255.0
        y_train_hot = keras.utils.to_categorical(y_train, num_classes=OUTPUT_SIZE)
        y_test_hot = keras.utils.to_categorical(y_test, num_classes=OUTPUT_SIZE)
        print("y_train_hot shape:", y_train_hot.shape)
        y_train_hot = tf.convert_to_tensor(y_train_hot, dtype=tf.float32)
        y_test_hot = tf.convert_to_tensor(y_test_hot, dtype=tf.float32)
        return x_train, y_train_hot, x_test, y_test_hot
    except Exception as e:
        print(f"Error loading or processing MNIST data: {e}")
        raise

# --- Activation Functions ---
def sigmoid(x):
    return tf.nn.sigmoid(x)

def softmax(x):
    return tf.nn.softmax(x)

# --- Weight Handling ---
def decode_weights(vector):
    # Using constants defined at the top
    idx = 0
    w1 = tf.reshape(vector[idx:idx + W1_SIZE], [INPUT_SIZE, HIDDEN_SIZE])
    idx += W1_SIZE
    b1 = vector[idx:idx + B1_SIZE]
    idx += B1_SIZE
    w2 = tf.reshape(vector[idx:idx + W2_SIZE], [HIDDEN_SIZE, OUTPUT_SIZE])
    idx += W2_SIZE
    b2 = vector[idx:idx + B2_SIZE]
    return w1, b1, w2, b2

# --- Forward Pass ---
def forward_pass(X, vector):
    w1, b1, w2, b2 = decode_weights(vector)
    hidden = sigmoid(tf.matmul(X, w1) + b1)
    output = softmax(tf.matmul(hidden, w2) + b2)
    return output

# --- Fitness Function (from Notebook - using batching) ---
def fitness(vector, X, y_true, batch_size=1000):
    # Ensure X and y_true are tensors for tf.gather
    if not isinstance(X, tf.Tensor):
        X = tf.convert_to_tensor(X, dtype=tf.float32)
    if not isinstance(y_true, tf.Tensor):
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)

    # Use tf.random.shuffle for indices to stay within TensorFlow graph if possible
    indices = tf.random.shuffle(tf.range(tf.shape(X)[0]))[:batch_size]
    X_batch = tf.gather(X, indices)
    y_batch = tf.gather(y_true, indices)

    y_pred = forward_pass(X_batch, vector)
    loss = tf.keras.losses.categorical_crossentropy(y_batch, y_pred)
    return tf.reduce_mean(loss)

# --- Population Initialization (from Notebook) ---
def initialize_population(population_size, D, input_dim=INPUT_SIZE, output_dim=OUTPUT_SIZE):
    xavier_scaling = tf.sqrt(6.0 / tf.cast(input_dim + output_dim, tf.float32))
    population = tf.random.uniform(
        shape=(population_size, D),
        minval=-xavier_scaling,
        maxval=xavier_scaling,
        dtype=tf.float32
    )
    return population

# --- DE Specific Functions (from Notebook) ---
def random_vectors(population, current_idx, NP): # Added NP param
    indices = tf.range(NP)
    mask = indices != current_idx
    valid_indices = tf.boolean_mask(indices, mask)
    # Ensure we don't try to select more unique indices than available
    num_available = tf.shape(valid_indices)[0]
    num_to_select = tf.minimum(3, num_available)
    selected_indices = tf.random.shuffle(valid_indices)[:num_to_select]
    # Handle case where NP is very small (less than 4) - repeat if necessary
    while tf.shape(selected_indices)[0] < 3:
         if num_available == 0: # Should not happen if NP > 1
              # Fallback: use current_idx if absolutely necessary (highly unlikely)
              print("Warning: DE random_vectors fallback due to extremely small population.")
              selected_indices = tf.concat([selected_indices, [current_idx]], axis=0)
         else:
              selected_indices = tf.concat([selected_indices, tf.random.shuffle(valid_indices)[:1]], axis=0)

    return [population[idx] for idx in selected_indices]

def mutate(x1, x2, x3, F, D, L=-1, H=1): # Changed from notebook default L/H
    v = x3 + F * (x1 - x2)
    # Clip values instead of replacing with random (closer to standard DE, might differ from notebook's explicit replacement)
    v = tf.clip_by_value(v, L, H)
    # --- Notebook's replacement logic (alternative) ---
    # out_of_bounds = tf.logical_or(v < L, v > H)
    # random_values = tf.random.uniform(shape=(D,), minval=L, maxval=H, dtype=tf.float32)
    # v = tf.where(out_of_bounds, random_values, v)
    # --- End alternative ---
    return v

def crossover(target_vector, mutant_vector, CR, D):
    r = tf.random.uniform(shape=(D,), dtype=tf.float32)
    mask = r < CR
    # Ensure at least one gene is from mutant (j_rand equivalent)
    j_rand = tf.random.uniform(shape=(), minval=0, maxval=D, dtype=tf.int32)
    mask = tf.tensor_scatter_nd_update(mask, [[j_rand]], [True]) # Set one position to True

    trial_vector = tf.where(mask, mutant_vector, target_vector)
    return trial_vector

# --- DE Evolution Loop (from Notebook - adapted to use tf.Variable population) ---
# (Keep all other functions like load_and_prepare_data, fitness, mutate, crossover, etc., the same)

# --- DE Evolution Loop (Optimized Memory) ---
def evolve(population_variable, GEN, NP, D, F, CR, x_train, y_train, fitness_func, L=-1, H=1):
    print(f"Starting DE (Optimized): Generations={GEN}, Population Size={NP}, F={F}, CR={CR}, L={L}, H={H}")
    # Initial fitness calculation
    # Convert fitness scores to a list of Python floats for easier management
    fitness_scores = [f.numpy() for f in [fitness_func(ind, x_train, y_train) for ind in population_variable]]
    best_idx = np.argmin(fitness_scores) # Use numpy argmin on the list
    # Use tf.identity to copy tensor value, not the variable itself
    best_solution = tf.identity(population_variable[best_idx])
    best_fitness = fitness_scores[best_idx] # Already a float

    fitness_history = [best_fitness] # Store initial best fitness

    for g in range(GEN):
        current_best_fitness_gen = best_fitness # Track best fitness found *within* this generation's loop

        for j in range(NP):
            # Select random vectors (pass NP)
            v1, v2, v3 = random_vectors(population_variable, j, NP)
            # Mutate (pass D, L, H)
            mutant_vector = mutate(v1, v2, v3, F, D, L, H)
            # Crossover (pass D)
            target_vector = population_variable[j] # Get current target variable slice
            trial_vector = crossover(target_vector, mutant_vector, CR, D)
            # Evaluate trial vector
            trial_fitness_tensor = fitness_func(trial_vector, x_train, y_train)
            trial_fitness = trial_fitness_tensor.numpy() # Convert to float
            target_fitness = fitness_scores[j] # Get score from the list

            # --- Direct Update Logic ---
            # Selection: If trial is better, update the variable and score list directly
            if trial_fitness < target_fitness:
                population_variable[j].assign(trial_vector) # Update the variable slice
                fitness_scores[j] = trial_fitness # Update the score in the list

                # Update overall best if trial is the new best for the whole run
                if trial_fitness < best_fitness:
                    best_solution = tf.identity(trial_vector) # Copy the new best tensor
                    best_fitness = trial_fitness # Update overall best fitness (float)

            # Keep track of the best fitness found just in this generation's main loop
            # This might differ slightly from notebook if elitism happened *during* the loop
            if trial_fitness < current_best_fitness_gen:
                 current_best_fitness_gen = trial_fitness


        # --- Elitism Step (After processing all individuals for the generation) ---
        worst_idx_in_gen = np.argmax(fitness_scores) # Find worst in the *updated* scores list
        # Replace worst if it's worse than the overall best solution found so far
        if fitness_scores[worst_idx_in_gen] > best_fitness:
            #print(f"  Gen {g+1}: Elitism replacing worst ({fitness_scores[worst_idx_in_gen]:.6f}) with best ({best_fitness:.6f})")
            population_variable[worst_idx_in_gen].assign(best_solution) # Assign best solution tensor
            fitness_scores[worst_idx_in_gen] = best_fitness # Update score list

        # --- Logging and Cleanup ---
        fitness_history.append(best_fitness) # Store best fitness for this gen
        gc.collect()

        

        if (g + 1) % 10 == 0:
             # Log the best fitness found *up to the end of this generation*
             print(f"DE Generation {g+1}/{GEN}: Best fitness (loss) = {best_fitness:.6f}")
             # Optional: Log memory usage if psutil is installed
             # try:
             #      import psutil, os
             #      process = psutil.Process(os.getpid())
             #      mem_mb = process.memory_info().rss / (1024 * 1024)
             #      print(f"  Memory Usage: {mem_mb:.2f} MB")
             # except ImportError:
             #      pass


    print(f"DE Finished. Final best fitness: {best_fitness:.6f}")
    # Return the best solution found and history. Also return the final population state.
    return best_solution, fitness_history, population_variable # Return variable
# --- GA Specific Functions (from Notebook) ---
def tournament_selection(population, fitness_scores, tournament_size):
    population_size = len(population)
    if population_size == 0: return None
    # Ensure tournament size is not larger than population size
    actual_tournament_size = min(tournament_size, population_size)
    if actual_tournament_size <= 0: return None # Cannot select from empty/zero size

    # Use numpy choice as TF doesn't have direct equivalent for choice without replacement easily
    candidates_indices = np.random.choice(population_size, actual_tournament_size, replace=False)
    # Get fitness scores for the candidates
    # Ensure fitness_scores is indexable (list or numpy array)
    try:
        candidate_fitness = [fitness_scores[i] for i in candidates_indices]
    except TypeError: # If fitness_scores is a Tensor maybe?
        candidate_fitness = tf.gather(fitness_scores, candidates_indices)


    winner_local_idx = tf.argmin(candidate_fitness)
    winner_global_idx = candidates_indices[winner_local_idx]
    return winner_global_idx

def genetic_crossover(parent1, parent2):
    mask = tf.cast(tf.random.uniform(shape=parent1.shape) < 0.5, tf.float32)
    child = parent1 * mask + parent2 * (1 - mask)
    return child

def genetic_mutation(individual, mutation_rate, mutation_strength=0.1): # Added strength param
    mutation_mask = tf.cast(tf.random.uniform(shape=individual.shape) < mutation_rate, tf.float32)
    mutation_addition = tf.random.normal(shape=individual.shape, mean=0.0, stddev=mutation_strength)
    mutated = individual + mutation_addition * mutation_mask
    # Optional: Clip results if bounds L, H are relevant for GA too
    # mutated = tf.clip_by_value(mutated, L, H)
    return mutated

# --- GA Evolution Loop (from Notebook - adapted) ---
def genetic_algorithm(initial_population_variable, x_train, y_train, fitness_func, generations=500, mutation_rate=0.05, tournament_size=3):
    population_size = tf.shape(initial_population_variable)[0]
    print(f"Starting GA: Generations={generations}, Population Size={population_size}, Mutation Rate={mutation_rate}") 
    # Work with tf.Variable list for potential in-place updates if needed, though typically GA rebuilds population
    population = [tf.Variable(ind) for ind in initial_population_variable] # Create list of variables
    #population_size = len(population)

    fitness_scores = [fitness_func(ind, x_train, y_train) for ind in population]
    best_idx = tf.argmin(fitness_scores)
    best_weights = tf.identity(population[best_idx]) # Copy best
    best_fitness = fitness_scores[best_idx]
    fitness_history = [best_fitness.numpy()] # Store initial best
    print(f"GA Initial best fitness (loss): {best_fitness:.6f}")

    for gen in range(generations):
        new_population_list = [] # Build the next generation here

        # Elitism: Keep the best individual from the current population
        elite_idx = tf.argmin(fitness_scores)
        new_population_list.append(tf.identity(population[elite_idx])) # Add copy of elite

        # Generate the rest of the population through selection, crossover, mutation
        for _ in range(population_size - 1): # Fill remaining slots
            parent1_idx = tournament_selection(population, fitness_scores, tournament_size)
            parent2_idx = tournament_selection(population, fitness_scores, tournament_size)
            # Ensure parents are different (if possible)
            tries = 0
            while parent1_idx == parent2_idx and population_size > 1 and tries < 10:
                parent2_idx = tournament_selection(population, fitness_scores, tournament_size)
                tries += 1
            # Handle cases where selection might fail (e.g., pop size 1)
            if parent1_idx is None or parent2_idx is None:
                 # If selection fails, just copy a random individual or the elite one
                 fallback_idx = elite_idx if elite_idx is not None else 0
                 new_population_list.append(tf.identity(population[fallback_idx]))
                 continue

            child = genetic_crossover(population[parent1_idx], population[parent2_idx])
            child = genetic_mutation(child, mutation_rate) # Pass mutation rate
            new_population_list.append(tf.Variable(child)) # Add as a new variable

        # Update population and fitness scores for the next generation
        population = new_population_list
        fitness_scores = [fitness_func(ind, x_train, y_train) for ind in population]
        current_best_idx = tf.argmin(fitness_scores)
        current_best_fitness = fitness_scores[current_best_idx]

        # Update overall best if current generation's best is better
        if current_best_fitness < best_fitness:
            best_weights = tf.identity(population[current_best_idx]) # Copy new best
            best_fitness = current_best_fitness

        fitness_history.append(best_fitness.numpy()) # Log best fitness for plot

        if (gen + 1) % 10 == 0:
            print(f"GA Generation {gen+1}/{generations}, Best fitness: {best_fitness:.6f}")
            gc.collect()

    print(f"GA Finished. Final best fitness: {best_fitness:.6f}")
    # Return the best weights found and history. Also return the final population state.
    return best_weights, fitness_history, population # Return list of variables


# --- Hybrid DE-GA Function (Wrapper based on Notebook) ---
def run_de_ga_hybrid(NP=50, GEN_DE=100, F=0.8, CR=0.7, ga_generations=50, mutation_rate=0.05, tournament_size=3, L=-1, H=1, seed: int | None = None):
    """
    Runs the hybrid DE-GA optimization based on notebook logic.
    """
    print("--- Starting Hybrid DE-GA (Notebook Logic) ---")
    if seed is not None:
        print(f"Setting random seed: {seed}")
        np.random.seed(seed) # Seed numpy for tournament selection
        tf.random.set_seed(seed) # Seed tensorflow

    # Load data
    x_train, y_train_hot, x_test, y_test_hot = load_and_prepare_data()

    # Initialize Population as tf.Variable for DE
    initial_population = initialize_population(population_size=NP, D=D)
    population_de_var = tf.Variable(initial_population)
    print("Initial population shape:", population_de_var.shape)

    # --- Run DE Phase ---
    # Pass the tf.Variable to evolve
    best_de_solution_tensor, de_history, final_de_population_var = evolve(
        population_de_var, GEN_DE, NP, D, F, CR, x_train, y_train_hot, fitness, L, H
    )

    print("--- DE phase complete, starting GA phase ---")
    # Pass the final DE population *variable* to GA
    # GA function expects a list/iterable of tensors/variables
    best_ga_solution_tensor, ga_history, final_ga_population_vars = genetic_algorithm(
        final_de_population_var, # Pass the variable itself
        x_train, y_train_hot, fitness,
        generations=ga_generations, mutation_rate=mutation_rate, tournament_size=tournament_size
    )

    # --- Determine Final Best Solution ---
    # Use the final fitness values from the history arrays
    de_final_fitness = de_history[-1] if de_history else float('inf')
    ga_final_fitness = ga_history[-1] if ga_history else float('inf')

    print(f"Final DE best fitness: {de_final_fitness:.6f}")
    print(f"Final GA best fitness: {ga_final_fitness:.6f}")

    if ga_final_fitness < de_final_fitness:
        print("GA achieved the best overall fitness.")
        best_solution_tensor = best_ga_solution_tensor
        final_loss = ga_final_fitness
    else:
        print("DE achieved the best overall fitness (or GA did not improve).")
        best_solution_tensor = best_de_solution_tensor
        final_loss = de_final_fitness

    # Prepare results dictionary
    # Convert histories and solution to Python lists/floats for JSON
    de_history_float = [float(f) for f in de_history]
    ga_history_float = [float(f) for f in ga_history]
    best_solution_flat = tf.reshape(best_solution_tensor, [-1]).numpy().tolist()

    results = {
        "best_solution_weights": best_solution_flat, # Keep this if needed, but it's large
        "de_history": de_history_float,
        "ga_history": ga_history_float,
        "final_loss": float(final_loss),
        "de_generations_count": len(de_history_float), # Number of recorded steps
        "ga_generations_count": len(ga_history_float)  # Number of recorded steps
    }
    print("--- Hybrid DE-GA Finished ---")
    # Clean up memory explicitly
    del x_train, y_train_hot, x_test, y_test_hot
    del population_de_var, final_de_population_var, final_ga_population_vars
    gc.collect()
    return results


# --- Evaluation Function (Optional - Keep if needed for testing) ---
def evaluate_model(best_solution_vector, x_test, y_test):
    print("Evaluating final model...")
    if isinstance(best_solution_vector, list):
         best_solution_vector = tf.convert_to_tensor(best_solution_vector, dtype=tf.float32)

    # Ensure data is loaded if not already available globally (it won't be in Flask context)
    # This evaluation might need data loading if called independently
    # _, _, x_test_eval, y_test_eval = load_and_prepare_data() # Example

    probabilities = forward_pass(x_test, best_solution_vector) # Pass appropriate x_test
    predictions = tf.argmax(probabilities, axis=1)
    true_labels = tf.argmax(y_test, axis=1) # Pass appropriate y_test
    correct = tf.equal(predictions, true_labels)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_test, probabilities))
    print(f"Test accuracy: {accuracy.numpy() * 100:.2f}%")
    print(f"Test loss: {loss.numpy():.4f}")
    return accuracy.numpy(), predictions.numpy(), probabilities.numpy()


# --- Example __main__ block (for direct testing of this script) ---
# if __name__ == "__main__":
#      print("Testing hybrid_nn.py script directly...")
#      # Use smaller parameters for quick testing
#      test_results = run_de_ga_hybrid(
#          NP=20, GEN_DE=10, F=0.8, CR=0.7,
#          ga_generations=5, mutation_rate=0.1, tournament_size=3,
#          L=-1, H=1, seed=42
#      )
#      print("\nTest Run Results:")
#      print(f"  DE History Length: {len(test_results['de_history'])}")
#      print(f"  GA History Length: {len(test_results['ga_history'])}")
#      print(f"  Final Loss: {test_results['final_loss']:.6f}")

#      # Optional: Evaluate the result
    _, _, x_test_main, y_test_main = load_and_prepare_data()
    evaluate_model(test_results['best_solution_weights'], x_test_main, y_test_main)