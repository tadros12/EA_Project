import numpy as np
# import matplotlib.pyplot as plt # Not needed for backend logic
# from sklearn.datasets import fetch_openml # Not needed if using keras.datasets.mnist
# from sklearn.preprocessing import OneHotEncoder, StandardScaler # Not needed if using keras.datasets.mnist
# from sklearn.model_selection import train_test_split # Not needed if using keras.datasets.mnist
# from sklearn.metrics import log_loss # Using categorical_crossentropy
import tensorflow as tf
from tensorflow import keras
import gc # For garbage collection

# --- Constants ---
INPUT_SIZE = 784
HIDDEN_SIZE = 128
OUTPUT_SIZE = 10
W1_SIZE = INPUT_SIZE * HIDDEN_SIZE
B1_SIZE = HIDDEN_SIZE
W2_SIZE = HIDDEN_SIZE * OUTPUT_SIZE
B2_SIZE = OUTPUT_SIZE
D = W1_SIZE + B1_SIZE + W2_SIZE + B2_SIZE # Total number of weights/biases

# --- Load and Prepare MNIST Data ---
# It's better to load this once globally if the app runs continuously,
# but loading it inside the function is simpler for now.
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
        # In a real app, you might want to handle this more gracefully
        # Maybe download manually or point to a local path if fetch fails.
        raise # Re-raise the exception to stop execution if data fails

# --- Activation Functions ---
def sigmoid(x):
    return tf.nn.sigmoid(x)

def softmax(x):
    return tf.nn.softmax(x)

# --- Weight Handling ---
def decode_weights(vector):
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

# --- Fitness Function ---
def fitness(vector, X, y_true, batch_size=1000):
    # Using batches for fitness calculation to potentially speed up and reduce memory
    if X.shape[0] > batch_size:
         indices = tf.random.shuffle(tf.range(X.shape[0]))[:batch_size]
         X_batch = tf.gather(X, indices)
         y_batch = tf.gather(y_true, indices)
    else:
         X_batch = X
         y_batch = y_true

    y_pred = forward_pass(X_batch, vector)
    loss = tf.keras.losses.categorical_crossentropy(y_batch, y_pred)
    return tf.reduce_mean(loss)


# --- DE Specific Functions ---
def initialize_population(population_size, D, input_dim=INPUT_SIZE, output_dim=OUTPUT_SIZE):
    # Using Xavier/Glorot initialization for weights might be better
    limit = tf.sqrt(6.0 / tf.cast(input_dim + output_dim, tf.float32))
    population = tf.random.uniform(
        shape=(population_size, D),
        minval=-limit,
        maxval=limit,
        dtype=tf.float32
    )
    return population

def random_vectors(population, current_idx, NP):
    indices = tf.range(NP)
    mask = indices != current_idx
    valid_indices = tf.boolean_mask(indices, mask)
    # Ensure we have enough unique indices if NP is small
    num_to_select = min(3, NP - 1)
    if num_to_select < 3:
         # Handle edge case where NP is too small (e.g., NP=3)
         # This might involve allowing replacement or adjusting the strategy
         # For simplicity, let's assume NP >= 4
         if NP <= 3:
              print("Warning: Population size <= 3, DE mutation might behave unexpectedly.")
              selected_indices = tf.random.shuffle(valid_indices)[:num_to_select]
              # Pad with existing indices if needed (not ideal DE)
              while len(selected_indices) < 3:
                   selected_indices = tf.concat([selected_indices, tf.random.shuffle(valid_indices)[:1]], axis=0)

         else:
              selected_indices = tf.random.shuffle(valid_indices)[:3]

    else:
         selected_indices = tf.random.shuffle(valid_indices)[:3]

    return [population[idx] for idx in selected_indices]

def de_mutate(x1, x2, x3, F, D, L=-1, H=1):
    v = x3 + F * (x1 - x2)
    # Clip to bounds instead of random replacement for stability
    v = tf.clip_by_value(v, L, H)
    # out_of_bounds = tf.logical_or(v < L, v > H)
    # random_values = tf.random.uniform(shape=(D,), minval=L, maxval=H, dtype=tf.float32)
    # v = tf.where(out_of_bounds, random_values, v)
    return v

def de_crossover(target_vector, mutant_vector, CR, D):
    r = tf.random.uniform(shape=(D,), dtype=tf.float32)
    mask = r < CR
    # Ensure at least one gene is from the mutant (j_rand)
    j_rand = tf.random.uniform(shape=(), minval=0, maxval=D, dtype=tf.int32)
    mask = tf.tensor_scatter_nd_update(mask, [[j_rand]], [True]) # Set one element to True
    trial_vector = tf.where(mask, mutant_vector, target_vector)
    return trial_vector

# --- DE Evolution Loop ---
def evolve(population, GEN, NP, D, F, CR, x_train, y_train, fitness_func, L=-1, H=1):
    print(f"Starting DE: Generations={GEN}, Population Size={NP}, F={F}, CR={CR}")
    fitness_scores = [fitness_func(ind, x_train, y_train) for ind in population]
    best_idx = tf.argmin(fitness_scores)
    best_solution = tf.identity(population[best_idx])
    best_fitness = fitness_scores[best_idx]
    fitness_history = [best_fitness.numpy()] # Store initial best

    for g in range(GEN):
        new_population_list = [] # Use list for flexibility before converting to Tensor
        new_fitness_scores = list(fitness_scores) # Copy scores

        for j in range(NP):
            target_vector = population[j]
            target_fitness = fitness_scores[j]

            v1, v2, v3 = random_vectors(population, j, NP)
            mutant_vector = de_mutate(v1, v2, v3, F, D, L, H)
            trial_vector = de_crossover(target_vector, mutant_vector, CR, D)

            trial_fitness = fitness_func(trial_vector, x_train, y_train)

            if trial_fitness < target_fitness:
                new_population_list.append(trial_vector)
                new_fitness_scores[j] = trial_fitness # Update score directly in the list

                if trial_fitness < best_fitness:
                    best_solution = tf.identity(trial_vector)
                    best_fitness = trial_fitness
            else:
                new_population_list.append(target_vector)
                # Score remains new_fitness_scores[j]

        # Update population for the next generation
        population = tf.stack(new_population_list) # Convert list back to Tensor
        fitness_scores = new_fitness_scores # Use the updated scores

        fitness_history.append(best_fitness.numpy()) # Log best fitness per generation

        if (g + 1) % 10 == 0: # Print progress periodically
             print(f"DE Generation {g+1}/{GEN}: Best fitness (loss) = {best_fitness:.6f}")
             gc.collect() # Garbage collect periodically

    print(f"DE Finished. Final best fitness: {best_fitness:.6f}")
    return best_solution, fitness_history, population


# --- GA Specific Functions ---
def tournament_selection(population, fitness_scores, tournament_size):
    population_size = len(population)
    if population_size == 0: return None # Handle empty population
    # Ensure tournament size isn't larger than population size
    actual_tournament_size = min(tournament_size, population_size)
    if actual_tournament_size <= 0: return None # Handle invalid size

    candidates_indices = np.random.choice(population_size, actual_tournament_size, replace=False)
    candidate_fitness = [fitness_scores[i] for i in candidates_indices]
    winner_local_idx = tf.argmin(candidate_fitness)
    winner_global_idx = candidates_indices[winner_local_idx]
    return winner_global_idx

def genetic_crossover(parent1, parent2):
    # Uniform crossover
    mask = tf.cast(tf.random.uniform(shape=parent1.shape) < 0.5, tf.float32)
    child = parent1 * mask + parent2 * (1 - mask)
    return child

def genetic_mutation(individual, mutation_rate, mutation_strength=0.1):
    mutation_mask = tf.cast(tf.random.uniform(shape=individual.shape) < mutation_rate, tf.float32)
    # Gaussian mutation: add small random values
    mutation_addition = tf.random.normal(shape=individual.shape, mean=0.0, stddev=mutation_strength)
    mutated = individual + mutation_addition * mutation_mask
    # Optional: Clip back to bounds if necessary, depends on how bounds are handled overall
    # mutated = tf.clip_by_value(mutated, L, H) # If L and H are defined/passed
    return mutated

# --- GA Evolution Loop ---
def genetic_algorithm(initial_population, x_train, y_train, fitness_func, generations=500, mutation_rate=0.05, tournament_size=3):
    print(f"Starting GA: Generations={generations}, Population Size={len(initial_population)}, Mutation Rate={mutation_rate}")
    # Convert initial population to tf.Variable if needed, or work with Tensors
    population = tf.stack(initial_population) # Ensure it's a Tensor
    population_size = population.shape[0]

    fitness_scores = [fitness_func(ind, x_train, y_train) for ind in population]
    best_idx = tf.argmin(fitness_scores)
    best_weights = tf.identity(population[best_idx])
    best_fitness = fitness_scores[best_idx]
    fitness_history = [best_fitness.numpy()]

    print(f"GA Initial best fitness: {best_fitness:.6f}")

    for gen in range(generations):
        new_population_list = []

        # Elitism: Keep the best individual directly
        elite_idx = tf.argmin(fitness_scores)
        new_population_list.append(population[elite_idx])

        # Generate the rest of the population
        for _ in range(population_size - 1): # -1 for the elite individual
            parent1_idx = tournament_selection(population, fitness_scores, tournament_size)
            parent2_idx = tournament_selection(population, fitness_scores, tournament_size)
            # Ensure parents are different if possible
            tries = 0
            while parent2_idx == parent1_idx and population_size > 1 and tries < 10:
                parent2_idx = tournament_selection(population, fitness_scores, tournament_size)
                tries += 1

            if parent1_idx is None or parent2_idx is None: continue # Skip if selection failed

            child = genetic_crossover(population[parent1_idx], population[parent2_idx])
            child = genetic_mutation(child, mutation_rate)
            new_population_list.append(child)

        if not new_population_list: # Handle empty list if errors occurred
             print(f"Warning: GA generation {gen+1} resulted in empty new population.")
             break # Stop if something went wrong

        population = tf.stack(new_population_list) # Update population

        # Recalculate fitness for the new population
        fitness_scores = [fitness_func(ind, x_train, y_train) for ind in population]
        current_best_idx = tf.argmin(fitness_scores)
        current_best_fitness = fitness_scores[current_best_idx]

        if current_best_fitness < best_fitness:
            best_weights = tf.identity(population[current_best_idx])
            best_fitness = current_best_fitness

        fitness_history.append(best_fitness.numpy()) # Log best fitness per generation

        if (gen + 1) % 10 == 0:
            print(f"GA Generation {gen+1}/{generations}, Best fitness: {best_fitness:.6f}")
            gc.collect()

    print(f"GA Finished. Final best fitness: {best_fitness:.6f}")
    return best_weights, fitness_history, population


# --- Hybrid DE-GA Function ---
def run_de_ga_hybrid(NP=50, GEN_DE=100, F=0.8, CR=0.7, ga_generations=50, mutation_rate=0.05, tournament_size=3, L=-1, H=1):
    """
    Runs the hybrid DE-GA optimization for the MNIST dataset.

    Args:
        NP (int): Population size.
        GEN_DE (int): Number of generations for Differential Evolution.
        F (float): DE mutation factor.
        CR (float): DE crossover rate.
        ga_generations (int): Number of generations for Genetic Algorithm.
        mutation_rate (float): GA mutation rate.
        tournament_size (int): GA tournament selection size.
        L (float): Lower bound for weights during DE mutation.
        H (float): Upper bound for weights during DE mutation.

    Returns:
        dict: A dictionary containing the optimization results:
              {
                  "best_solution_weights": list, # Flattened best weights
                  "de_history": list,          # Loss history during DE
                  "ga_history": list,          # Loss history during GA
                  "final_loss": float,         # Final best loss achieved
                  "de_generations_count": int,
                  "ga_generations_count": int
              }
    """
    print("--- Starting Hybrid DE-GA ---")
    # Set random seeds for reproducibility if needed
    # seed = 42
    # np.random.seed(seed)
    # tf.random.set_seed(seed)

    # Load data
    x_train, y_train_hot, x_test, y_test_hot = load_and_prepare_data()

    # Initialize Population
    initial_population = initialize_population(population_size=NP, D=D)
    print("Initial population shape:", initial_population.shape)
    # Make population variable for potential in-place updates if needed by functions
    population_de = tf.Variable(initial_population)

    # --- Run DE Phase ---
    best_de_solution, de_history, final_de_population = evolve(
        population_de, GEN_DE, NP, D, F, CR, x_train, y_train_hot, fitness, L, H
    )

    print("--- DE phase complete, starting GA phase ---")
    # Ensure the population passed to GA is in the right format (list of Tensors or stack)
    population_ga_initial = tf.stack(final_de_population) # Use the final population from DE

    # --- Run GA Phase ---
    best_ga_solution, ga_history, final_ga_population = genetic_algorithm(
        population_ga_initial, x_train, y_train_hot, fitness,
        generations=ga_generations, mutation_rate=mutation_rate, tournament_size=tournament_size
    )

    # --- Determine Final Best Solution ---
    de_final_fitness = de_history[-1] # Last recorded best fitness from DE phase
    ga_final_fitness = ga_history[-1] # Last recorded best fitness from GA phase

    print(f"Final DE best fitness: {de_final_fitness:.6f}")
    print(f"Final GA best fitness: {ga_final_fitness:.6f}")

    if ga_final_fitness < de_final_fitness:
        print("GA achieved the best overall fitness.")
        best_solution = best_ga_solution
        final_loss = ga_final_fitness
    else:
        print("DE achieved the best overall fitness (or GA did not improve).")
        best_solution = best_de_solution # Use the best from DE if GA didn't improve
        final_loss = de_final_fitness

    # --- Prepare results for returning ---
    # Convert TF tensors in history to floats for JSON
    de_history_float = [float(f) for f in de_history]
    ga_history_float = [float(f) for f in ga_history]

    # Flatten best solution weights for easier handling if needed, or return as is
    best_solution_flat = tf.reshape(best_solution, [-1]).numpy().tolist()

    results = {
        "best_solution_weights": best_solution_flat, # Sending weights might be large
        "de_history": de_history_float,
        "ga_history": ga_history_float,
        "final_loss": float(final_loss),
        "de_generations_count": len(de_history_float), # Should match GEN_DE + 1 initial
        "ga_generations_count": len(ga_history_float) # Should match ga_generations + 1 initial
    }

    # Optional: Evaluate final model on test set
    # accuracy, _, _ = evaluate_model(best_solution, x_test, y_test_hot)
    # results["test_accuracy"] = float(accuracy)
    # print(f"Test Accuracy of final model: {accuracy*100:.2f}%")

    print("--- Hybrid DE-GA Finished ---")
    return results


# --- Evaluation Function (Optional, can be called after getting results) ---
def evaluate_model(best_solution_vector, x_test, y_test):
    print("Evaluating final model...")
    # Ensure the vector is a TF tensor
    if isinstance(best_solution_vector, list):
         best_solution_vector = tf.convert_to_tensor(best_solution_vector, dtype=tf.float32)

    # Get predictions
    probabilities = forward_pass(x_test, best_solution_vector)
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


# --- Example of how to call (for testing hybrid_nn.py directly) ---
# if __name__ == "__main__":
#     print("Testing hybrid_nn.py script...")
#     results = run_de_ga_hybrid(NP=20, GEN_DE=10, ga_generations=5) # Small values for quick test
#     print("\nResults:")
#     print(f"  DE History Length: {len(results['de_history'])}")
#     print(f"  GA History Length: {len(results['ga_history'])}")
#     print(f"  Final Loss: {results['final_loss']:.6f}")
#     # print(f"  Test Accuracy: {results.get('test_accuracy', 'N/A')}") # If evaluation is included

#     # Example of evaluating after getting results
#     # x_tr, y_tr, x_te, y_te = load_and_prepare_data()
#     # evaluate_model(results['best_solution_weights'], x_te, y_te)
