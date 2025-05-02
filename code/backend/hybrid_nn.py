import numpy as np
# ... other imports ...
import tensorflow as tf
from tensorflow import keras
import gc

# --- (Keep Constants, Data Loading, Activations, Weight Handling, Forward Pass, Fitness) ---
INPUT_SIZE = 784
HIDDEN_SIZE = 128
OUTPUT_SIZE = 10
W1_SIZE = INPUT_SIZE * HIDDEN_SIZE
B1_SIZE = HIDDEN_SIZE
W2_SIZE = HIDDEN_SIZE * OUTPUT_SIZE
B2_SIZE = OUTPUT_SIZE
D = W1_SIZE + B1_SIZE + W2_SIZE + B2_SIZE

def load_and_prepare_data():
    # ... (same as before) ...
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

def sigmoid(x):
    return tf.nn.sigmoid(x)

def softmax(x):
    return tf.nn.softmax(x)

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

def forward_pass(X, vector):
    w1, b1, w2, b2 = decode_weights(vector)
    hidden = sigmoid(tf.matmul(X, w1) + b1)
    output = softmax(tf.matmul(hidden, w2) + b2)
    return output

def fitness(vector, X, y_true, batch_size=1000):
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


# --- (Keep DE Specific Functions: initialize_population, random_vectors, de_mutate, de_crossover) ---
def initialize_population(population_size, D, input_dim=INPUT_SIZE, output_dim=OUTPUT_SIZE):
    limit = tf.sqrt(6.0 / tf.cast(input_dim + output_dim, tf.float32))
    population = tf.random.uniform(shape=(population_size, D), minval=-limit, maxval=limit, dtype=tf.float32)
    return population

def random_vectors(population, current_idx, NP):
    indices = tf.range(NP)
    mask = indices != current_idx
    valid_indices = tf.boolean_mask(indices, mask)
    num_to_select = min(3, NP - 1)
    if num_to_select < 3:
         if NP <= 3:
              print("Warning: Population size <= 3, DE mutation might behave unexpectedly.")
              selected_indices = tf.random.shuffle(valid_indices)[:num_to_select]
              while len(selected_indices) < 3:
                   selected_indices = tf.concat([selected_indices, tf.random.shuffle(valid_indices)[:1]], axis=0)
         else:
              selected_indices = tf.random.shuffle(valid_indices)[:3]
    else:
         selected_indices = tf.random.shuffle(valid_indices)[:3]
    return [population[idx] for idx in selected_indices]

def de_mutate(x1, x2, x3, F, D, L=-1, H=1):
    v = x3 + F * (x1 - x2)
    v = tf.clip_by_value(v, L, H)
    return v

def de_crossover(target_vector, mutant_vector, CR, D):
    r = tf.random.uniform(shape=(D,), dtype=tf.float32)
    mask = r < CR
    j_rand = tf.random.uniform(shape=(), minval=0, maxval=D, dtype=tf.int32)
    mask = tf.tensor_scatter_nd_update(mask, [[j_rand]], [True])
    trial_vector = tf.where(mask, mutant_vector, target_vector)
    return trial_vector


# --- (Keep DE Evolution Loop: evolve) ---
def evolve(population, GEN, NP, D, F, CR, x_train, y_train, fitness_func, L=-1, H=1):
    print(f"Starting DE: Generations={GEN}, Population Size={NP}, F={F}, CR={CR}")
    fitness_scores = [fitness_func(ind, x_train, y_train) for ind in population]
    best_idx = tf.argmin(fitness_scores)
    best_solution = tf.identity(population[best_idx])
    best_fitness = fitness_scores[best_idx]
    fitness_history = [best_fitness.numpy()]

    for g in range(GEN):
        new_population_list = []
        new_fitness_scores = list(fitness_scores)

        for j in range(NP):
            target_vector = population[j]
            target_fitness = fitness_scores[j]
            v1, v2, v3 = random_vectors(population, j, NP)
            mutant_vector = de_mutate(v1, v2, v3, F, D, L, H)
            trial_vector = de_crossover(target_vector, mutant_vector, CR, D)
            trial_fitness = fitness_func(trial_vector, x_train, y_train)

            if trial_fitness < target_fitness:
                new_population_list.append(trial_vector)
                new_fitness_scores[j] = trial_fitness
                if trial_fitness < best_fitness:
                    best_solution = tf.identity(trial_vector)
                    best_fitness = trial_fitness
            else:
                new_population_list.append(target_vector)

        population = tf.stack(new_population_list)
        fitness_scores = new_fitness_scores
        fitness_history.append(best_fitness.numpy())

        if (g + 1) % 10 == 0:
             print(f"DE Generation {g+1}/{GEN}: Best fitness (loss) = {best_fitness:.6f}")
             gc.collect()

    print(f"DE Finished. Final best fitness: {best_fitness:.6f}")
    return best_solution, fitness_history, population


# --- (Keep GA Specific Functions: tournament_selection, genetic_crossover, genetic_mutation) ---
def tournament_selection(population, fitness_scores, tournament_size):
    population_size = len(population)
    if population_size == 0: return None
    actual_tournament_size = min(tournament_size, population_size)
    if actual_tournament_size <= 0: return None
    candidates_indices = np.random.choice(population_size, actual_tournament_size, replace=False)
    candidate_fitness = [fitness_scores[i] for i in candidates_indices]
    winner_local_idx = tf.argmin(candidate_fitness)
    winner_global_idx = candidates_indices[winner_local_idx]
    return winner_global_idx

def genetic_crossover(parent1, parent2):
    mask = tf.cast(tf.random.uniform(shape=parent1.shape) < 0.5, tf.float32)
    child = parent1 * mask + parent2 * (1 - mask)
    return child

def genetic_mutation(individual, mutation_rate, mutation_strength=0.1):
    mutation_mask = tf.cast(tf.random.uniform(shape=individual.shape) < mutation_rate, tf.float32)
    mutation_addition = tf.random.normal(shape=individual.shape, mean=0.0, stddev=mutation_strength)
    mutated = individual + mutation_addition * mutation_mask
    return mutated


# --- (Keep GA Evolution Loop: genetic_algorithm) ---
def genetic_algorithm(initial_population, x_train, y_train, fitness_func, generations=500, mutation_rate=0.05, tournament_size=3):
    print(f"Starting GA: Generations={generations}, Population Size={len(initial_population)}, Mutation Rate={mutation_rate}")
    population = tf.stack(initial_population)
    population_size = population.shape[0]
    fitness_scores = [fitness_func(ind, x_train, y_train) for ind in population]
    best_idx = tf.argmin(fitness_scores)
    best_weights = tf.identity(population[best_idx])
    best_fitness = fitness_scores[best_idx]
    fitness_history = [best_fitness.numpy()]
    print(f"GA Initial best fitness: {best_fitness:.6f}")

    for gen in range(generations):
        new_population_list = []
        elite_idx = tf.argmin(fitness_scores)
        new_population_list.append(population[elite_idx])

        for _ in range(population_size - 1):
            parent1_idx = tournament_selection(population, fitness_scores, tournament_size)
            parent2_idx = tournament_selection(population, fitness_scores, tournament_size)
            tries = 0
            while parent2_idx == parent1_idx and population_size > 1 and tries < 10:
                parent2_idx = tournament_selection(population, fitness_scores, tournament_size)
                tries += 1
            if parent1_idx is None or parent2_idx is None: continue
            child = genetic_crossover(population[parent1_idx], population[parent2_idx])
            child = genetic_mutation(child, mutation_rate)
            new_population_list.append(child)

        if not new_population_list:
             print(f"Warning: GA generation {gen+1} resulted in empty new population.")
             break
        population = tf.stack(new_population_list)
        fitness_scores = [fitness_func(ind, x_train, y_train) for ind in population]
        current_best_idx = tf.argmin(fitness_scores)
        current_best_fitness = fitness_scores[current_best_idx]

        if current_best_fitness < best_fitness:
            best_weights = tf.identity(population[current_best_idx])
            best_fitness = current_best_fitness
        fitness_history.append(best_fitness.numpy())

        if (gen + 1) % 10 == 0:
            print(f"GA Generation {gen+1}/{generations}, Best fitness: {best_fitness:.6f}")
            gc.collect()

    print(f"GA Finished. Final best fitness: {best_fitness:.6f}")
    return best_weights, fitness_history, population


# --- Updated Hybrid DE-GA Function ---
def run_de_ga_hybrid(NP=50, GEN_DE=100, F=0.8, CR=0.7, ga_generations=50, mutation_rate=0.05, tournament_size=3, L=-1, H=1, seed: int | None = None): # Add seed parameter
    """
    Runs the hybrid DE-GA optimization for the MNIST dataset.import pandas as pd
from flask import Flask, request, jsonify
# Remove old VRP function imports if no longer needed
# from functions import create_random_set , run_ga , initialize_population, run_differential_evolution
from flask_cors import CORS

# Import the function from your new NN script
from hybrid_nn import run_de_ga_hybrid # Make sure hybrid_nn.py is in the same folder

app = Flask(__name__)
# Make sure the origin matches your React app's address (likely localhost:5173 for Vite)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

# --- Old VRP Endpoints (Keep them if you still need them) ---
@app.route("/generate-dataset")
def generate_dataset():
    # This endpoint likely needs to be removed or adapted if you don't need VRP data
    # For now, we keep it but it's not used by the NN part.
    try:
        # Example: Generate dummy data or remove if unused
        n_customers = request.args.get("customers", default=20, type=int)
        # Replace create_random_set with appropriate logic or remove
        # response = create_random_set(n_customers)
        response = {"message": f"Dummy dataset endpoint called with {n_customers} customers. Adapt or remove."}
        return jsonify(response), 200
    except ValueError:
        return jsonify({"error": "Invalid 'customers' parameter."}), 400
    except NameError:
         return jsonify({"error": "'create_random_set' not defined or imported."}), 500


@app.route("/generate-vrp-ga", methods=["POST"])
def generate_vrp_ga():
    # Keep if needed, otherwise remove
    return jsonify({"message": "VRP GA endpoint called. Adapt or remove."}), 200

@app.route("/generate-vrp-de", methods=["POST"])
def generate_vrp_de():
     # Keep if needed, otherwise remove
    return jsonify({"message": "VRP DE endpoint called. Adapt or remove."}), 200

# --- New Endpoint for Hybrid Neural Network ---
@app.route("/run-hybrid-nn", methods=["POST"])
def run_hybrid_nn_endpoint():
    print("Received request for /run-hybrid-nn")
    data = request.json or {}

    # Extract parameters from the request, providing defaults
    try:
        NP = int(data.get("population_size", 50)) # Match React form name 'population_size'
        GEN_DE = int(data.get("de_generations", 100)) # Suggest clear names in React form
        F = float(data.get("f_factor", 0.8))       # Suggest 'f_factor' in React form
        CR = float(data.get("cr_rate", 0.7))        # Suggest 'cr_rate' in React form
        ga_generations = int(data.get("ga_generations", 50))
        mutation_rate = float(data.get("mutation_rate", 0.05))
        tournament_size = int(data.get("tournament_size", 3))
        # Optional bounds (can be omitted if defaults in hybrid_nn.py are fine)
        L = float(data.get("lower_bound", -1.0))
        H = float(data.get("upper_bound", 1.0))

        print(f"Parameters received: NP={NP}, GEN_DE={GEN_DE}, F={F}, CR={CR}, GA_GEN={ga_generations}, MutRate={mutation_rate}, TournSize={tournament_size}")

    except (ValueError, TypeError) as e:
        print(f"Error parsing parameters: {e}")
        return jsonify({"error": f"Invalid parameter type: {str(e)}"}), 400

    try:
        # Call the main function from hybrid_nn.py
        results = run_de_ga_hybrid(
            NP=NP,
            GEN_DE=GEN_DE,
            F=F,
            CR=CR,
            ga_generations=ga_generations,
            mutation_rate=mutation_rate,
            tournament_size=tournament_size,
            L=L,
            H=H
        )
        print("Hybrid DE-GA run completed successfully.")
        # The results dictionary should already be JSON serializable
        return jsonify(results), 200

    except Exception as e:
        # Catch potential errors during the NN training run
        print(f"Error during hybrid NN execution: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback to Flask console
        return jsonify({"error": f"An error occurred during execution: {str(e)}"}), 500


if __name__ == "__main__":
    # Set debug=False for production, True for development
    # Use host='0.0.0.0' to make it accessible on your network if needed
    app.run(debug=True, host='127.0.0.1', port=5000) # Keep port 5000 as React expects

    Args:
        # ... (other args) ...
        seed (int | None): Optional random seed for reproducibility. Defaults to None.

    Returns:
        dict: A dictionary containing the optimization results...
    """
    print("--- Starting Hybrid DE-GA ---")

    # --- Set Random Seed if provided ---
    if seed is not None:
        print(f"Setting random seed: {seed}")
        # import numpy as np # Uncomment if you use np.random directly
        # np.random.seed(seed)
        tf.random.set_seed(seed)
    # --- End Seed Setting ---

    # Load data
    x_train, y_train_hot, x_test, y_test_hot = load_and_prepare_data()

    # Initialize Population
    initial_population = initialize_population(population_size=NP, D=D)
    print("Initial population shape:", initial_population.shape)
    population_de = tf.Variable(initial_population)

    # --- Run DE Phase ---
    best_de_solution, de_history, final_de_population = evolve(
        population_de, GEN_DE, NP, D, F, CR, x_train, y_train_hot, fitness, L, H
    )

    print("--- DE phase complete, starting GA phase ---")
    population_ga_initial = tf.stack(final_de_population)

    # --- Run GA Phase ---
    best_ga_solution, ga_history, final_ga_population = genetic_algorithm(
        population_ga_initial, x_train, y_train_hot, fitness,
        generations=ga_generations, mutation_rate=mutation_rate, tournament_size=tournament_size
    )

    # --- Determine Final Best Solution ---
    # ... (rest of the function remains the same: determining best, preparing results) ...
    de_final_fitness = de_history[-1]
    ga_final_fitness = ga_history[-1]
    print(f"Final DE best fitness: {de_final_fitness:.6f}")
    print(f"Final GA best fitness: {ga_final_fitness:.6f}")
    if ga_final_fitness < de_final_fitness:
        print("GA achieved the best overall fitness.")
        best_solution = best_ga_solution
        final_loss = ga_final_fitness
    else:
        print("DE achieved the best overall fitness (or GA did not improve).")
        best_solution = best_de_solution
        final_loss = de_final_fitness

    de_history_float = [float(f) for f in de_history]
    ga_history_float = [float(f) for f in ga_history]
    best_solution_flat = tf.reshape(best_solution, [-1]).numpy().tolist()
    results = {
        "best_solution_weights": best_solution_flat,
        "de_history": de_history_float,
        "ga_history": ga_history_float,
        "final_loss": float(final_loss),
        "de_generations_count": len(de_history_float),
        "ga_generations_count": len(ga_history_float)
    }
    print("--- Hybrid DE-GA Finished ---")
    return results


# --- (Keep Evaluation Function: evaluate_model) ---
def evaluate_model(best_solution_vector, x_test, y_test):
    # ... (same as before) ...
    print("Evaluating final model...")
    if isinstance(best_solution_vector, list):
         best_solution_vector = tf.convert_to_tensor(best_solution_vector, dtype=tf.float32)
    probabilities = forward_pass(x_test, best_solution_vector)
    predictions = tf.argmax(probabilities, axis=1)
    true_labels = tf.argmax(y_test, axis=1)
    correct = tf.equal(predictions, true_labels)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_test, probabilities))
    print(f"Test accuracy: {accuracy.numpy() * 100:.2f}%")
    print(f"Test loss: {loss.numpy():.4f}")
    return accuracy.numpy(), predictions.numpy(), probabilities.numpy()


# --- (Keep Example __main__ block if desired for testing) ---
# if __name__ == "__main__":
#     print("Testing hybrid_nn.py script...")
#     results = run_de_ga_hybrid(NP=20, GEN_DE=10, ga_generations=5, seed=42) # Test with seed
#     print("\nResults:")
#     print(f"  DE History Length: {len(results['de_history'])}")
#     print(f"  GA History Length: {len(results['ga_history'])}")
#     print(f"  Final Loss: {results['final_loss']:.6f}")
