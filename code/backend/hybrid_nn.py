import numpy as np
import tensorflow as tf
from tensorflow import keras
import gc

INPUT_SIZE = 784
HIDDEN_SIZE = 128
OUTPUT_SIZE = 10
W1_SIZE = INPUT_SIZE * HIDDEN_SIZE
B1_SIZE = HIDDEN_SIZE
W2_SIZE = HIDDEN_SIZE * OUTPUT_SIZE
B2_SIZE = OUTPUT_SIZE
D = W1_SIZE + B1_SIZE + W2_SIZE + B2_SIZE


def load_and_prepare_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = tf.reshape(x_train, [-1, INPUT_SIZE])
    x_test = tf.reshape(x_test, [-1, INPUT_SIZE])
    x_train = tf.cast(x_train, tf.float32) / 255.0
    x_test = tf.cast(x_test, tf.float32) / 255.0
    y_train_hot = keras.utils.to_categorical(y_train, num_classes=OUTPUT_SIZE)
    y_test_hot = keras.utils.to_categorical(y_test, num_classes=OUTPUT_SIZE)
    y_train_hot = tf.convert_to_tensor(y_train_hot, dtype=tf.float32)
    y_test_hot = tf.convert_to_tensor(y_test_hot, dtype=tf.float32)
    return x_train, y_train_hot, x_test, y_test_hot


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
    if not isinstance(X, tf.Tensor):
        X = tf.convert_to_tensor(X, dtype=tf.float32)
    if not isinstance(y_true, tf.Tensor):
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    indices = tf.random.shuffle(tf.range(tf.shape(X)[0]))[:batch_size]
    X_batch = tf.gather(X, indices)
    y_batch = tf.gather(y_true, indices)
    y_pred = forward_pass(X_batch, vector)
    loss = tf.keras.losses.categorical_crossentropy(y_batch, y_pred)
    return tf.reduce_mean(loss)


def initialize_population(population_size, D, input_dim=INPUT_SIZE, output_dim=OUTPUT_SIZE):
    xavier_scaling = tf.sqrt(6.0 / tf.cast(input_dim + output_dim, tf.float32))
    population = tf.random.uniform(
        shape=(population_size, D),
        minval=-xavier_scaling,
        maxval=xavier_scaling,
        dtype=tf.float32
    )
    return population


def random_vectors(population, current_idx, NP):
    indices = tf.range(NP)
    mask = indices != current_idx
    valid_indices = tf.boolean_mask(indices, mask)
    num_available = tf.shape(valid_indices)[0]
    num_to_select = tf.minimum(3, num_available)
    selected_indices = tf.random.shuffle(valid_indices)[:num_to_select]
    while tf.shape(selected_indices)[0] < 3:
         if num_available == 0:
              selected_indices = tf.concat([selected_indices, [current_idx]], axis=0)
         else:
              selected_indices = tf.concat([selected_indices, tf.random.shuffle(valid_indices)[:1]], axis=0)
    return [population[idx] for idx in selected_indices]


def mutate(x1, x2, x3, F, D, L=-1, H=1):
    v = x3 + F * (x1 - x2)
    v = tf.clip_by_value(v, L, H)
    return v


def crossover(target_vector, mutant_vector, CR, D):
    r = tf.random.uniform(shape=(D,), dtype=tf.float32)
    mask = r < CR
    j_rand = tf.random.uniform(shape=(), minval=0, maxval=D, dtype=tf.int32)
    mask = tf.tensor_scatter_nd_update(mask, [[j_rand]], [True])
    trial_vector = tf.where(mask, mutant_vector, target_vector)
    return trial_vector


def extinction(population_variable, extinction_percentage, NP, D, fitness_func, x_train, y_train):
    if extinction_percentage <= 0.0:
        return population_variable

    if extinction_percentage >= 1.0:
         num_survivors = 1
    else:
        num_survivors = tf.cast(tf.round(tf.cast(NP, tf.float32) * (1.0 - extinction_percentage)), tf.int32)
        num_survivors = tf.maximum(num_survivors, 1)

    num_new = NP - num_survivors
    print(f"  Extinction Event: Keeping {num_survivors} best, replacing {num_new} individuals.")

    # On-the-fly fitness without storing persistent lists
    fitness_values = []
    for ind in population_variable:
        fitness_values.append(float(fitness_func(ind, x_train, y_train).numpy()))
    fitness_values = np.array(fitness_values)
    survivor_indices = tf.argsort(fitness_values)[:num_survivors]

    survivors = tf.gather(population_variable, survivor_indices)
    new_individuals = initialize_population(population_size=num_new, D=D)
    new_population_tensor = tf.concat([survivors, new_individuals], axis=0)
    new_population_tensor = tf.random.shuffle(new_population_tensor)
    del survivors, new_individuals, fitness_values
    gc.collect()
    return new_population_tensor


def evolve(population_variable, GEN, NP, D, F, CR, x_train, y_train, fitness_func, L=-1, H=1,
           extinction_percentage=0.0, extinction_generation=0):
    print(f"Starting DE: Generations={GEN}, Population Size={NP}, F={F}, CR={CR}, L={L}, H={H}")
    if extinction_generation > 0 and extinction_percentage > 0:
         print(f"  Extinction enabled: {extinction_percentage*100:.1f}% every {extinction_generation} generations.")
    else:
         print("  Extinction disabled.")

    # fitness_scores are used only for current generation, not kept for long
    fitness_scores = [float(fitness_func(ind, x_train, y_train).numpy()) for ind in population_variable]
    best_idx = np.argmin(fitness_scores)
    best_solution = tf.identity(population_variable[best_idx])
    best_fitness = fitness_scores[best_idx]
    fitness_history = [best_fitness]

    for g in range(GEN):
        for j in range(NP):
            v1, v2, v3 = random_vectors(population_variable, j, NP)
            mutant_vector = mutate(v1, v2, v3, F, D, L, H)
            target_vector = population_variable[j]
            trial_vector = crossover(target_vector, mutant_vector, CR, D)
            trial_fitness = float(fitness_func(trial_vector, x_train, y_train).numpy())
            target_fitness = fitness_scores[j]

            if trial_fitness < target_fitness:
                population_variable[j].assign(trial_vector)
                fitness_scores[j] = trial_fitness
                if trial_fitness < best_fitness:
                    best_solution = tf.identity(trial_vector)
                    best_fitness = trial_fitness

        worst_idx_in_gen = np.argmax(fitness_scores)
        if fitness_scores[worst_idx_in_gen] > best_fitness:
            population_variable[worst_idx_in_gen].assign(best_solution)
            fitness_scores[worst_idx_in_gen] = best_fitness

        if extinction_generation > 0 and extinction_percentage > 0 and (g + 1) % extinction_generation == 0:
            print(f"\n--- Triggering Extinction Event at Generation {g+1} ---")
            new_population_tensor = extinction(
                population_variable, extinction_percentage, NP, D, fitness_func, x_train, y_train
            )
            population_variable.assign(new_population_tensor)
            fitness_scores = [float(fitness_func(ind, x_train, y_train).numpy()) for ind in population_variable]
            best_idx = np.argmin(fitness_scores)
            best_fitness = fitness_scores[best_idx]
            best_solution = tf.identity(population_variable[best_idx])
            print(f"  New best fitness after extinction: {best_fitness:.6f}")
            gc.collect()

        fitness_history.append(best_fitness)
        gc.collect()

        if (g + 1) % 10 == 0 or g == GEN -1 :
             print(f"DE Generation {g+1}/{GEN}: Best fitness (loss) = {best_fitness:.6f}")

    print(f"DE Finished. Final best fitness: {best_fitness:.6f}")
    return best_solution, fitness_history, population_variable


def tournament_selection(population, fitness_scores, tournament_size):
    population_size = len(population)
    if population_size == 0: return None
    actual_tournament_size = min(tournament_size, population_size)
    if actual_tournament_size <= 0: return None

    candidates_indices = np.random.choice(population_size, actual_tournament_size, replace=False)
    try:
        candidate_fitness = [fitness_scores[i] for i in candidates_indices]
    except Exception as e:
         print(f"Error getting candidate fitness: {e}")
         return np.random.choice(candidates_indices)

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


def genetic_algorithm(initial_population_variable, x_train, y_train, fitness_func, generations=500, mutation_rate=0.05, tournament_size=3):
    population_size = tf.shape(initial_population_variable)[0]
    print(f"Starting GA: Generations={generations}, Population Size={population_size}, Mutation Rate={mutation_rate}")
    population = [tf.Variable(ind) for ind in initial_population_variable]

    fitness_scores = [float(fitness_func(ind, x_train, y_train).numpy()) for ind in population]
    best_idx = np.argmin(fitness_scores)
    best_weights = tf.identity(population[best_idx])
    best_fitness = fitness_scores[best_idx]
    fitness_history = [best_fitness]
    print(f"GA Initial best fitness (loss): {best_fitness:.6f}")

    for gen in range(generations):
        new_population_list = []
        elite_idx = np.argmin(fitness_scores)
        new_population_list.append(tf.identity(population[elite_idx]))

        current_pop_size = len(population)

        for _ in range(current_pop_size - 1):
            parent1_idx = tournament_selection(population, fitness_scores, tournament_size)
            parent2_idx = tournament_selection(population, fitness_scores, tournament_size)
            tries = 0
            while parent1_idx == parent2_idx and current_pop_size > 1 and tries < 10:
                parent2_idx = tournament_selection(population, fitness_scores, tournament_size)
                tries += 1
            if parent1_idx is None or parent2_idx is None:
                 fallback_idx = elite_idx if elite_idx is not None else 0
                 new_population_list.append(tf.identity(population[fallback_idx]))
                 continue

            child = genetic_crossover(population[parent1_idx], population[parent2_idx])
            child = genetic_mutation(child, mutation_rate)
            new_population_list.append(tf.Variable(child))

        population = new_population_list
        fitness_scores = [float(fitness_func(ind, x_train, y_train).numpy()) for ind in population]
        current_best_idx = np.argmin(fitness_scores)
        current_best_fitness = fitness_scores[current_best_idx]

        if current_best_fitness < best_fitness:
            best_weights = tf.identity(population[current_best_idx])
            best_fitness = current_best_fitness

        fitness_history.append(best_fitness)

        if (gen + 1) % 10 == 0 or gen == generations -1:
            print(f"GA Generation {gen+1}/{generations}, Best fitness: {best_fitness:.6f}")
            gc.collect()

    print(f"GA Finished. Final best fitness: {best_fitness:.6f}")
    return best_weights, fitness_history, population


def evaluate_model(best_solution_vector, x_test, y_test):
    if isinstance(best_solution_vector, list):
         best_solution_vector = tf.convert_to_tensor(best_solution_vector, dtype=tf.float32)
    elif not isinstance(best_solution_vector, tf.Tensor):
         best_solution_vector = tf.convert_to_tensor(best_solution_vector, dtype=tf.float32)

    probabilities = forward_pass(x_test, best_solution_vector)
    predictions = tf.argmax(probabilities, axis=1)
    true_labels = tf.argmax(y_test, axis=1)
    correct = tf.equal(predictions, true_labels)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_test, probabilities))
    print(f"Evaluation - Test accuracy: {accuracy.numpy() * 100:.2f}%")
    print(f"Evaluation - Test loss: {loss.numpy():.4f}")
    return accuracy.numpy()


def run_de_ga_hybrid(NP=50, GEN_DE=100, F=0.8, CR=0.7, ga_generations=50, mutation_rate=0.05, tournament_size=3, L=-1, H=1, seed: int | None = None, extinction_percentage=0.0, extinction_generation=0):
    print("--- Starting Hybrid DE-GA ---")
    if seed is not None:
        print(f"Setting random seed: {seed}")
        np.random.seed(seed)
        tf.random.set_seed(seed)

    x_train, y_train_hot, x_test, y_test_hot = load_and_prepare_data()
    initial_population = initialize_population(population_size=NP, D=D)
    population_de_var = tf.Variable(initial_population)
    print("Initial population shape:", population_de_var.shape)

    best_de_solution_tensor, de_history, final_de_population_var = evolve(
        population_de_var, GEN_DE, NP, D, F, CR, x_train, y_train_hot, fitness, L, H,
        extinction_percentage=extinction_percentage,
        extinction_generation=extinction_generation
    )

    print("--- DE phase complete, starting GA phase ---")
    best_ga_solution_tensor, ga_history, final_ga_population_vars = genetic_algorithm(
        final_de_population_var,
        x_train, y_train_hot, fitness,
        generations=ga_generations, mutation_rate=mutation_rate, tournament_size=tournament_size
    )

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

    print("--- Evaluating Final Model ---")
    final_accuracy = evaluate_model(best_solution_tensor, x_test, y_test_hot)

    de_history_float = [float(f) for f in de_history]
    ga_history_float = [float(f) for f in ga_history]

    results = {
        "algorithm_run": "DE-GA",
        "de_history": de_history_float,
        "ga_history": ga_history_float,
        "final_loss": float(final_loss),
        "final_accuracy": float(final_accuracy),
        "de_generations_count": len(de_history_float),
        "ga_generations_count": len(ga_history_float)
    }
    print("--- Hybrid DE-GA Finished ---")
    del x_train, y_train_hot, x_test, y_test_hot, population_de_var, final_de_population_var, final_ga_population_vars
    gc.collect()
    return results


def run_ga_de_hybrid(NP=50, GEN_DE=100, F=0.8, CR=0.7, ga_generations=50, mutation_rate=0.05, tournament_size=3, L=-1, H=1, seed: int | None = None, extinction_percentage=0.0, extinction_generation=0):
    print("--- Starting Hybrid GA-DE ---")
    if seed is not None:
        print(f"Setting random seed: {seed}")
        np.random.seed(seed)
        tf.random.set_seed(seed)

    x_train, y_train_hot, x_test, y_test_hot = load_and_prepare_data()
    initial_population = initialize_population(population_size=NP, D=D)
    population_ga_var_list = [tf.Variable(ind) for ind in initial_population]
    print("Initial population shape:", initial_population.shape)

    print("--- Starting GA phase ---")
    best_ga_solution_tensor, ga_history, final_ga_population_vars_list = genetic_algorithm(
        population_ga_var_list,
        x_train, y_train_hot, fitness,
        generations=ga_generations, mutation_rate=mutation_rate, tournament_size=tournament_size
    )

    print("--- GA phase complete, starting DE phase ---")
    try:
         population_de_var = tf.Variable(tf.stack(final_ga_population_vars_list))
    except Exception as e:
         print(f"Error stacking GA population for DE: {e}. Using initial population for DE.")
         population_de_var = tf.Variable(initialize_population(population_size=NP, D=D))

    best_de_solution_tensor, de_history, final_de_population_var = evolve(
        population_de_var, GEN_DE, NP, D, F, CR, x_train, y_train_hot, fitness, L, H,
        extinction_percentage=extinction_percentage,
        extinction_generation=extinction_generation
    )

    ga_final_fitness = ga_history[-1] if ga_history else float('inf')
    de_final_fitness = de_history[-1] if de_history else float('inf')
    print(f"Final GA best fitness: {ga_final_fitness:.6f}")
    print(f"Final DE best fitness: {de_final_fitness:.6f}")

    if de_final_fitness < ga_final_fitness:
        print("DE achieved the best overall fitness.")
        best_solution_tensor = best_de_solution_tensor
        final_loss = de_final_fitness
    else:
        print("GA achieved the best overall fitness (or DE did not improve).")
        best_solution_tensor = best_ga_solution_tensor
        final_loss = ga_final_fitness

    print("--- Evaluating Final Model ---")
    final_accuracy = evaluate_model(best_solution_tensor, x_test, y_test_hot)

    ga_history_float = [float(f) for f in ga_history]
    de_history_float = [float(f) for f in de_history]

    results = {
        "algorithm_run": "GA-DE",
        "ga_history": ga_history_float,
        "de_history": de_history_float,
        "final_loss": float(final_loss),
        "final_accuracy": float(final_accuracy),
        "ga_generations_count": len(ga_history_float),
        "de_generations_count": len(de_history_float)
    }
    print("--- Hybrid GA-DE Finished ---")
    del x_train, y_train_hot, x_test, y_test_hot, population_ga_var_list, final_ga_population_vars_list, population_de_var, final_de_population_var
    gc.collect()
    return results