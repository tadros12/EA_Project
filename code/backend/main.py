import pandas as pd
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