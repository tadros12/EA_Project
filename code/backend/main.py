import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from hybrid_nn import run_de_ga_hybrid
import traceback # Keep traceback for error logging

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}}) # Adjust origin if needed

# --- (Keep other VRP endpoints if necessary) ---

# --- Updated Endpoint for Hybrid Neural Network ---
@app.route("/run-hybrid-nn", methods=["POST"])
def run_hybrid_nn_endpoint():
    print("Received request for /run-hybrid-nn")
    data = request.json or {}
    seed_value = None # Initialize seed as None

    try:
        NP = int(data.get("population_size", 50))
        GEN_DE = int(data.get("de_generations", 100))
        F = float(data.get("f_factor", 0.8))
        CR = float(data.get("cr_rate", 0.7))
        ga_generations = int(data.get("ga_generations", 50))
        mutation_rate = float(data.get("mutation_rate", 0.05))
        tournament_size = int(data.get("tournament_size", 3))
        L = float(data.get("lower_bound", -1.0))
        H = float(data.get("upper_bound", 1.0))

        # --- Get and parse the seed ---
        raw_seed = data.get("seed")
        if raw_seed is not None and str(raw_seed).strip(): # Check if not None and not empty string
            try:
                 seed_value = int(raw_seed)
                 print(f"Received seed value: {seed_value}")
            except (ValueError, TypeError):
                 print(f"Warning: Could not parse seed value '{raw_seed}'. Using random seed.")
                 # Optionally return an error if seed must be valid
                 # return jsonify({"error": f"Invalid seed value: {raw_seed}"}), 400
        else:
             print("No seed value provided. Using random seed.")
        # --- End seed parsing ---


        print(f"Parameters received: NP={NP}, GEN_DE={GEN_DE}, F={F}, CR={CR}, GA_GEN={ga_generations}, MutRate={mutation_rate}, TournSize={tournament_size}, Seed={seed_value}")

    except (ValueError, TypeError) as e:
        print(f"Error parsing parameters: {e}")
        return jsonify({"error": f"Invalid parameter type: {str(e)}"}), 400

    try:
        # Call the main function from hybrid_nn.py, passing the parsed seed
        results = run_de_ga_hybrid(
            NP=NP,
            GEN_DE=GEN_DE,
            F=F,
            CR=CR,
            ga_generations=ga_generations,
            mutation_rate=mutation_rate,
            tournament_size=tournament_size,
            L=L,
            H=H,
            seed=seed_value # Pass the seed here
        )
        print("Hybrid DE-GA run completed successfully.")
        return jsonify(results), 200

    except Exception as e:
        print(f"Error during hybrid NN execution: {e}")
        traceback.print_exc()
        return jsonify({"error": f"An error occurred during execution: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)