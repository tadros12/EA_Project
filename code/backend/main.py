import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from hybrid_nn import run_de_ga_hybrid, run_ga_de_hybrid
import traceback


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})


@app.route("/run-hybrid-de-ga", methods=["POST"])
def run_hybrid_de_ga_endpoint():
    print("Received request for /run-hybrid-de-ga")
    data = request.json or {}
    seed_value = None
    extinction_perc_val = 0.0
    extinction_gen_val = 0

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

        raw_seed = data.get("seed")
        if raw_seed is not None and str(raw_seed).strip():
            try:
                 seed_value = int(raw_seed)
            except (ValueError, TypeError):
                 print(f"Warning: Could not parse seed value '{raw_seed}'. Using random seed.")

        raw_ext_perc = data.get("extinction_percentage")
        if raw_ext_perc is not None and str(raw_ext_perc).strip():
             try:
                  extinction_perc_val = float(raw_ext_perc)
                  if not (0.0 <= extinction_perc_val <= 1.0):
                       print(f"Warning: Extinction percentage {extinction_perc_val} out of range [0, 1]. Clamping.")
                       extinction_perc_val = max(0.0, min(1.0, extinction_perc_val))
             except (ValueError, TypeError):
                  print(f"Warning: Invalid extinction percentage '{raw_ext_perc}'. Using 0.")

        raw_ext_gen = data.get("extinction_generation")
        if raw_ext_gen is not None and str(raw_ext_gen).strip():
             try:
                  extinction_gen_val = int(raw_ext_gen)
                  if extinction_gen_val < 0:
                       print(f"Warning: Negative extinction generation '{raw_ext_gen}'. Using 0.")
                       extinction_gen_val = 0
             except (ValueError, TypeError):
                  print(f"Warning: Invalid extinction generation '{raw_ext_gen}'. Using 0.")

        print(f"DE-GA Params: NP={NP}, GEN_DE={GEN_DE}, F={F}, CR={CR}, GA_GEN={ga_generations}, MutRate={mutation_rate}, TournSize={tournament_size}, Seed={seed_value}, ExtPerc={extinction_perc_val}, ExtGen={extinction_gen_val}")

    except (ValueError, TypeError) as e:
        print(f"Error parsing parameters: {e}")
        return jsonify({"error": f"Invalid parameter type: {str(e)}"}), 400

    try:
        results = run_de_ga_hybrid(
            NP=NP, GEN_DE=GEN_DE, F=F, CR=CR,
            ga_generations=ga_generations, mutation_rate=mutation_rate, tournament_size=tournament_size,
            L=L, H=H, seed=seed_value,
            extinction_percentage=extinction_perc_val, extinction_generation=extinction_gen_val
        )
        print("Hybrid DE-GA run completed successfully.")
        return jsonify(results), 200

    except Exception as e:
        print(f"Error during hybrid DE-GA execution: {e}")
        traceback.print_exc()
        return jsonify({"error": f"An error occurred during execution: {str(e)}"}), 500


@app.route("/run-hybrid-ga-de", methods=["POST"])
def run_hybrid_ga_de_endpoint():
    print("Received request for /run-hybrid-ga-de")
    data = request.json or {}
    seed_value = None
    extinction_perc_val = 0.0
    extinction_gen_val = 0

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

        raw_seed = data.get("seed")
        if raw_seed is not None and str(raw_seed).strip():
            try:
                 seed_value = int(raw_seed)
            except (ValueError, TypeError):
                 print(f"Warning: Could not parse seed value '{raw_seed}'. Using random seed.")

        raw_ext_perc = data.get("extinction_percentage")
        if raw_ext_perc is not None and str(raw_ext_perc).strip():
             try:
                  extinction_perc_val = float(raw_ext_perc)
                  if not (0.0 <= extinction_perc_val <= 1.0):
                       print(f"Warning: Extinction percentage {extinction_perc_val} out of range [0, 1]. Clamping.")
                       extinction_perc_val = max(0.0, min(1.0, extinction_perc_val))
             except (ValueError, TypeError):
                  print(f"Warning: Invalid extinction percentage '{raw_ext_perc}'. Using 0.")

        raw_ext_gen = data.get("extinction_generation")
        if raw_ext_gen is not None and str(raw_ext_gen).strip():
             try:
                  extinction_gen_val = int(raw_ext_gen)
                  if extinction_gen_val < 0:
                       print(f"Warning: Negative extinction generation '{raw_ext_gen}'. Using 0.")
                       extinction_gen_val = 0
             except (ValueError, TypeError):
                  print(f"Warning: Invalid extinction generation '{raw_ext_gen}'. Using 0.")

        print(f"GA-DE Params: NP={NP}, GA_GEN={ga_generations}, MutRate={mutation_rate}, TournSize={tournament_size}, GEN_DE={GEN_DE}, F={F}, CR={CR}, Seed={seed_value}, ExtPerc={extinction_perc_val}, ExtGen={extinction_gen_val}")

    except (ValueError, TypeError) as e:
        print(f"Error parsing parameters: {e}")
        return jsonify({"error": f"Invalid parameter type: {str(e)}"}), 400

    try:
        results = run_ga_de_hybrid(
            NP=NP, GEN_DE=GEN_DE, F=F, CR=CR,
            ga_generations=ga_generations, mutation_rate=mutation_rate, tournament_size=tournament_size,
            L=L, H=H, seed=seed_value,
            extinction_percentage=extinction_perc_val, extinction_generation=extinction_gen_val
        )
        print("Hybrid GA-DE run completed successfully.")
        return jsonify(results), 200

    except Exception as e:
        print(f"Error during hybrid GA-DE execution: {e}")
        traceback.print_exc()
        return jsonify({"error": f"An error occurred during execution: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
