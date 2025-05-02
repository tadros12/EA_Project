// Potentially other imports if needed

export async function generateDataset(prevState: any, formData: FormData) {
  // ... your original code for generateDataset ...
  const n_customers = formData.get("n_customers");
  try {
      const data = await fetch(`http://127.0.0.1:5000/generate-dataset?customers=${n_customers}`);
      if (!data.ok) throw new Error(`HTTP error! Status: ${data.status}`);
      return await data.json();
  } catch (error) {
      console.error("Error generating dataset:", error);
      return { error: error instanceof Error ? error.message : "Failed to generate dataset" };
  }
}

export async function runGeneticAlgorithm(prevState: any, formData: FormData) {
  // ... your original code for runGeneticAlgorithm ...
  const population_size = formData.get("population_size");
  const generations = formData.get("generations");
  const mutation_rate = formData.get("mutation_rate");
  const retain_rate = formData.get("retain_rate");
  try {
      const response = await fetch("http://127.0.0.1:5000/generate-vrp-ga", { /* ... options ... */ });
      if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
      return await response.json();
  } catch (error) {
      console.error("Error running GA:", error);
      return { error: error instanceof Error ? error.message : "Failed to run GA" };
  }
}

export async function runDifferentialEvolution(prevState: any, formData: FormData) {
  // ... your original code for runDifferentialEvolution ...
   const population_size = formData.get("population_size");
   const generations = formData.get("generations");
   const F = formData.get("F");
   const CR = formData.get("CR");
   try {
       const response = await fetch("http://127.0.0.1:5000/generate-vrp-de", { /* ... options ... */ });
       if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
       return await response.json();
   } catch (error) {
       console.error("Error running DE:", error);
       return { error: error instanceof Error ? error.message : "Failed to run DE" };
   }
}

// Add the new function, also with 'export'
export async function runHybridNN(prevState: any, formData: FormData) {
  // Extract parameters - ensure names match the form's 'name' attributes
  const population_size = formData.get("population_size");
  const de_generations = formData.get("de_generations");
  const f_factor = formData.get("f_factor");
  const cr_rate = formData.get("cr_rate");
  const ga_generations = formData.get("ga_generations");
  const mutation_rate = formData.get("mutation_rate");
  const tournament_size = formData.get("tournament_size");
  const lower_bound = formData.get("lower_bound"); // Optional
  const upper_bound = formData.get("upper_bound"); // Optional


  console.log("Sending parameters to backend:", { // Log what's being sent
      population_size, de_generations, f_factor, cr_rate, ga_generations, mutation_rate, tournament_size, lower_bound, upper_bound
  });

  try {
    const response = await fetch("http://127.0.0.1:5000/run-hybrid-nn", { // New endpoint URL
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        population_size, // Key must match backend expectation
        de_generations,  // Key must match backend expectation
        f_factor,        // Key must match backend expectation
        cr_rate,         // Key must match backend expectation
        ga_generations,  // Key must match backend expectation
        mutation_rate,   // Key must match backend expectation
        tournament_size, // Key must match backend expectation
        lower_bound,     // Key must match backend expectation
        upper_bound      // Key must match backend expectation
      }),
    });

    if (!response.ok) {
      // Try to parse error message from backend if available
      let errorData;
      try {
          errorData = await response.json();
      } catch (parseError) {
          // If parsing fails, use the status text
          throw new Error(`HTTP error! Status: ${response.status} - ${response.statusText}`);
      }
      console.error("Backend error response:", errorData);
      throw new Error(errorData.error || `HTTP error! Status: ${response.status}`);
    }

    const results = await response.json();
    console.log("Received results from backend:", results); // Log successful results
    return results; // Return the JSON containing history arrays etc.

  } catch (error) {
    console.error("Error fetching hybrid NN data:", error);
    // Return an object with an error field, compatible with useActionState
    return {
      error: error instanceof Error ? error.message : "Failed to fetch hybrid NN data"
    };
  }
}
