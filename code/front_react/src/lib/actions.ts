export async function runDeGaHybrid(prevState: any, formData: FormData) {
  const population_size = formData.get("population_size");
  const de_generations = formData.get("de_generations");
  const f_factor = formData.get("f_factor");
  const cr_rate = formData.get("cr_rate");
  const ga_generations = formData.get("ga_generations");
  const mutation_rate = formData.get("mutation_rate");
  const tournament_size = formData.get("tournament_size");
  const lower_bound = formData.get("lower_bound");
  const upper_bound = formData.get("upper_bound");
  const seed = formData.get("seed");
  const extinction_percentage = formData.get("extinction_percentage"); // New
  const extinction_generation = formData.get("extinction_generation"); // New

  const payload = {
      population_size, de_generations, f_factor, cr_rate, ga_generations, mutation_rate, tournament_size, lower_bound, upper_bound, seed, extinction_percentage, extinction_generation // Include new params
  };
  console.log("Sending DE-GA parameters to backend:", payload);

  try {
    const response = await fetch("http://127.0.0.1:5000/run-hybrid-de-ga", { // Updated endpoint
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      let errorData;
      try { errorData = await response.json(); }
      catch (parseError) { throw new Error(`HTTP error! Status: ${response.status} - ${response.statusText}`); }
      console.error("Backend error response:", errorData);
      throw new Error(errorData?.error || `HTTP error! Status: ${response.status}`);
    }

    const results = await response.json();
    console.log("Received DE-GA results from backend:", results);
    return results;

  } catch (error) {
    console.error("Error caught in runDeGaHybrid action:", error);
    let errorMessage = "An unknown error occurred during the DE-GA operation.";
    if (error instanceof Error) { errorMessage = error.message; }
    else if (typeof error === 'string') { errorMessage = error; }
    return { error: errorMessage };
  }
}


export async function runGaDeHybrid(prevState: any, formData: FormData) {
  const population_size = formData.get("population_size");
  const de_generations = formData.get("de_generations");
  const f_factor = formData.get("f_factor");
  const cr_rate = formData.get("cr_rate");
  const ga_generations = formData.get("ga_generations");
  const mutation_rate = formData.get("mutation_rate");
  const tournament_size = formData.get("tournament_size");
  const lower_bound = formData.get("lower_bound");
  const upper_bound = formData.get("upper_bound");
  const seed = formData.get("seed");
  const extinction_percentage = formData.get("extinction_percentage"); // New
  const extinction_generation = formData.get("extinction_generation"); // New

  const payload = {
      population_size, de_generations, f_factor, cr_rate, ga_generations, mutation_rate, tournament_size, lower_bound, upper_bound, seed, extinction_percentage, extinction_generation // Include new params
  };
  console.log("Sending GA-DE parameters to backend:", payload);

  try {
    const response = await fetch("http://127.0.0.1:5000/run-hybrid-ga-de", { // New endpoint
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      let errorData;
      try { errorData = await response.json(); }
      catch (parseError) { throw new Error(`HTTP error! Status: ${response.status} - ${response.statusText}`); }
      console.error("Backend error response:", errorData);
      throw new Error(errorData?.error || `HTTP error! Status: ${response.status}`);
    }

    const results = await response.json();
    console.log("Received GA-DE results from backend:", results);
    return results;

  } catch (error) {
    console.error("Error caught in runGaDeHybrid action:", error);
    let errorMessage = "An unknown error occurred during the GA-DE operation.";
    if (error instanceof Error) { errorMessage = error.message; }
    else if (typeof error === 'string') { errorMessage = error; }
    return { error: errorMessage };
  }
}