import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { useState } from "react";
import { Label } from "./ui/label";

export default function RunHybridNN({
  action,
  pending,
  title,
}: {
  action: (payload: FormData) => void;
  pending: boolean;
  title: string;
}) {
  const [population_size, setPopulation_size] = useState(50);
  const [de_generations, setDe_generations] = useState(100);
  const [f_factor, setF_factor] = useState(0.8);
  const [cr_rate, setCr_rate] = useState(0.7);
  const [ga_generations, setGa_generations] = useState(50);
  const [mutation_rate, setMutation_rate] = useState(0.05);
  const [tournament_size, setTournament_size] = useState(3);
  const [lower_bound, setLower_bound] = useState(-1.0);
  const [upper_bound, setUpper_bound] = useState(1.0);
  const [seed, setSeed] = useState<string>("");
  const [extinction_percentage, setExtinction_percentage] = useState(0.0); // New
  const [extinction_generation, setExtinction_generation] = useState(0); // New


  return (
    <form
      action={action}
      className="flex flex-col w-full max-w-sm gap-3 items-center p-2"
    >
      <h3 className="text-lg font-semibold mt-2">DE Parameters</h3>
      <div className="w-full">
        <Label htmlFor="population_size">Population Size (NP)</Label>
        <Input type="number" onChange={(e) => setPopulation_size(parseInt(e.target.value))} name="population_size" id="population_size" placeholder="e.g., 50" value={population_size} min={10} />
      </div>
      <div className="w-full">
        <Label htmlFor="de_generations">DE Generations</Label>
        <Input type="number" onChange={(e) => setDe_generations(parseInt(e.target.value))} name="de_generations" id="de_generations" placeholder="e.g., 100" value={de_generations} min={1}/>
      </div>
      <div className="w-full">
        <Label htmlFor="f_factor">DE Mutation Factor (F)</Label>
        <Input type="number" onChange={(e) => setF_factor(parseFloat(e.target.value))} name="f_factor" id="f_factor" placeholder="e.g., 0.8" value={f_factor} min={0.1} step={0.1} max={2.0}/>
      </div>
      <div className="w-full">
        <Label htmlFor="cr_rate">DE Crossover Rate (CR)</Label>
        <Input type="number" onChange={(e) => setCr_rate(parseFloat(e.target.value))} name="cr_rate" id="cr_rate" placeholder="e.g., 0.7" value={cr_rate} min={0.0} step={0.1} max={1.0}/>
      </div>

       <h3 className="text-lg font-semibold mt-4">GA Parameters</h3>
       <div className="w-full">
        <Label htmlFor="ga_generations">GA Generations</Label>
        <Input type="number" onChange={(e) => setGa_generations(parseInt(e.target.value))} name="ga_generations" id="ga_generations" placeholder="e.g., 50" value={ga_generations} min={1}/>
      </div>
      <div className="w-full">
        <Label htmlFor="mutation_rate">GA Mutation Rate</Label>
        <Input type="number" onChange={(e) => setMutation_rate(parseFloat(e.target.value))} name="mutation_rate" id="mutation_rate" placeholder="e.g., 0.05" value={mutation_rate} min={0.0} step={0.01} max={1.0}/>
      </div>
      <div className="w-full">
        <Label htmlFor="tournament_size">GA Tournament Size</Label>
        <Input type="number" onChange={(e) => setTournament_size(parseInt(e.target.value))} name="tournament_size" id="tournament_size" placeholder="e.g., 3" value={tournament_size} min={2}/>
      </div>

      <h3 className="text-lg font-semibold mt-4">Optional Bounds, Seed & Extinction</h3>
       <div className="w-full">
        <Label htmlFor="lower_bound">Lower Bound (L)</Label>
        <Input type="number" onChange={(e) => setLower_bound(parseFloat(e.target.value))} name="lower_bound" id="lower_bound" placeholder="e.g., -1.0" value={lower_bound} step={0.1}/>
      </div>
      <div className="w-full">
        <Label htmlFor="upper_bound">Upper Bound (H)</Label>
        <Input type="number" onChange={(e) => setUpper_bound(parseFloat(e.target.value))} name="upper_bound" id="upper_bound" placeholder="e.g., 1.0" value={upper_bound} step={0.1}/>
      </div>
      <div className="w-full">
        <Label htmlFor="seed">Random Seed (Optional)</Label>
        <Input type="number" onChange={(e) => setSeed(e.target.value)} name="seed" id="seed" placeholder="Leave empty for random" value={seed} min={0} step={1} />
      </div>

      {/* --- Add Extinction Inputs --- */}
      <div className="w-full">
        <Label htmlFor="extinction_percentage">DE Extinction % (0.0-1.0, 0=off)</Label>
        <Input type="number" onChange={(e) => setExtinction_percentage(parseFloat(e.target.value))} name="extinction_percentage" id="extinction_percentage" placeholder="e.g., 0.1 for 10%" value={extinction_percentage} min={0.0} max={1.0} step={0.05} />
      </div>
      <div className="w-full">
        <Label htmlFor="extinction_generation">DE Extinction Interval (Generations, 0=off)</Label>
        <Input type="number" onChange={(e) => setExtinction_generation(parseInt(e.target.value))} name="extinction_generation" id="extinction_generation" placeholder="e.g., 100" value={extinction_generation} min={0} step={10} />
      </div>
      {/* --- End Extinction Inputs --- */}

      <Button disabled={pending} type="submit" className="mt-4">
        {pending ? "Running..." : `Run ${title}`}
      </Button>
    </form>
  );
}