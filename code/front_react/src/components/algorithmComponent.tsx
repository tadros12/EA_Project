import LineChart from "@/components/LineChart";

type ResultState = {
  algorithm_run?: 'DE-GA' | 'GA-DE';
  de_history?: number[];
  ga_history?: number[];
  final_loss?: number;
  final_accuracy?: number;
  de_generations_count?: number;
  ga_generations_count?: number;
  error?: string;
};

export default function AlgorithmComponent({
  state,
  elapsed,
}: {
  state: ResultState | null;
  elapsed?: number | null;
}) {
  if (!state) {
    return <div className="text-center text-muted-foreground">Run an algorithm to see results.</div>;
  }

  if (state.error) {
    return <div className="text-destructive text-center p-4">Error: {state.error}</div>;
  }

  // Helper to format ms as s or m:s
  const formatElapsed = (ms?: number | null) => {
    if (!ms || ms < 0) return null;
    if (ms < 1000) return `${ms} ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(2)} s`;
    const min = Math.floor(ms / 60000);
    const sec = ((ms % 60000) / 1000).toFixed(2);
    return `${min}m ${sec}s`;
  };

  return (
    <div className="flex flex-col items-center p-4 gap-4 w-full">
      <h2 className="text-xl font-semibold">Results ({state.algorithm_run})</h2>

      <div className="flex justify-around w-full max-w-md text-center">
          <div>
              <p className="text-muted-foreground text-sm">Final Loss</p>
              <p className="text-lg font-bold">{state.final_loss?.toFixed(6) ?? 'N/A'}</p>
          </div>
          <div>
              <p className="text-muted-foreground text-sm">Final Accuracy</p>
              <p className="text-lg font-bold">{(state.final_accuracy != null ? state.final_accuracy * 100 : NaN).toFixed(2) ?? 'N/A'}%</p>
          </div>
      </div>

      {elapsed != null && (
        <div className="mt-2 mb-2">
          <span className="font-semibold text-blue-500">
            ⏱️ Elapsed Time:&nbsp;
          </span>
          <span>{formatElapsed(elapsed)}</span>
        </div>
      )}

      {/* Plot only if there's data */}
      {(state.de_history && state.de_history.length > 0) || (state.ga_history && state.ga_history.length > 0) ? (
        <div className="w-full max-w-3xl">
          <LineChart
            de_history={state.de_history}
            ga_history={state.ga_history}
            de_generations_count={state.de_generations_count}
          />
        </div>
      ) : null}

      <div className="flex flex-col md:flex-row justify-around w-full max-w-3xl gap-4 mt-4">
        {/* DE History Box (always rendered) */}
        <div className="flex-1">
          <h3 className="text-lg font-medium mb-2 text-center">DE History (Loss)</h3>
          <div className="h-48 overflow-y-auto border rounded p-2 bg-muted/50 text-sm">
            {Array.isArray(state.de_history) && state.de_history.length > 0 ? (
              <ul>
                {state.de_history.map((loss, index) => (
                  <li key={`de-${index}`}>Gen {index + 1}: {loss.toFixed(6)}</li>
                ))}
              </ul>
            ) : (
              <div className="text-center text-muted-foreground">No DE History</div>
            )}
          </div>
        </div>
        {/* GA History Box (always rendered) */}
        <div className="flex-1">
          <h3 className="text-lg font-medium mb-2 text-center">GA History (Loss)</h3>
          <div className="h-48 overflow-y-auto border rounded p-2 bg-muted/50 text-sm">
            {Array.isArray(state.ga_history) && state.ga_history.length > 0 ? (
              <ul>
                {state.ga_history.map((loss, index) => (
                  <li key={`ga-${index}`}>Gen {index + 1}: {loss.toFixed(6)}</li>
                ))}
              </ul>
            ) : (
              <div className="text-center text-muted-foreground">No GA History</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}