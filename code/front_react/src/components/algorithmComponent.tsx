import LineChart from "../components/LineChart";

type TestSample = {
  image_b64: string;
  true_label: number | string;
  prediction: number | string;
};

type ResultState = {
  algorithm_run?: 'DE-GA' | 'GA-DE';
  de_history?: number[];
  ga_history?: number[];
  final_loss?: number;
  final_accuracy?: number;
  de_generations_count?: number;
  ga_generations_count?: number;
  error?: string;
  test_samples?: TestSample[];
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
          <p className="text-lg font-bold">
            {(state.final_accuracy != null ? state.final_accuracy * 100 : NaN).toFixed(2) ?? 'N/A'}%
          </p>
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

      {/* Plot */}
      {(state.de_history && state.de_history.length > 0) || (state.ga_history && state.ga_history.length > 0) ? (
        <div className="w-full max-w-3xl">
          <LineChart
            de_history={state.de_history}
            ga_history={state.ga_history}
            de_generations_count={state.de_generations_count}
          />
        </div>
      ) : null}

      {/* History Lists */}
      <div className="flex flex-col md:flex-row justify-around w-full max-w-3xl gap-4 mt-4">
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

      {/* Test samples table */}
      {state.test_samples && state.test_samples.length > 0 && (
        <div className="w-full max-w-3xl mt-8">
          <h3 className="text-lg font-medium mb-2 text-center">Test Samples & Predictions</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full border rounded bg-muted/50 text-sm">
              <thead>
                <tr>
                  <th className="border p-2">Sample #</th>
                  <th className="border p-2">Image</th>
                  <th className="border p-2">True Label</th>
                  <th className="border p-2">Prediction</th>
                </tr>
              </thead>
              <tbody>
                {state.test_samples.map((sample, idx) => (
                  <tr key={idx}>
                    <td className="border p-2 text-center">{idx + 1}</td>
                    <td className="border p-2 text-center">
                      <img
                        src={`data:image/png;base64,${sample.image_b64}`}
                        alt={`Sample ${idx + 1}`}
                        style={{ width: 48, height: 48, imageRendering: "pixelated" }}
                      />
                    </td>
                    <td className="border p-2 text-blue-700 font-semibold">{sample.true_label}</td>
                    <td className={`border p-2 font-semibold ${
                      sample.prediction === sample.true_label ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {sample.prediction}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="text-xs text-muted-foreground text-center mt-1">
              (Showing 5 random test samples)
            </div>
          </div>
        </div>
      )}
    </div>
  );
}