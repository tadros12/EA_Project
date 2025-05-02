// --- Imports ---
import { Badge } from "./ui/badge";
// import ScatterChart from "./ScatterChart"; // Remove this import
import LineChart from "./LineChart";
// import GenerationsTable from "./generationsTable"; // Already removed
// import RunGA from "./runGA"; // Remove this import
// import RunDE from "./runDE"; // Remove this import
import RunHybridNN from "./RunHybridNN"; // Import the new form
import { ScrollArea } from "./ui/scroll-area";
import { useActionState } from "react"; // Assuming you use useActionState
import { runHybridNN } from "@/lib/actions"; // Import the new action

// --- Define the expected message type from the new backend ---
interface HybridNNMessage {
  de_history: number[];
  ga_history: number[];
  de_generations_count: number;
  ga_generations_count: number; // Optional, but good to have
  final_loss?: number; // Optional
  test_accuracy?: number; // Optional
  error?: string; // To handle potential errors from the action
  // best_solution_weights is likely too large to display directly
}

// --- Component Definition ---
// Removed DSmessage, DSisPending props as they related to the VRP dataset generation
export default function AlgorithmComponent({
  // GAmessageSt, // Rename state variable below
  // GAisPending, // Rename state variable below
  // GAformAction, // Replace with new action
  headerTitle, // Keep or update as needed
}: {
  headerTitle: string; // Example: "Hybrid DE-GA Neural Network Training"
}) {
  // --- State Management for the new action ---
  // Use useActionState (React 19+) or useState/useEffect with manual fetch
  // This example uses useActionState
  const [hybridNNState, hybridNNFormAction, isHybridNNPending] = useActionState<HybridNNMessage | null, FormData>(
    runHybridNN, // Use the new action function
    null // Initial state
  );

  // Determine if loading indicator should show
  const showLoader = isHybridNNPending;

  // Extract data for display (handle potential null state and errors)
  const finalLoss = hybridNNState?.final_loss?.toFixed(6); // Display more precision for loss
  const testAccuracy = hybridNNState?.test_accuracy
    ? (hybridNNState.test_accuracy * 100).toFixed(2) + "%"
    : null; // Optional accuracy display

  // Extract data for the LineChart
  const deHistory = hybridNNState?.de_history;
  const gaHistory = hybridNNState?.ga_history;
  const deGenCount = hybridNNState?.de_generations_count;

  // Handle potential errors from the backend action
  const errorMessage = hybridNNState?.error;

  return (
    <>
      {/* Central Content Area */}
      <ScrollArea className="p-2 w-2/3 max-w-full relative">
        <h1 className="text-3xl font-bold mb-2 text-center">{headerTitle}</h1>

        {/* Loading Indicator */}
        {showLoader && (
          <div className=" absolute h-full w-full flex justify-center items-center bg-transparent backdrop-blur-sm z-10">
            {/* Consider a simpler loader if the cat is too much */}
            <img
              className="bg-transparent w-48 h-48 rounded-3xl" // Slightly smaller
              src="/spinning-cat.gif" // Make sure this GIF is in your public folder
              alt="Loading..."
            />
            {/* Maybe remove the audio or make it controllable */}
            {/* <audio src="/Crambone.mp3" preload="auto" autoPlay loop hidden /> */}
          </div>
        )}

        {/* Display Error Message */}
        {errorMessage && !showLoader && (
          <div className="w-full my-3 p-3 bg-red-100 border border-red-400 text-red-700 rounded text-center">
            <p>Error: {errorMessage}</p>
          </div>
        )}


        {/* Result Badges */}
        {(finalLoss || testAccuracy) && !showLoader && !errorMessage && (
          <div className="w-full flex flex-wrap justify-center gap-4 my-3">
            {finalLoss && (
              <Badge
                variant="outline"
                className="text-xl p-2 hover:bg-accent transition-colors"
              >
                Final Loss: {finalLoss}
              </Badge>
            )}
             {testAccuracy && (
              <Badge
                variant="outline"
                className="text-xl p-2 hover:bg-accent transition-colors"
              >
                Test Accuracy: {testAccuracy}
              </Badge>
            )}
          </div>
        )}

        {/* Customer Locations Scatter Chart - REMOVED */}
        {/* {DSmessage && !DSisPending && !GAisPending && ( ... )} */}

        {/* Performance Line Chart - Updated Props */}
        {/* Only render chart if history data exists */}
        {deHistory && gaHistory && deGenCount && !showLoader && !errorMessage && (
          <LineChart
            // Pass the history arrays and DE generation count
            de_history={deHistory}
            ga_history={gaHistory}
            de_generations_count={deGenCount}
          />
        )}
      </ScrollArea>

      {/* Right Sidebar - Hybrid NN Controls */}
      <div className="w-1/3 p-2 flex flex-col justify-between">
        {/* Generations Table - REMOVED */}
        {/* <GenerationsTable generationsRes={GAmessageSt} /> */}

        {/* Use the new RunHybridNN form */}
        <RunHybridNN
          action={hybridNNFormAction} // Pass the action dispatcher
          pending={isHybridNNPending}  // Pass the pending state
          title={"Hybrid DE-GA"}    // Updated title for the button
        />
      </div>
    </>
  );
}