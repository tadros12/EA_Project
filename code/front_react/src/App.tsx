// Remove unused imports if generateDataset is no longer needed
// import { generateDataset, runGeneticAlgorithm, runDifferentialEvolution } from "./lib/actions";
// import SubmitCustomersButton from "./components/SubmitCustomersButton";
// import CustomersTable from "./components/CustomersTable";

import AlgorithmComponent from "./components/algorithmComponent";
import { ThemeProvider } from "./components/theme-provider";
// import { useActionState } from "react"; // Moved into AlgorithmComponent


function App() {
  // If you were using useActionState here, it's now inside AlgorithmComponent
  // const [DSmessage, DSformAction, DSisPending] = useActionState(generateDataset, null);

  return (
    <ThemeProvider defaultTheme="dark" storageKey="vite-ui-theme">
      <div className="flex h-screen bg-background text-foreground">
        {/* Left Sidebar (If you had one for dataset generation) - Can be removed or repurposed */}
        {/*
        <div className="w-1/4 p-4 border-r flex flex-col justify-between">
           <CustomersTable customers={DSmessage} />
           <SubmitCustomersButton action={DSformAction} setClear={()=>{}} pending={DSisPending} />
        </div>
        */}

        {/* Main Content Area using AlgorithmComponent */}
        {/* Pass the new title */}
        <AlgorithmComponent headerTitle="Hybrid DE-GA Neural Network Training" />

         {/* Optional: Keep theme toggle */}
         <div className="absolute top-4 right-4">
        </div>
      </div>
    </ThemeProvider>
  );
}

export default App;