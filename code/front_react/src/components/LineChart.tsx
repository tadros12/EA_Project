import React from "react";
import Plot from "react-plotly.js";
import { Data } from "plotly.js";

// --- Updated Props Interface ---
interface LineChartProps {
  de_history?: number[];
  ga_history?: number[];
  de_generations_count?: number;
}

const LineChart: React.FC<LineChartProps> = ({
  de_history,
  ga_history,
  de_generations_count,
}) => {
  // --- Safety checks ---
  const safeDeHistory = de_history && de_history.length > 0 ? de_history : [];
  const safeGaHistory = ga_history && ga_history.length > 0 ? ga_history : [];
  const safeDeGenCount = de_generations_count !== undefined ? de_generations_count : safeDeHistory.length; // Use actual length if count not passed

  // --- Generate X-axis values ---
  const deGenerationsX = Array.from({ length: safeDeHistory.length }, (_, i) => i);
  // GA generations start after DE finishes
  const gaGenerationsX = Array.from({ length: safeGaHistory.length }, (_, i) => safeDeGenCount + i); // Start from de_generations_count

  // --- Prepare Plotly data traces ---
  const plotData: Data[] = [];

  if (safeDeHistory.length > 0) {
    plotData.push({
      x: deGenerationsX,
      y: safeDeHistory,
      type: "scatter",
      mode: "lines",
      name: "DE Phase Loss", // Legend entry for DE
      line: { color: 'blue' } // Assign color
    });
  }

  if (safeGaHistory.length > 0) {
    // Optionally, add the last point of DE history to the start of GA for continuity
    const gaY = (safeDeHistory.length > 0 && safeGaHistory.length > 0)
      ? [safeDeHistory[safeDeHistory.length - 1], ...safeGaHistory]
      : safeGaHistory;
    const gaX = (safeDeHistory.length > 0 && safeGaHistory.length > 0)
      ? [safeDeGenCount - 1, ...gaGenerationsX] // Start GA line from last DE point
      : gaGenerationsX;

     plotData.push({
       x: gaX,
       y: gaY,
       type: "scatter",
       mode: "lines",
       name: "GA Phase Loss", // Legend entry for GA
       line: { color: 'red' } // Assign color
     });
  }


  // --- Render placeholder if no data ---
  if (plotData.length === 0) {
    return (
      <div className="w-full h-64 flex items-center justify-center text-gray-500 border rounded my-4"> {/* Added some styling */}
        Run the algorithm to see performance data.
      </div>
    );
  }

  // --- Prepare Layout ---
  const layout = {
    title: "Optimization Progress (Loss vs. Generation)", // Updated title
    xaxis: { title: "Generation" },
    yaxis: { title: "Loss (Lower is Better)" }, // Updated label
    showlegend: true,
    margin: { t: 50, l: 60, r: 30, b: 50 }, // Adjusted margins
    hovermode: 'x unified' // Improved hover behavior
  };

  // Add vertical line at transition if both phases exist
  if (safeDeHistory.length > 0 && safeGaHistory.length > 0 && safeDeGenCount > 0) {
    const minY = Math.min(...safeDeHistory, ...safeGaHistory);
    const maxY = Math.max(...safeDeHistory, ...safeGaHistory);
    layout.shapes = [{
        type: 'line',
        x0: safeDeGenCount -1, // Line at the end of DE phase
        y0: minY,
        x1: safeDeGenCount -1,
        y1: maxY,
        line: {
          color: 'grey',
          width: 2,
          dash: 'dashdot',
        }
      }];
      layout.annotations = [{
          x: safeDeGenCount - 1,
          y: safeDeHistory[safeDeHistory.length - 1],
          xref: 'x',
          yref: 'y',
          text: 'DE → GA',
          showarrow: true,
          arrowhead: 7,
          ax: 0,
          ay: -40
      }];
  }


  // --- Render Plot ---
  return (
    <div className="w-full h-full my-4"> {/* Added margin */}
      <Plot
        data={plotData}
        layout={layout}
        useResizeHandler={true}
        className="w-full h-full" // Ensure Plotly takes available space
        config={{ responsive: true }}
      />
    </div>
  );
};

export default LineChart;