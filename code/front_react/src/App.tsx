import { useState, useEffect, useRef } from "react";
import { useActionState } from "react";
import { ThemeProvider } from "./components/theme-provider";
import RunHybridNN from "./components/RunHybridNN";
import AlgorithmComponent from "./components/algorithmComponent";
import { runDeGaHybrid, runGaDeHybrid } from "./lib/actions";
import { Separator } from "@/components/ui/separator";

type AlgorithmType = 'DE_GA' | 'GA_DE';

const loadingGifPath = '/spinning-cat.gif';
const backgroundMusicPath = '/Elevator Music (Kevin MacLeod) - Background Music (HD).mp3';

function App() {
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<AlgorithmType>('DE_GA');
  const [deGaState, submitDeGaAction, isDeGaPending] = useActionState(runDeGaHybrid, null);
  const [gaDeState, submitGaDeAction, isGaDePending] = useActionState(runGaDeHybrid, null);

  const [startTime, setStartTime] = useState<number | null>(null);
  const [elapsed, setElapsed] = useState<number | null>(null);

  const audioRef = useRef<HTMLAudioElement | null>(null);

  const currentAction = selectedAlgorithm === 'DE_GA' ? submitDeGaAction : submitGaDeAction;
  const currentState = selectedAlgorithm === 'DE_GA' ? deGaState : gaDeState;
  const isPending = selectedAlgorithm === 'DE_GA' ? isDeGaPending : isGaDePending;
  const currentTitle = selectedAlgorithm === 'DE_GA' ? 'Hybrid DE-GA' : 'Hybrid GA-DE';

  // Track start/stop time for algorithm run
  useEffect(() => {
    if (isPending && startTime === null) {
      setStartTime(Date.now());
      setElapsed(null);
    }
    if (!isPending && startTime !== null) {
      setElapsed(Date.now() - startTime);
      setStartTime(null);
    }
    // eslint-disable-next-line
  }, [isPending]);

  // Reset elapsed on algorithm switch or rerun
  useEffect(() => {
    setElapsed(null);
    setStartTime(null);
  }, [selectedAlgorithm]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'ArrowDown' || event.key === 'ArrowUp' || event.key === 'Tab') {
        event.preventDefault();
        setSelectedAlgorithm(prev => (prev === 'DE_GA' ? 'GA_DE' : 'DE_GA'));
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => { window.removeEventListener('keydown', handleKeyDown); };
  }, []);

  useEffect(() => {
    if (isPending && audioRef.current) {
      audioRef.current.currentTime = 0;
      audioRef.current.play().catch(() => {});
    } else if (!isPending && audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
  }, [isPending]);

  // Custom action handler to wrap with timer
  const wrappedAction = async (formData: FormData) => {
    setStartTime(Date.now());
    setElapsed(null);
    return await currentAction(formData);
  }

  return (
    <ThemeProvider defaultTheme="dark" storageKey="vite-ui-theme">
      <div className="flex flex-col md:flex-row h-screen bg-background text-foreground relative">
        {/* Fullscreen Loading Overlay */}
        {isPending && (
          <div
            className="fixed inset-0 bg-black bg-opacity-70 flex flex-col justify-center items-center z-50"
            style={{ pointerEvents: "all" }}>
            <img
              src={loadingGifPath}
              alt="Loading..."
              className="w-40 h-40 mb-4"
              style={{ filter: "drop-shadow(0 0 12px #fff8)" }}
            />
            <audio
              ref={audioRef}
              src={backgroundMusicPath}
              loop
              preload="auto"
              style={{ display: "none" }}
            />
            <div className="text-xl text-white text-center font-bold mt-2">
              Please wait, your neural network is being optimized...
            </div>
          </div>
        )}
        {/* Main App Layout */}
        <div className="w-full md:w-1/3 lg:w-1/4 p-4 border-r flex flex-col items-center overflow-y-auto">
          <h1 className="text-2xl font-bold mb-4">Hybrid NN Optimizer</h1>
          <p className="text-muted-foreground mb-1 text-center">Use Arrow Up/Down or Tab to switch</p>
          <p className="text-lg font-semibold mb-4 text-center">{currentTitle}</p>
          <RunHybridNN
            action={wrappedAction}
            pending={isPending}
            title={currentTitle}
          />
        </div>
        <div className="w-full md:w-2/3 lg:w-3/4 p-4 overflow-y-auto">
          <AlgorithmComponent state={currentState} elapsed={elapsed} />
          <Separator className="my-4" />
          <div className="text-center text-muted-foreground text-sm">
            created by : tadros  | contributors: tadros
          </div>
        </div>
      </div>
    </ThemeProvider>
  );
}

export default App;