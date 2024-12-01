import React, {
  useState,
  useEffect,
  useCallback,
  useMemo,
  useRef,
} from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { LineChart, Line, XAxis, YAxis, CartesianGrid } from "recharts";
import { RefreshCcw, Pause, Play } from "lucide-react";

const Knob = ({
  value,
  min,
  max,
  step,
  onChange,
  label,
  formatValue = (v) => v.toFixed(1),
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [inputValue, setInputValue] = useState(formatValue(value));
  const startPosRef = useRef({ y: 0, value: 0 });
  const sensitivityFactor = 200;

  const getRotation = (val) => {
    const range = max - min;
    return -150 + ((val - min) / range) * 300;
  };

  const handlePointerEvents = useCallback(
    (e, type) => {
      if (type === "down") {
        e.currentTarget.setPointerCapture(e.pointerId);
        setIsDragging(true);
        startPosRef.current = { y: e.clientY, value };
      } else if (type === "move" && isDragging) {
        const deltaY = startPosRef.current.y - e.clientY;
        const range = max - min;
        const deltaValue = (deltaY * range) / sensitivityFactor;
        const newValue = Math.min(
          max,
          Math.max(min, startPosRef.current.value + deltaValue),
        );
        onChange(Math.round(newValue / step) * step);
      } else if (type === "up") {
        setIsDragging(false);
      }
    },
    [isDragging, max, min, step, value, onChange],
  );

  const handleEditing = (e, type) => {
    if (type === "change") setInputValue(e.target.value);
    else if (type === "blur" || (type === "key" && e.key === "Enter")) {
      const newValue = parseFloat(inputValue);
      if (!isNaN(newValue))
        onChange(
          Math.round(Math.min(max, Math.max(min, newValue)) / step) * step,
        );
      setIsEditing(false);
    } else if (type === "key" && e.key === "Escape") {
      setInputValue(formatValue(value));
      setIsEditing(false);
    }
  };

  return (
    <div className="flex flex-col items-center space-y-2">
      <span className="text-sm text-gray-600">{label}</span>
      <div
        className="relative w-16 h-16 cursor-pointer select-none"
        onPointerDown={(e) => handlePointerEvents(e, "down")}
        onPointerMove={(e) => handlePointerEvents(e, "move")}
        onPointerUp={(e) => handlePointerEvents(e, "up")}
        onPointerCancel={(e) => handlePointerEvents(e, "up")}
      >
        <div className="absolute inset-0 rounded-full bg-gray-200" />
        <div
          className="absolute inset-0 rounded-full bg-blue-500"
          style={{
            clipPath: "polygon(50% 50%, 50% 0%, 52% 0%, 52% 50%)",
            transform: `rotate(${getRotation(value)}deg)`,
            transition: isDragging ? "none" : "transform 0.1s",
          }}
        />
        <div className="absolute inset-[30%] rounded-full bg-white shadow-md" />
      </div>
      {isEditing ? (
        <Input
          type="text"
          value={inputValue}
          onChange={(e) => handleEditing(e, "change")}
          onBlur={() => handleEditing(null, "blur")}
          onKeyDown={(e) => handleEditing(e, "key")}
          className="w-16 h-6 px-1 text-center text-sm font-mono"
          autoFocus
        />
      ) : (
        <span
          className="text-sm font-mono cursor-pointer select-none"
          onDoubleClick={() => setIsEditing(true)}
        >
          {formatValue(value)}
        </span>
      )}
    </div>
  );
};

const BrownianMotion3D = () => {
  const [numParticles, setNumParticles] = useState(5);
  const [speed, setSpeed] = useState(1);
  const [diffusionCoeff, setDiffusionCoeff] = useState(0.2);
  const [trailLength, setTrailLength] = useState(50);
  const [showTrails, setShowTrails] = useState(true);
  const [showStatistics, setShowStatistics] = useState(true);
  const [showAxes, setShowAxes] = useState(false);
  const [isPlaying, setIsPlaying] = useState(true);
  const [particles, setParticles] = useState([]);

  const initializeParticles = useCallback(
    () =>
      Array.from({ length: numParticles }, () => ({
        position: { x: 0, y: 0, z: 0 },
        trail: [{ x: 0, y: 0, z: 0 }],
      })),
    [numParticles],
  );

  const handleReset = useCallback(() => {
    setIsPlaying(false);
    setParticles(initializeParticles());
  }, [initializeParticles]);

  useEffect(() => {
    setParticles(initializeParticles());
  }, [numParticles, initializeParticles]);

  // Animation logic, particle bouncing, and MSD calculation here...

  return (
    <Card className="w-full max-w-3xl">
      <CardHeader className="flex justify-between items-center">
        <CardTitle>3D Brownian Motion</CardTitle>
        <div className="flex space-x-2">
          <Button
            variant="outline"
            size="icon"
            onClick={() => setIsPlaying((prev) => !prev)}
            className="h-8 w-8"
          >
            {isPlaying ? (
              <Pause className="h-4 w-4" />
            ) : (
              <Play className="h-4 w-4" />
            )}
          </Button>
          <Button
            variant="outline"
            size="icon"
            onClick={handleReset}
            className="h-8 w-8"
          >
            <RefreshCcw className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          <div className="relative aspect-square w-full">
            {/* SVG rendering of Brownian motion */}
          </div>
          {showStatistics && (
            <LineChart width={700} height={200} data={statisticsData}>
              {/* Chart components */}
            </LineChart>
          )}
          <div className="flex flex-wrap justify-center gap-8">
            <Knob
              label="Particles"
              value={numParticles}
              min={1}
              max={10}
              step={1}
              onChange={setNumParticles}
            />
            <Knob
              label="Speed"
              value={speed}
              min={0.1}
              max={3}
              step={0.1}
              onChange={setSpeed}
            />
            <Knob
              label="Diffusion"
              value={diffusionCoeff}
              min={0.1}
              max={2}
              step={0.1}
              onChange={setDiffusionCoeff}
            />
            <Knob
              label="Trail Length"
              value={trailLength}
              min={1}
              max={200}
              step={1}
              onChange={setTrailLength}
            />
          </div>
          <div className="flex justify-center space-x-6">
            {[
              ["Show Trails", showTrails, setShowTrails],
              ["Show Statistics", showStatistics, setShowStatistics],
              ["Show Axes", showAxes, setShowAxes],
            ].map(([label, state, setState]) => (
              <div key={label} className="flex items-center space-x-2">
                <Switch checked={state} onCheckedChange={setState} />
                <Label>{label}</Label>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default BrownianMotion3D;
