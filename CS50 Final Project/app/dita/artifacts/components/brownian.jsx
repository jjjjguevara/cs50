import React, { useState, useEffect, useCallback, useRef } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid } from "recharts";
import { Pause, Play, RefreshCcw } from "@utils/icons";

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

  // ... rest of Knob logic stays the same ...

  return (
    <div className="d-flex flex-column align-items-center gap-2">
      <span className="text-muted small">{label}</span>
      <div
        className="position-relative"
        style={{ width: "4rem", height: "4rem", cursor: "pointer" }}
        onPointerDown={(e) => handlePointerEvents(e, "down")}
        onPointerMove={(e) => handlePointerEvents(e, "move")}
        onPointerUp={(e) => handlePointerEvents(e, "up")}
        onPointerCancel={(e) => handlePointerEvents(e, "up")}
      >
        <div className="position-absolute w-100 h-100 rounded-circle bg-light" />
        <div
          className="position-absolute w-100 h-100 rounded-circle bg-primary"
          style={{
            clipPath: "polygon(50% 50%, 50% 0%, 52% 0%, 52% 50%)",
            transform: `rotate(${getRotation(value)}deg)`,
            transition: isDragging ? "none" : "transform 0.1s",
          }}
        />
        <div
          className="position-absolute rounded-circle bg-white shadow-sm"
          style={{ inset: "30%" }}
        />
      </div>
      {isEditing ? (
        <input
          type="text"
          value={inputValue}
          onChange={(e) => handleEditing(e, "change")}
          onBlur={() => handleEditing(null, "blur")}
          onKeyDown={(e) => handleEditing(e, "key")}
          className="form-control form-control-sm text-center"
          style={{ width: "4rem" }}
          autoFocus
        />
      ) : (
        <span
          className="font-monospace small user-select-none"
          style={{ cursor: "pointer" }}
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

  // ... rest of your state management code ...

  return (
    <div className="card w-100 mx-auto" style={{ maxWidth: "48rem" }}>
      <div className="card-header d-flex justify-content-between align-items-center">
        <h5 className="card-title mb-0">3D Brownian Motion</h5>
        <div className="d-flex gap-2">
          <button
            className="btn btn-outline-primary btn-sm"
            onClick={() => setIsPlaying((prev) => !prev)}
          >
            {isPlaying ? (
              <Pause className="bi" style={{ width: "1rem", height: "1rem" }} />
            ) : (
              <Play className="bi" style={{ width: "1rem", height: "1rem" }} />
            )}
          </button>
          <button
            className="btn btn-outline-primary btn-sm"
            onClick={handleReset}
          >
            <RefreshCcw
              className="bi"
              style={{ width: "1rem", height: "1rem" }}
            />
          </button>
        </div>
      </div>

      <div className="card-body">
        <div className="d-flex flex-column gap-4">
          <div className="position-relative" style={{ aspectRatio: "1/1" }}>
            {/* SVG rendering of Brownian motion */}
          </div>

          {showStatistics && (
            <div className="chart-container">
              <LineChart width={700} height={200} data={statisticsData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" label="Time" />
                <YAxis label="MSD" />
                <Line
                  type="monotone"
                  dataKey="msd"
                  stroke="#0d6efd"
                  dot={false}
                  name="Measured MSD"
                />
                <Line
                  type="monotone"
                  dataKey="theoretical"
                  stroke="#dc3545"
                  strokeDasharray="5 5"
                  dot={false}
                  name="Theoretical MSD"
                />
              </LineChart>
            </div>
          )}

          <div className="d-flex flex-wrap justify-content-center gap-4">
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

          <div className="d-flex justify-content-center gap-4">
            {[
              ["Show Trails", showTrails, setShowTrails],
              ["Show Statistics", showStatistics, setShowStatistics],
              ["Show Axes", showAxes, setShowAxes],
            ].map(([label, state, setState]) => (
              <div key={label} className="form-check form-switch">
                <input
                  className="form-check-input"
                  type="checkbox"
                  checked={state}
                  onChange={(e) => setState(e.target.checked)}
                  id={`switch-${label}`}
                />
                <label className="form-check-label" htmlFor={`switch-${label}`}>
                  {label}
                </label>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default BrownianMotion3D;
