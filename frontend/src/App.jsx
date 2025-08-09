// src/App.jsx

import React, { Suspense, useState, useRef } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Environment } from "@react-three/drei";
import { RiggedHand } from "./Hand.jsx";
import { CameraFeed } from "./Camera.jsx";
import { Chat } from "./Chat.jsx";
import { CameraProvider } from "./Websocket.jsx";

export default function App() {
  const [showAnimation, setShowAnimation] = useState(false);
  const [currentText, setCurrentText] = useState("");
  const animationRef = useRef(null);

  const words = ["Rock", "Paper", "Scissors"];

  const handlePlay = () => {
    setShowAnimation(true);
    let i = 0;
    setCurrentText(words[i]);
    animationRef.current = setInterval(() => {
      i = (i + 1) % words.length;
      setCurrentText(words[i]);
    }, 350);
    setTimeout(() => {
      clearInterval(animationRef.current);
      setShowAnimation(false);
      setCurrentText("");
    }, 2100); // 6 cycles
  };

  return (
    <CameraProvider>
      <div className="min-h-screen w-full bg-gray-900">
        <div className="flex flex-col items-center py-8">
          <h1
            className="text-5xl font-extrabold text-blue-400 drop-shadow-lg mb-2 tracking-wide"
            style={{ fontFamily: "Impact, sans-serif", letterSpacing: "0.1em" }}
          >
            Rock Paper Roast
          </h1>
          <p className="text-lg text-gray-300 italic mb-6">
            Play RPS and Get Roasted for no reason
          </p>
        </div>
        <div className="flex flex-col items-center w-full mb-4">
          <button
            className="bg-yellow-400 hover:bg-yellow-500 text-gray-900 font-bold py-2 px-6 rounded-full shadow-lg transition-all duration-200 mb-4"
            onClick={handlePlay}
          >
            Play
          </button>
          {showAnimation && (
            <div className="text-4xl font-extrabold text-pink-400 animate-pulse h-12 flex items-center justify-center">
              {currentText}
            </div>
          )}
        </div>
        <div className="flex justify-center items-center w-full mb-4">
          <CameraFeed />
        </div>
        <Chat />
        <Canvas camera={{ position: [0, 0, 3], fov: 50 }}>
          {/* Lights are essential to see anything */}
          <ambientLight intensity={0.5} />
          <directionalLight position={[1, 2, 3]} intensity={1.5} />

          {/* Using an environment map gives nice reflections and lighting */}
          <Environment preset="city" />

          {/* Suspense is needed to handle the async loading of the model */}
          <Suspense fallback={null}>
            <RiggedHand position={[0, -1, 0]} scale={2} />
          </Suspense>

          <OrbitControls />
        </Canvas>
      </div>
    </CameraProvider>
  );
}
