// src/App.jsx

import React, { Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Environment } from '@react-three/drei';
import { RiggedHand } from './Hand.jsx';
import {CameraFeed} from './Camera.jsx';

export default function App() {
  return (
    <div className="min-h-screen w-full bg-black">
      <CameraFeed />
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
  );
}