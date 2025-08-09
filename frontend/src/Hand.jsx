// src/components/RiggedHand.jsx

import React, { useRef, useEffect } from 'react';
import { useGLTF, useAnimations } from '@react-three/drei';

export function RiggedHand(props) {
  const group = useRef();

  // 1. Load the GLB file from the public folder
  const { scene, animations } = useGLTF('/public/rigged_lowpoly_hand.glb'); // <-- IMPORTANT: UPDATE FILENAME

  // 2. The useAnimations hook gives you access to the animation actions
  const { actions } = useAnimations(animations, group);
  
  // 3. This effect will run once, when the component mounts
  useEffect(() => {
    // --- THIS IS THE MOST IMPORTANT PART ---
    // Log all the animation names to the console
    console.log("Available animations:", Object.keys(actions));
    
    // Play a specific animation by name
    // You must replace "Hand_Animation_Name" with one of the names you see in the console
    if (actions["Hand_Animation_Name"]) {
      actions["Hand_Animation_Name"].play();
    }

  }, [actions]);

  // The <primitive> object is a way to render a complex, pre-made scene
  return <primitive ref={group} object={scene} {...props} />;
}

// Pre-loading the model is good practice for performance
useGLTF.preload('/low-poly-hand.glb'); // <-- UPDATE FILENAME