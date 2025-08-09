import React, { createContext, useState, useEffect, useRef } from "react";
import { handleFrame } from "./scripts/photoroast";

// Create context
export const CameraContext = createContext(null);

export function CameraProvider({ children }) {
  const [cameraStream, setCameraStream] = useState(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);

  useEffect(() => {
    async function getCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
        });
        setCameraStream(stream);
      } catch (err) {
        setCameraStream(null);
      }
    }
    getCamera();
    return () => {
      if (cameraStream) {
        cameraStream.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  useEffect(() => {
    let intervalId;
    let ws;
    if (cameraStream) {
      // Setup video element for frame capture
      if (!videoRef.current) {
        videoRef.current = document.createElement("video");
        videoRef.current.setAttribute("playsinline", "");
        videoRef.current.setAttribute("muted", "");
        videoRef.current.autoplay = true;
        videoRef.current.width = 400;
        videoRef.current.height = 300;
      }
      videoRef.current.srcObject = cameraStream;
      videoRef.current.play();

      // Setup canvas for frame capture
      if (!canvasRef.current) {
        canvasRef.current = document.createElement("canvas");
        canvasRef.current.width = 400;
        canvasRef.current.height = 300;
      }

      // Setup WebSocket
      ws = new window.WebSocket("ws://localhost:8765");
      wsRef.current = ws;
      ws.onmessage = (event) => {
        console.log("WebSocket response:", event.data);
      };
      ws.onerror = (err) => {
        console.error("WebSocket error:", err);
      };

      // Send frames at intervals
      intervalId = setInterval(() => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (video && canvas && ws && ws.readyState === 1) {
          const ctx = canvas.getContext("2d");
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          const dataUrl = canvas.toDataURL("image/jpeg");
          ws.send(dataUrl);
          handleFrame(dataUrl, 0);
        }
      }, 500);
    }
    return () => {
      if (intervalId) clearInterval(intervalId);
      if (ws) ws.close();
    };
  }, [cameraStream]);

  return (
    <CameraContext.Provider value={{ cameraStream }}>
      {children}
    </CameraContext.Provider>
  );
}
