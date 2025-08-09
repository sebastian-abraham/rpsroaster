import React, { useRef, useEffect } from "react";
import { handleFrame } from "./scripts/photoroast";

const Websocket = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);

  useEffect(() => {
    let stream;
    let intervalId;
    let ws;

    // Setup webcam
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices
        .getUserMedia({ video: true })
        .then((s) => {
          stream = s;
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
          }
        })
        .catch((err) => {
          console.error("Error accessing webcam:", err);
        });
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
      }
    }, 500); // Send every 500ms

    return () => {
      if (intervalId) clearInterval(intervalId);
      if (ws) ws.close();
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  const processCurrentFrame = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (video && canvas) {
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL("image/jpeg");
      handleFrame(dataUrl, 2);
    }
  };

  return (
    <div>
      <div>Websocket Component</div>
      <div>
        <h3>Webcam Preview</h3>
        <video
          ref={videoRef}
          autoPlay
          style={{ width: "100%", maxWidth: "400px" }}
          width={400}
          height={300}
        />
        <canvas
          ref={canvasRef}
          width={400}
          height={300}
          style={{ display: "none" }}
        />
        <button onClick={processCurrentFrame}>Process Current Frame</button>
      </div>
    </div>
  );
};

export default Websocket;
