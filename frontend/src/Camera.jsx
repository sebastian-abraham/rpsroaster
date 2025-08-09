import React, { useContext, useRef, useEffect } from "react";
import { CameraContext } from "./Websocket.jsx";

export const CameraFeed = () => {
  const { cameraStream } = useContext(CameraContext);
  const videoRef = useRef(null);

  useEffect(() => {
    if (videoRef.current && cameraStream) {
      videoRef.current.srcObject = cameraStream;
    }
  }, [cameraStream]);

  return (
    <div className="absolute top-70 left-[400px] right-30 w-[700px] h-100 bg-black rounded-lg overflow-hidden border-2 border-green-500">
      <div className="flex justify-center items-center w-full h-full">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="h-full w-auto rounded shadow-lg"
          style={{ background: "#222" }}
        />
      </div>
    </div>
  );
};
