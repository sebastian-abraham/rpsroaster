import { useState, useEffect, useRef } from 'react';

const responses = [
  "U noob",
  "U know nothing",
  "just stop",
  "u stuppid",
  "This is a much longer sentence to test how the chat container expands its width dynamically based on the message length.",
  "Another message to test scrolling.",
  "Keep going, more messages!",
  "Scroll to see older messages.",
];

export const Chat = () => {
  const [currentIdx, setCurrentIdx] = useState(0);
  const [containerWidth, setContainerWidth] = useState("w-[220px]");
  const measureRef = useRef(null);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentIdx((prev) => (prev + 1) % responses.length);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  // Measure text width before rendering
  useEffect(() => {
    if (measureRef.current) {
      const textWidth = measureRef.current.offsetWidth;
      if (textWidth > 340) {
        setContainerWidth("w-[400px]");
      } else {
        setContainerWidth("w-[220px]");
      }
    }
  }, [currentIdx]);

  const currentMessage = responses[currentIdx];

  return (
    <>
      {/* Hidden element for measuring text width */}
      <div
        ref={measureRef}
        className="fixed left-[-9999px] top-[-9999px] px-4 py-2 font-bold"
        style={{ maxWidth: "95%", wordBreak: "break-word" }}
      >
        Bot: <span className="font-normal">{currentMessage}</span>
      </div>
      <div className={`absolute bottom-[110px] left-[850px] ${containerWidth} h-[80px] bg-gray-900 rounded-lg p-4 flex items-center z-50 overflow-hidden transition-all duration-700`}>
        <div
          className="bg-gray-700 px-4 py-2 rounded-lg shadow text-white font-bold w-full transition-all duration-300"
          style={{
            maxWidth: "95%",
            wordBreak: "break-word",
          }}
        >
          Bot: <span className="font-normal">{currentMessage}</span>
        </div>
      </div>
    </>
  );
};
