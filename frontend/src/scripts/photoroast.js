import { GoogleGenerativeAI } from "@google/generative-ai";

const GEMINI_API_KEY = import.meta.env.VITE_GEMINI_API_KEY;
const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);

function base64ToGenerativePart(base64, mimeType) {
  // Remove data URL prefix if present
  const base64Data = base64.split(",")[1] || base64;
  return {
    inlineData: {
      data: base64Data,
      mimeType,
    },
  };
}

export async function handleFrame(frame, n) {
  // frame is expected to be a base64 string (e.g., dataUrl from canvas.toDataURL)
  const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash-latest" });
  const prompt = `You need to reply to this with a roast the context is, you have beaten the person in the image in game of rock paper scissors, and now roast them you can use any physical appearnece or outfit they wear or have make the roast creative surprise, also ${n} is the number of times you have lost to them and roasted them so far so include that in the context `;
  const imagePart = base64ToGenerativePart(frame, "image/jpeg");
  try {
    const result = await model.generateContent([prompt, imagePart]);
    const response = await result.response;
    const text = response.text();
    console.log(text);
  } catch (err) {
    console.error("Gemini image processing error:", err);
  }
}
