/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      keyframes: {
        scanY: { "0%, 100%": { top: "0" }, "50%": { top: "100%" } },
        scanX: { "0%, 100%": { left: "0" }, "50%": { left: "100%" } },
      },
      animation: {
        scanY: "scanY 3s linear infinite",
        scanX: "scanX 4s linear infinite",
      },
    },
  },
  plugins: [],
};
