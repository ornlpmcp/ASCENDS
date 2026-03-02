// UI lab source (TypeScript)
// NOTE: Browser serves compiled JS from static/js/ui-lab.js in this PoC.

type Status = "Idle" | "Data Previewed" | "Model Trained" | "Prediction Ready";

const demoStatus = document.getElementById("demoStatus") as HTMLSpanElement | null;
const demoBar = document.getElementById("demoBar") as HTMLDivElement | null;
const demoLog = document.getElementById("demoLog") as HTMLUListElement | null;
const btnPreview = document.getElementById("btnPreview") as HTMLButtonElement | null;
const btnTrain = document.getElementById("btnTrain") as HTMLButtonElement | null;
const btnPredict = document.getElementById("btnPredict") as HTMLButtonElement | null;
const btnReset = document.getElementById("btnReset") as HTMLButtonElement | null;

let progress = 0;

function appendLog(text: string): void {
  if (!demoLog) return;
  const li = document.createElement("li");
  li.textContent = text;
  demoLog.prepend(li);
}

function setProgress(next: number, statusText: Status): void {
  progress = Math.max(0, Math.min(100, next));
  if (demoBar) demoBar.style.width = `${progress}%`;
  if (demoStatus) demoStatus.textContent = statusText;
}

btnPreview?.addEventListener("click", () => {
  setProgress(Math.max(progress, 30), "Data Previewed");
  appendLog("Preview step complete.");
});

btnTrain?.addEventListener("click", () => {
  setProgress(Math.max(progress, 70), "Model Trained");
  appendLog("Training step complete.");
});

btnPredict?.addEventListener("click", () => {
  setProgress(100, "Prediction Ready");
  appendLog("Prediction step complete.");
});

btnReset?.addEventListener("click", () => {
  setProgress(0, "Idle");
  if (demoLog) demoLog.innerHTML = "";
  appendLog("Reset.");
});
