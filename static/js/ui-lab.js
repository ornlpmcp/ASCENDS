const demoStatus = document.getElementById("demoStatus");
const demoBar = document.getElementById("demoBar");
const demoLog = document.getElementById("demoLog");
const btnPreview = document.getElementById("btnPreview");
const btnTrain = document.getElementById("btnTrain");
const btnPredict = document.getElementById("btnPredict");
const btnReset = document.getElementById("btnReset");

let progress = 0;

function appendLog(text) {
  if (!demoLog) return;
  const li = document.createElement("li");
  li.textContent = text;
  demoLog.prepend(li);
}

function setProgress(next, statusText) {
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

