function previewAudio() {
  const fileInput = document.getElementById("audioFile");
  const audio = document.getElementById("audioPreview");

  const file = fileInput.files[0];
  if (!file) return;

  audio.src = URL.createObjectURL(file);
  audio.classList.remove("hidden");
}

async function predict() {
  const file = document.getElementById("audioFile").files[0];
  if (!file) {
    alert("Please select an audio file");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  document.getElementById("loading").classList.remove("hidden");
  document.getElementById("result").classList.add("hidden");

  const response = await fetch("http://127.0.0.1:8000/predict", {
    method: "POST",
    body: formData
  });

  const data = await response.json();

  document.getElementById("loading").classList.add("hidden");

  document.getElementById("genre").innerText =
    `🎧 ${data.predicted_genre.toUpperCase()}`;

  const confidence = Math.round(data.confidence * 100);
  document.getElementById("confidenceBar").style.width = confidence + "%";
  document.getElementById("confidenceText").innerText =
    `Confidence: ${confidence}%`;

  document.getElementById("result").classList.remove("hidden");
}
