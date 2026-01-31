async function loadMetrics() {
  const res = await fetch("http://127.0.0.1:8000/metrics");
  const data = await res.json();

  document.getElementById("accuracyText").innerHTML =
    `<strong>Overall Accuracy:</strong> ${(data.accuracy * 100).toFixed(2)}%`;

  const labels = data.labels;
  const matrix = data.confusion_matrix;

  let maxVal = Math.max(...matrix.flat());

  let html = "<table style='border-collapse:collapse;width:100%;font-size:11px'>";
  html += "<tr><th></th>" + labels.map(l => `<th>${l}</th>`).join("") + "</tr>";

  for (let i = 0; i < matrix.length; i++) {
    html += `<tr><th>${labels[i]}</th>`;
    for (let j = 0; j < matrix[i].length; j++) {
      const val = matrix[i][j];
      const intensity = val / maxVal;
      const color = `rgba(76, 175, 80, ${intensity})`;
      html += `<td style="background:${color};text-align:center;padding:4px;">${val}</td>`;
    }
    html += "</tr>";
  }

  html += "</table>";
  document.getElementById("confusionContainer").innerHTML = html;
}

loadMetrics();
