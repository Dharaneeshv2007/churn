const form = document.getElementById("churnForm");
const resultDiv = document.getElementById("resultCard");
const loading = document.getElementById("loading");

// -------------------
// PREDICT
// -------------------
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  loading.style.display = "block";
  resultDiv.innerHTML = "";

  try {
    const data = Object.fromEntries(new FormData(form));

    // convert numbers
    data.tenure = Number(data.tenure);
    data.MonthlyCharges = Number(data.MonthlyCharges);
    data.TotalCharges = Number(data.TotalCharges);

    const res = await fetch("https://churn-dki6.onrender.com/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(data)
    });

    if (!res.ok) throw new Error("Prediction failed");

    const result = await res.json();
    showResult(result);

  } catch (err) {
    alert("❌ Backend error or not running");
    console.error(err);
  }

  loading.style.display = "none";
});


// -------------------
// CLEAN TOP REASONS (🔥 IMPORTANT FIX)
// -------------------
function cleanReasons(reasons) {
  if (!reasons) return [];

  return reasons.map(r => {

    // Handle Contract properly
    if (r.includes("Contract_Month-to-month"))
      return "Contract: Month-to-month (High churn risk)";

    if (r.includes("Contract_One year"))
      return "Contract: One year (Moderate stability)";

    if (r.includes("Contract_Two year"))
      return "Contract: Two year (Low churn risk)";

    // General cleanup
    return r.replace(/_/g, " : ");
  });
}


// -------------------
// SHOW RESULT
// -------------------
function showResult(data) {

  let percent = Math.round((data.churn_probability || 0) * 100);

  // Color logic
  let color = "green";
  if (percent > 70) color = "red";
  else if (percent > 40) color = "orange";

  const reasonsList = cleanReasons(data.top_reasons);

  let reasonsHTML = reasonsList.map(r => `<li>${r}</li>`).join("");

  resultDiv.innerHTML = `
    <h2>Prediction Result</h2>

    <div class="progress-bar">
      <div class="progress-inner" 
           style="width:${percent}%; background:${color}">
      </div>
    </div>

    <p><b>Churn Probability:</b> ${percent}%</p>
    <p><b>Risk Level:</b> ${data.risk_level}</p>
    <p><b>Time to Churn:</b> ${data.time_to_churn}</p>
    <p><b>Customer Value:</b> ${data.customer_value}</p>
    <p><b>Recommended Action:</b> ${data.recommended_action}</p>

    <h3>Top Reasons</h3>
    <ul>${reasonsHTML}</ul>
  `;
}


// -------------------
// EXPLAIN
// -------------------
document.getElementById("explainBtn").onclick = async () => {

  try {
    const data = Object.fromEntries(new FormData(form));

    const res = await fetch("http://127.0.0.1:5000/explain", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(data)
    });

    if (!res.ok) throw new Error("Explain failed");

    const result = await res.json();

    console.log("🔍 SHAP Values:", result);
    alert("Explanation printed in console");

  } catch (err) {
    alert("❌ Explain failed");
    console.error(err);
  }
};


// -------------------
// TRAIN
// -------------------
document.getElementById("trainBtn").onclick = async () => {

  try {
    await fetch("http://127.0.0.1:5000/train");
    alert("✅ Model Trained Successfully!");
  } catch (err) {
    alert("❌ Training failed");
    console.error(err);
  }
};