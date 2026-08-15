const API_URL = "https://churn-dki6.onrender.com";

const form = document.getElementById("churnForm");
const resultDiv = document.getElementById("resultCard");
const loading = document.getElementById("loading");


// ======================================================
// PREDICT
// ======================================================

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  loading.style.display = "block";
  resultDiv.innerHTML = "";

  try {
    // Get form data
    const data = Object.fromEntries(new FormData(form));

    // Convert numeric values
    data.tenure = Number(data.tenure);
    data.MonthlyCharges = Number(data.MonthlyCharges);
    data.TotalCharges = Number(data.TotalCharges);

    console.log("📤 Sending prediction data:", data);

    // Send request to Render backend
    const res = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(data)
    });

    console.log("📥 Prediction response status:", res.status);

    // Read response
    const responseText = await res.text();

    console.log("📥 Backend response:", responseText);

    // Check HTTP status
    if (!res.ok) {
      throw new Error(
        `Backend returned ${res.status}: ${responseText}`
      );
    }

    // Convert response to JSON
    const result = JSON.parse(responseText);

    console.log("✅ Prediction result:", result);

    // Display result
    showResult(result);

  } catch (err) {
    console.error("❌ Prediction error:", err);

    resultDiv.innerHTML = `
      <div class="error-card">
        <h3>❌ Prediction Failed</h3>
        <p>${err.message}</p>
        <p>Please check whether the backend is running on Render.</p>
      </div>
    `;

    alert("❌ Prediction failed. Check the browser Console for details.");

  } finally {
    loading.style.display = "none";
  }
});


// ======================================================
// CLEAN TOP REASONS
// ======================================================

function cleanReasons(reasons) {

  if (!reasons || !Array.isArray(reasons)) {
    return [];
  }

  return reasons.map(r => {

    // Make sure the value is a string
    r = String(r);

    // Contract handling
    if (r.includes("Contract_Month-to-month")) {
      return "Contract: Month-to-month (High churn risk)";
    }

    if (r.includes("Contract_One year")) {
      return "Contract: One year (Moderate stability)";
    }

    if (r.includes("Contract_Two year")) {
      return "Contract: Two year (Low churn risk)";
    }

    // General cleanup
    return r.replace(/_/g, " : ");
  });
}


// ======================================================
// SHOW RESULT
// ======================================================

function showResult(data) {

  console.log("📊 Showing result:", data);

  const probability = Number(data.churn_probability || 0);

  const percent = Math.round(probability * 100);

  // Color logic
  let color = "green";

  if (percent > 70) {
    color = "red";
  } else if (percent > 40) {
    color = "orange";
  }

  // Get top reasons
  const reasonsList = cleanReasons(data.top_reasons);

  let reasonsHTML = "";

  if (reasonsList.length > 0) {

    reasonsHTML = reasonsList
      .map(reason => `<li>${reason}</li>`)
      .join("");

  } else {

    reasonsHTML = "<li>No specific reasons available</li>";
  }


  // Display result
  resultDiv.innerHTML = `
    <div class="prediction-result">

      <h2>Prediction Result</h2>

      <div class="progress-bar">
        <div
          class="progress-inner"
          style="width:${percent}%; background:${color};">
        </div>
      </div>

      <p>
        <b>Churn Probability:</b>
        ${percent}%
      </p>

      <p>
        <b>Risk Level:</b>
        ${data.risk_level || "Unknown"}
      </p>

      <p>
        <b>Time to Churn:</b>
        ${data.time_to_churn || "Not available"}
      </p>

      <p>
        <b>Customer Value:</b>
        ${data.customer_value || "Unknown"}
      </p>

      <p>
        <b>Recommended Action:</b>
        ${data.recommended_action || "No recommendation available"}
      </p>

      <h3>Top Reasons</h3>

      <ul>
        ${reasonsHTML}
      </ul>

    </div>
  `;
}


// ======================================================
// EXPLAIN
// ======================================================

document.getElementById("explainBtn").onclick = async () => {

  try {

    const data = Object.fromEntries(new FormData(form));

    // Convert numbers
    data.tenure = Number(data.tenure);
    data.MonthlyCharges = Number(data.MonthlyCharges);
    data.TotalCharges = Number(data.TotalCharges);

    console.log("📤 Sending explain request:", data);

    const res = await fetch(`${API_URL}/explain`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(data)
    });

    console.log("📥 Explain response status:", res.status);

    const responseText = await res.text();

    console.log("📥 Explain backend response:", responseText);

    if (!res.ok) {
      throw new Error(
        `Explain failed with ${res.status}: ${responseText}`
      );
    }

    const result = JSON.parse(responseText);

    console.log("🔍 SHAP Values:", result);

    alert("✅ Explanation generated successfully! Check the browser Console.");

  } catch (err) {

    console.error("❌ Explain error:", err);

    alert(`❌ Explain failed: ${err.message}`);
  }
};


// ======================================================
// TRAIN
// ======================================================

document.getElementById("trainBtn").onclick = async () => {

  try {

    console.log("📤 Sending training request...");

    const res = await fetch(`${API_URL}/train`, {
      method: "GET"
    });

    console.log("📥 Train response status:", res.status);

    const responseText = await res.text();

    console.log("📥 Train backend response:", responseText);

    if (!res.ok) {
      throw new Error(
        `Training failed with ${res.status}: ${responseText}`
      );
    }

    alert("✅ Model Trained Successfully!");

  } catch (err) {

    console.error("❌ Training error:", err);

    alert(`❌ Training failed: ${err.message}`);
  }
};