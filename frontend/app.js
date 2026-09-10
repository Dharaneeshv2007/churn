const API_URL = "https://churn-dki6.onrender.com";

// ======================================================
// API CONFIGURATION
// ======================================================

const API_TIMEOUT = 120000;


// ======================================================
// API REQUEST HELPER
// ======================================================

async function apiRequest(path, payload = null, timeoutMs = API_TIMEOUT) {

    const controller = new AbortController();

    const timer = setTimeout(() => {
        controller.abort();
    }, timeoutMs);

    const startedAt = performance.now();

    try {

        const options = {
            method: payload === null ? "GET" : "POST",

            headers: {
                "Accept": "application/json"
            },

            signal: controller.signal
        };

        // Add JSON body for POST requests
        if (payload !== null) {

            options.headers["Content-Type"] = "application/json";

            options.body = JSON.stringify(payload);
        }

        console.log(
            `[API] ${options.method} ${path} started`
        );

        const response = await fetch(
            `${API_URL}${path}`,
            options
        );

        const elapsed = Math.round(
            performance.now() - startedAt
        );

        console.log(
            `[API] ${options.method} ${path} response: ${response.status} (${elapsed} ms)`
        );

        const text = await response.text();

        let body = null;

        try {

            body = text ? JSON.parse(text) : null;

        } catch {

            body = null;
        }

        // Handle HTTP errors
        if (!response.ok) {

            const message =
                (body && (body.error || body.message)) ||
                `Backend returned HTTP ${response.status}`;

            const error = new Error(message);

            error.status = response.status;
            error.responseBody = body;

            throw error;
        }

        return body;

    } catch (error) {

        // Request timeout
        if (error.name === "AbortError") {

            const timeoutError = new Error(
                `${path} timed out after ${Math.round(timeoutMs / 1000)} seconds.`
            );

            timeoutError.code = "TIMEOUT";

            throw timeoutError;
        }

        // Network error
        if (error instanceof TypeError) {

            const networkError = new Error(
                `Could not reach ${API_URL}${path}. Check the Render backend and network connection.`
            );

            networkError.code = "NETWORK_ERROR";

            throw networkError;
        }

        throw error;

    } finally {

        clearTimeout(timer);
    }
}


// ======================================================
// BACKEND WARM-UP
// ======================================================

let backendReadyPromise = null;

function warmBackend() {

    // Avoid sending multiple health requests at the same time
    if (backendReadyPromise) {
        return backendReadyPromise;
    }

    backendReadyPromise = (async () => {

        try {

            console.log(
                "[Backend] Checking backend health..."
            );

            const result = await apiRequest(
                "/health",
                null,
                API_TIMEOUT
            );

            console.log(
                "[Backend] Backend is ready:",
                result
            );

            return true;

        } catch (error) {

            console.warn(
                "[Backend] Health check failed:",
                error.message
            );

            return false;
        }

    })();

    return backendReadyPromise;
}


// Start backend warm-up when page loads
warmBackend();


// ======================================================
// DOM ELEMENTS
// ======================================================

const form = document.getElementById("churnForm");

const resultDiv = document.getElementById("resultCard");

const explanationDiv =
    document.getElementById("explanationCard");

const loading =
    document.getElementById("loading");

const loadingText =
    document.getElementById("loadingText");

const predictBtn =
    document.getElementById("predictBtn");

const explainBtn =
    document.getElementById("explainBtn");


// ======================================================
// INITIAL STATE
// ======================================================

initStates();


// ======================================================
// PREDICT
// ======================================================

form.addEventListener("submit", async (e) => {

    e.preventDefault();

    if (!form.reportValidity()) {
        return;
    }

    setLoading(
        true,
        "Connecting to prediction server...",
        predictBtn
    );

    try {

        // --------------------------------------------------
        // STEP 1
        // Check that Render is awake
        // --------------------------------------------------

        await warmBackend();

        setLoading(
            true,
            "Analyzing customer data...",
            predictBtn
        );

        // --------------------------------------------------
        // STEP 2
        // Collect form data
        // --------------------------------------------------

        const data = collectFormData();

        console.log(
            "[Predict] Sending prediction data:",
            data
        );

        // --------------------------------------------------
        // STEP 3
        // Send prediction request
        // --------------------------------------------------

        const result = await apiRequest(
            "/predict",
            data,
            API_TIMEOUT
        );

        console.log(
            "[Predict] Prediction result:",
            result
        );

        // --------------------------------------------------
        // STEP 4
        // Display result
        // --------------------------------------------------

        showResult(result);

    } catch (err) {

        console.error(
            "[Predict] Prediction error:",
            err
        );

        let message =
            err.message ||
            "Unexpected prediction error.";

        if (err.code === "TIMEOUT") {

            message =
                "The prediction request timed out. " +
                "The Render server is reachable, but /predict did not finish within 120 seconds.";

        } else if (err.code === "NETWORK_ERROR") {

            message =
                "The browser could not reach the Render backend. " +
                "Please check that the Render service is running.";

        } else if (err.status) {

            message =
                `Backend error (HTTP ${err.status}): ${message}`;
        }

        showError(
            resultDiv,
            "Prediction Failed",
            message
        );

    } finally {

        setLoading(false);
    }
});


// ======================================================
// CLEAN TOP REASONS
// ======================================================

function cleanReasons(reasons) {

    if (
        !reasons ||
        !Array.isArray(reasons)
    ) {

        return [];
    }

    return reasons.map((r) => {

        r = String(r);

        if (
            r.includes(
                "Contract_Month-to-month"
            )
        ) {

            return (
                "Contract: Month-to-month (High churn risk)"
            );
        }

        if (
            r.includes(
                "Contract_One year"
            )
        ) {

            return (
                "Contract: One year (Moderate stability)"
            );
        }

        if (
            r.includes(
                "Contract_Two year"
            )
        ) {

            return (
                "Contract: Two year (Low churn risk)"
            );
        }

        return r.replace(
            /_/g,
            " "
        );
    });
}


// ======================================================
// SHOW RESULT
// ======================================================

function showResult(data) {

    console.log(
        "Showing prediction result:",
        data
    );

    const probability =
        normalizeProbability(
            data.churn_probability
        );

    const percent =
        Math.round(
            probability * 1000
        ) / 10;

    const levelInfo =
        getRiskInfo(percent);

    const reasonsList =
        cleanReasons(
            data.top_reasons
        );

    const reasonsHTML =
        reasonsList.length > 0

            ? reasonsList
                .map(
                    (reason) =>
                        `<li>${escapeHTML(reason)}</li>`
                )
                .join("")

            : `
                <li class="muted">
                    Click "Explain Prediction"
                    to see the main churn factors.
                </li>
              `;

    const recommendation =
        data.recommended_action ||
        data.recommendation ||
        "No recommendation available";

    const riskLabel =
        data.risk_level ||
        levelInfo.label;

    resultDiv.innerHTML = `

        <div class="panel-header">

            <h2>
                Churn Risk Result
            </h2>

            <p>
                Probability and retention
                insights from the latest prediction.
            </p>

        </div>

        <div class="result-shell">

            <div class="risk-meter">

                <div class="radial-wrap">

                    <div
                        class="radial"
                        id="riskRadial"
                        style="
                            --progress: 0;
                            --risk-color: ${levelInfo.color};
                        "
                    >

                        <div
                            class="radial-value"
                            id="riskPercentText"
                        >
                            0.0%
                        </div>

                    </div>

                </div>

                <div>

                    <p class="risk-level">
                        RISK LEVEL
                    </p>

                    <span
                        class="risk-badge ${levelInfo.className}"
                    >
                        ${escapeHTML(
                            String(
                                riskLabel
                            ).toUpperCase()
                        )}
                        RISK
                    </span>

                    <p class="muted">
                        ${escapeHTML(
                            levelInfo.caption
                        )}
                    </p>

                </div>

            </div>

            <div class="result-meta">

                <div class="meta-item">

                    <h4>
                        Time to Churn
                    </h4>

                    <p>
                        ${escapeHTML(
                            data.time_to_churn ||
                            "Not available"
                        )}
                    </p>

                </div>

                <div class="meta-item">

                    <h4>
                        Customer Value
                    </h4>

                    <p>
                        ${escapeHTML(
                            String(
                                data.customer_value ??
                                "Unknown"
                            )
                        )}
                    </p>

                </div>

                <div class="meta-item field-wide">

                    <h4>
                        Recommended Action
                    </h4>

                    <p>
                        ${escapeHTML(
                            recommendation
                        )}
                    </p>

                </div>

            </div>

            <h3>
                Top Reasons
            </h3>

            <ul class="reasons-list">

                ${reasonsHTML}

            </ul>

        </div>
    `;

    animateProbability(
        percent,
        900
    );

    // /predict does not generate SHAP.
    // SHAP is generated only by /explain.
}


// ======================================================
// EXPLAIN
// ======================================================

explainBtn.onclick = async () => {

    if (!form.reportValidity()) {
        return;
    }

    setLoading(
        true,
        "Generating SHAP explanation...",
        explainBtn
    );

    try {

        const data =
            collectFormData();

        console.log(
            "[Explain] Sending explain request:",
            data
        );

        const result =
            await apiRequest(
                "/explain",
                data,
                API_TIMEOUT
            );

        console.log(
            "[Explain] Explanation payload:",
            result
        );

        showExplanation(result);

    } catch (err) {

        console.error(
            "[Explain] Explanation error:",
            err
        );

        let message =
            err.message ||
            "Unexpected explanation error.";

        if (
            err.code === "TIMEOUT"
        ) {

            message =
                "The SHAP explanation timed out after 120 seconds.";

        } else if (
            err.code === "NETWORK_ERROR"
        ) {

            message =
                "The browser could not reach the Render backend.";

        } else if (
            err.status
        ) {

            message =
                `Backend error (HTTP ${err.status}): ${message}`;
        }

        showError(
            explanationDiv,
            "Explanation Failed",
            message
        );

    } finally {

        // Do not use clearTimeout(timeout) here.
        // apiRequest() manages its own timeout.
        setLoading(false);
    }
};


// ======================================================
// INITIAL STATES
// ======================================================

function initStates() {

    loading.hidden = true;

    explanationDiv.innerHTML = `

        <div class="panel-header">

            <h2>
                Prediction Explanation
            </h2>

            <p>
                Run Explain Prediction
                to view model reasoning.
            </p>

        </div>

    `;
}


// ======================================================
// LOADING STATE
// ======================================================

function setLoading(
    isLoading,
    message = "",
    sourceButton = null
) {

    loading.hidden =
        !isLoading;

    if (
        isLoading &&
        message
    ) {

        loadingText.textContent =
            message;
    }

    predictBtn.disabled =
        isLoading;

    explainBtn.disabled =
        isLoading;

    [
        predictBtn,
        explainBtn
    ].forEach(
        (button) => {

            button.classList.remove(
                "is-loading"
            );

        }
    );

    if (
        isLoading &&
        sourceButton
    ) {

        sourceButton.classList.add(
            "is-loading"
        );
    }
}


// ======================================================
// COLLECT FORM DATA
// ======================================================

function collectFormData() {

    const data =
        Object.fromEntries(
            new FormData(form)
        );

    data.tenure =
        Number(
            data.tenure
        );

    data.MonthlyCharges =
        Number(
            data.MonthlyCharges
        );

    data.TotalCharges =
        Number(
            data.TotalCharges
        );

    return data;
}


// ======================================================
// NORMALIZE PROBABILITY
// ======================================================

function normalizeProbability(
    rawProbability
) {

    const numeric =
        Number(
            rawProbability || 0
        );

    if (
        Number.isNaN(
            numeric
        )
    ) {

        return 0;
    }

    if (
        numeric > 1
    ) {

        return Math.min(
            Math.max(
                numeric / 100,
                0
            ),
            1
        );
    }

    return Math.min(
        Math.max(
            numeric,
            0
        ),
        1
    );
}


// ======================================================
// RISK INFORMATION
// ======================================================

function getRiskInfo(
    percent
) {

    if (
        percent < 40
    ) {

        return {

            label:
                "Low",

            className:
                "risk-low",

            color:
                "#0f766e",

            caption:
                "Customer is less likely to churn"
        };
    }

    if (
        percent < 75
    ) {

        return {

            label:
                "Medium",

            className:
                "risk-medium",

            color:
                "#b45309",

            caption:
                "Customer has moderate churn probability"
        };
    }

    return {

        label:
            "High",

        className:
            "risk-high",

        color:
            "#b91c1c",

        caption:
            "Customer is likely to churn"
    };
}


// ======================================================
// ANIMATE PROBABILITY
// ======================================================

function animateProbability(
    targetPercent,
    duration
) {

    const radial =
        document.getElementById(
            "riskRadial"
        );

    const label =
        document.getElementById(
            "riskPercentText"
        );

    if (
        !radial ||
        !label
    ) {

        return;
    }

    const clampedTarget =
        Math.min(
            Math.max(
                targetPercent,
                0
            ),
            100
        );

    const start =
        performance.now();

    const tick =
        (timestamp) => {

            const elapsed =
                timestamp -
                start;

            const progress =
                Math.min(
                    elapsed / duration,
                    1
                );

            const eased =
                1 -
                Math.pow(
                    1 - progress,
                    3
                );

            const current =
                clampedTarget *
                eased;

            radial.style.setProperty(
                "--progress",
                current.toFixed(1)
            );

            label.textContent =
                `${current.toFixed(1)}%`;

            if (
                progress < 1
            ) {

                requestAnimationFrame(
                    tick
                );
            }
        };

    requestAnimationFrame(
        tick
    );
}


// ======================================================
// SHOW EXPLANATION
// ======================================================

function showExplanation(
    result
) {

    const explanation =
        result.prediction_explanation ||
        result;

    const positiveFactors =
        Array.isArray(
            explanation.positive_factors
        )

            ? explanation.positive_factors

            : [];

    const negativeFactors =
        Array.isArray(
            explanation.negative_factors
        )

            ? explanation.negative_factors

            : [];

    const neutralFactors =
        Array.isArray(
            explanation.neutral_factors
        )

            ? explanation.neutral_factors

            : [];

    const features =
        Array.isArray(
            explanation.features
        )

            ? explanation.features

            : [];

    const summary =
        explanation.summary ||
        result.message ||
        "Explanation generated successfully.";

    const finalReason =
        explanation.final_reason ||
        "No detailed final reason provided.";

    explanationDiv.innerHTML = `

        <div class="panel-header">

            <h2>
                Prediction Explanation
            </h2>

            <p>
                ${escapeHTML(
                    summary
                )}
            </p>

        </div>

        <div
            class="explanation-body result-shell"
        >

            <div class="explain-block">

                <h4>
                    Final Reason
                </h4>

                <p>
                    ${escapeHTML(
                        finalReason
                    )}
                </p>

            </div>

            <div class="explain-block">

                <h4>
                    Factors Increasing Churn Risk
                </h4>

                ${renderFactorList(
                    positiveFactors,
                    "chip-positive",
                    "No high-risk contributing factors found"
                )}

            </div>

            <div class="explain-block">

                <h4>
                    Factors Reducing Churn Risk
                </h4>

                ${renderFactorList(
                    negativeFactors,
                    "chip-negative",
                    "No stability factors found"
                )}

            </div>

            <div class="explain-block">

                <h4>
                    Minimal Influence Factors
                </h4>

                ${renderFactorList(
                    neutralFactors,
                    "chip-neutral",
                    "No neutral factors found"
                )}

            </div>

            <div class="explain-block">

                <h4>
                    SHAP Feature Contributions
                </h4>

                ${renderFeatureRows(
                    features
                )}

            </div>

        </div>

    `;
}


// ======================================================
// RENDER FACTOR LIST
// ======================================================

function renderFactorList(
    items,
    className,
    fallbackText
) {

    if (
        !items.length
    ) {

        return `

            <p class="muted">
                ${escapeHTML(
                    fallbackText
                )}
            </p>

        `;
    }

    return `

        <div class="chip-list">

            ${items
                .map(
                    (item) => {

                        const feature =
                            item.feature ||
                            "Feature";

                        const value =
                            item.value ||
                            "";

                        const impact =
                            item.impact ||
                            "";

                        return `

                            <span
                                class="chip ${className}"
                            >

                                ${escapeHTML(
                                    feature
                                )}

                                ${
                                    value
                                        ? `: ${escapeHTML(
                                            String(
                                                value
                                            )
                                        )}`
                                        : ""
                                }

                                ${
                                    impact
                                        ? ` (${escapeHTML(
                                            String(
                                                impact
                                            )
                                        )})`
                                        : ""
                                }

                            </span>

                        `;
                    }
                )
                .join("")}

        </div>

    `;
}


// ======================================================
// RENDER SHAP FEATURE ROWS
// ======================================================

function renderFeatureRows(
    features
) {

    if (
        !features.length
    ) {

        return `

            <p class="muted">
                No SHAP feature values returned.
            </p>

        `;
    }

    const rows =
        features
            .slice(
                0,
                9
            )
            .map(
                (entry) => {

                    const name =
                        escapeHTML(
                            String(
                                entry.feature ||
                                "Feature"
                            )
                        );

                    const value =
                        escapeHTML(
                            String(
                                entry.value ??
                                "Unknown"
                            )
                        );

                    const shapValue =
                        Number(
                            entry.shap_value ||
                            0
                        );

                    const contribution =
                        Number(
                            entry.contribution_percentage_points ||
                            0
                        );

                    const formatted =
                        Number.isFinite(
                            shapValue
                        )

                            ? shapValue.toFixed(
                                4
                            )

                            : "N/A";

                    const contributionText =
                        Number.isFinite(
                            contribution
                        )

                            ? `${contribution.toFixed(
                                2
                            )} pp`

                            : "N/A";

                    const impact =
                        escapeHTML(
                            String(
                                entry.impact ||
                                "Unknown"
                            )
                        );

                    return `

                        <div class="json-row">

                            <strong>
                                ${name}
                            </strong>

                            <span>
                                ${value}
                            </span>

                            <span>
                                SHAP:
                                ${formatted}
                            </span>

                            <span>
                                ${contributionText}
                            </span>

                            <span>
                                ${impact}
                            </span>

                        </div>

                    `;
                }
            )
            .join("");

    return `

        <div class="json-grid">

            ${rows}

        </div>

    `;
}


// ======================================================
// SHOW ERROR
// ======================================================

function showError(
    targetElement,
    title,
    message
) {

    targetElement.innerHTML = `

        <div class="panel-header">

            <h2>
                ${escapeHTML(
                    title
                )}
            </h2>

            <p>
                Please try again after
                checking the backend status.
            </p>

        </div>

        <div class="error-card">

            <p>
                ${escapeHTML(
                    message ||
                    "Unexpected error"
                )}
            </p>

        </div>

    `;
}


// ======================================================
// ESCAPE HTML
// ======================================================

function escapeHTML(
    value
) {

    return String(
        value
    )

        .replace(
            /&/g,
            "&amp;"
        )

        .replace(
            /</g,
            "&lt;"
        )

        .replace(
            />/g,
            "&gt;"
        )

        .replace(
            /"/g,
            "&quot;"
        )

        .replace(
            /'/g,
            "&#39;"
        );
}