const API_URL = "https://churn-dki6.onrender.com";

// ======================================================
// API REQUEST HELPER
// ======================================================
// Handles Render cold starts, timeouts and non-JSON errors
// without turning a backend failure into a confusing CORS message.
async function apiRequest(path, payload = null, timeoutMs = 90000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    try {
        const options = {
            method: payload === null ? "GET" : "POST",
            headers: {
                "Accept": "application/json"
            },
            signal: controller.signal
        };

        if (payload !== null) {
            options.headers["Content-Type"] = "application/json";
            options.body = JSON.stringify(payload);
        }

        const response = await fetch(`${API_URL}${path}`, options);
        const text = await response.text();

        let body = null;
        try {
            body = text ? JSON.parse(text) : null;
        } catch {
            body = null;
        }

        if (!response.ok) {
            const message =
                (body && body.error) ||
                `Backend returned HTTP ${response.status}`;
            const error = new Error(message);
            error.status = response.status;
            throw error;
        }

        return body;
    } catch (error) {
        if (error.name === "AbortError") {
            throw new Error(
                "The backend is taking too long. Render may be waking up; please try again."
            );
        }
        if (error instanceof TypeError) {
            throw new Error(
                "Could not reach the prediction server. Please check that the Render service is running."
            );
        }
        throw error;
    } finally {
        clearTimeout(timer);
    }
}

// Wake the Render service when the page opens.
// This removes most of the cold-start delay from the first prediction.
async function warmBackend() {
    try {
        await apiRequest("/health", null, 90000);
        console.log("Backend is ready.");
    } catch (error) {
        console.warn("Backend warm-up:", error.message);
    }
}

warmBackend();


const form = document.getElementById("churnForm");

const resultDiv = document.getElementById("resultCard");

const explanationDiv = document.getElementById("explanationCard");

const loading = document.getElementById("loading");

const loadingText = document.getElementById("loadingText");

const predictBtn = document.getElementById("predictBtn");

const explainBtn = document.getElementById("explainBtn");


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
        "Analyzing customer data...",
        predictBtn
    );

    try {

        const data = collectFormData();

        console.log(
            "Sending prediction data:",
            data
        );

        const result = await apiRequest(
            "/predict",
            data,
            90000
        );

        console.log(
            "Prediction result:",
            result
        );

        showResult(result);

    } catch (err) {

        console.error(
            "Prediction error:",
            err
        );

        showError(
            resultDiv,
            "Prediction Failed",
            `${err.message}. Please verify that the backend service is reachable.`
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


    // IMPORTANT:
    //
    // /predict no longer generates SHAP.
    //
    // Therefore do NOT automatically call
    // showExplanation() here.
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
            "Sending explain request:",
            data
        );


        const result = await apiRequest(
            "/explain",
            data,
            90000
        );


        console.log(
            "Explanation payload:",
            result
        );


        showExplanation(
            result
        );


    } catch (err) {

        console.error(
            "Explain error:",
            err
        );


        showError(
            explanationDiv,
            "Explanation Failed",
            err.message
        );


    } finally {

        clearTimeout(
            timeout
        );

        setLoading(
            false
        );
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
                timestamp - start;


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

                                ${value
                                    ? `: ${escapeHTML(
                                        String(value)
                                    )}`
                                    : ""
                                }

                                ${impact
                                    ? ` (${escapeHTML(
                                        String(impact)
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
            .slice(0, 9)
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

                            ? shapValue.toFixed(4)

                            : "N/A";


                    const contributionText =
                        Number.isFinite(
                            contribution
                        )

                            ? `${contribution.toFixed(2)} pp`

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

    return String(value)

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