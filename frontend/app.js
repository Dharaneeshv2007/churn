const API_URL = "https://churn-dki6.onrender.com";

const form = document.getElementById("churnForm");

const resultDiv = document.getElementById("resultCard");
const explanationDiv = document.getElementById("explanationCard");

const loading = document.getElementById("loading");
const loadingText = document.getElementById("loadingText");

const predictBtn = document.getElementById("predictBtn");
const explainBtn = document.getElementById("explainBtn");

initStates();


// ======================================================
// PREDICT
// IMPORTANT:
// /predict ONLY performs the ML prediction.
// It does NOT request SHAP.
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

        console.log("Sending prediction data:", data);

        const res = await fetch(`${API_URL}/predict`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        });

        console.log(
            "Prediction response status:",
            res.status
        );

        const responseText = await res.text();

        console.log(
            "Prediction backend response:",
            responseText
        );

        if (!res.ok) {

            throw new Error(
                `Backend returned ${res.status}: ${responseText}`
            );
        }

        const result = JSON.parse(responseText);

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

    if (!reasons || !Array.isArray(reasons)) {
        return [];
    }

    return reasons.map((r) => {

        let reason = String(r);

        if (
            reason.includes(
                "Contract_Month-to-month"
            )
        ) {
            return "Contract: Month-to-month (High churn risk)";
        }

        if (
            reason.includes(
                "Contract_One year"
            )
        ) {
            return "Contract: One year (Moderate stability)";
        }

        if (
            reason.includes(
                "Contract_Two year"
            )
        ) {
            return "Contract: Two year (Low churn risk)";
        }

        return reason.replace(/_/g, " ");

    });

}


// ======================================================
// SHOW PREDICTION RESULT
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

            : "<li>No specific reasons available</li>";


    const recommendation =
        data.recommended_action ||
        data.recommendation ||
        "No recommendation available";


    const riskLabel =
        data.risk_level ||
        levelInfo.label;


    resultDiv.innerHTML = `

        <div class="panel-header">

            <h2>Churn Risk Result</h2>

            <p>
                Probability and retention insights
                from the latest prediction.
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
    // Prediction no longer contains SHAP explanation.
    // Therefore we DO NOT call showExplanation() here.

}


// ======================================================
// EXPLAIN
// IMPORTANT:
// SHAP is ONLY generated here.
// ======================================================

explainBtn.onclick = async () => {

    if (!form.reportValidity()) {
        return;
    }


    setLoading(
        true,
        "Generating prediction explanation...",
        explainBtn
    );


    try {

        const data =
            collectFormData();


        console.log(
            "Sending explain request:",
            data
        );


        const res =
            await fetch(
                `${API_URL}/explain`,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify(data)
                }
            );


        console.log(
            "Explain response status:",
            res.status
        );


        const responseText =
            await res.text();


        console.log(
            "Explain backend response:",
            responseText
        );


        if (!res.ok) {

            throw new Error(
                `Explain failed with ${res.status}: ${responseText}`
            );

        }


        const result =
            JSON.parse(
                responseText
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

        setLoading(false);

    }

};


// ======================================================
// INITIAL STATE
// ======================================================

function initStates() {

    loading.hidden = true;


    explanationDiv.innerHTML = `

        <div class="panel-header">

            <h2>
                Prediction Explanation
            </h2>

            <p>
                Run Explain Prediction to view
                model reasoning.
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


    data.SeniorCitizen =
        Number(
            data.SeniorCitizen
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
        Number.isNaN(numeric)
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
        percent <= 30
    ) {

        return {

            label: "Low",

            className:
                "risk-low",

            color:
                "#0f766e",

            caption:
                "Customer is less likely to churn"

        };

    }


    if (
        percent <= 60
    ) {

        return {

            label: "Medium",

            className:
                "risk-medium",

            color:
                "#b45309",

            caption:
                "Customer has moderate churn probability"

        };

    }


    return {

        label: "High",

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
                ${escapeHTML(summary)}
            </p>

        </div>


        <div class="explanation-body result-shell">


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
                    SHAP Feature Contributions
                </h4>

                ${renderShapRows(
                    explanation.features
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
        !items ||
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

                        // Backend returns objects.
                        // Convert them into useful text.

                        if (
                            typeof item ===
                            "object" &&
                            item !== null
                        ) {

                            const feature =
                                item.feature ||
                                "Feature";

                            const value =
                                item.value ||
                                "Unknown";

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
                                    :
                                    ${escapeHTML(
                                        String(value)
                                    )}
                                    ${impact
                                        ? `(${escapeHTML(
                                            impact
                                          )})`
                                        : ""}
                                </span>

                            `;

                        }


                        return `

                            <span
                                class="chip ${className}"
                            >
                                ${escapeHTML(
                                    String(item)
                                )}
                            </span>

                        `;

                    }
                )
                .join("")}

        </div>

    `;

}


// ======================================================
// RENDER SHAP ROWS
// ======================================================

function renderShapRows(
    features
) {

    if (
        !Array.isArray(features) ||
        !features.length
    ) {

        return `
            <p class="muted">
                No SHAP values returned.
            </p>
        `;

    }


    const rows =
        features
            .slice(0, 8)
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


                    const formatted =
                        Number.isFinite(
                            shapValue
                        )
                            ? shapValue.toFixed(4)
                            : "N/A";


                    const impact =
                        escapeHTML(
                            String(
                                entry.impact ||
                                ""
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
// ERROR DISPLAY
// ======================================================

function showError(
    targetElement,
    title,
    message
) {

    targetElement.innerHTML = `

        <div class="panel-header">

            <h2>
                ${escapeHTML(title)}
            </h2>

            <p>
                Please try again after checking
                the backend status.
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
// HTML ESCAPE
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