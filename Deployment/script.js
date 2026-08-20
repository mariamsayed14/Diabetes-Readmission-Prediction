// =====================================================
// PAGE NAVIGATION
// =====================================================

function showPrediction() {

    document
        .getElementById("predictionPage")
        .style.display = "block";


    document
        .getElementById("aboutPage")
        .classList.remove("show");

}


function showAbout() {

    document
        .getElementById("predictionPage")
        .style.display = "none";


    document
        .getElementById("aboutPage")
        .classList.add("show");

}


// =====================================================
// GET VALUE
// =====================================================

function getValue(id) {

    return document
        .getElementById(id)
        .value;

}


// =====================================================
// FORM SUBMISSION
// =====================================================

document
    .getElementById("predictionForm")
    .addEventListener(
        "submit",
        async function(event) {

            event.preventDefault();


            const button =
                document.querySelector(
                    ".predict-button"
                );


            button.disabled = true;

            button.innerText =
                "Analyzing Patient...";


            // =========================================
            // COLLECT DATA
            // =========================================

            const patientData = {

                age:
                    getValue("age"),

                gender:
                    getValue("gender"),

                race:
                    getValue("race"),


                time_in_hospital:
                    Number(
                        getValue(
                            "time_in_hospital"
                        )
                    ),

                num_lab_procedures:
                    Number(
                        getValue(
                            "num_lab_procedures"
                        )
                    ),

                num_procedures:
                    Number(
                        getValue(
                            "num_procedures"
                        )
                    ),

                num_medications:
                    Number(
                        getValue(
                            "num_medications"
                        )
                    ),

                number_diagnoses:
                    Number(
                        getValue(
                            "number_diagnoses"
                        )
                    ),

                number_inpatient:
                    Number(
                        getValue(
                            "number_inpatient"
                        )
                    ),

                number_emergency:
                    Number(
                        getValue(
                            "number_emergency"
                        )
                    ),

                number_outpatient:
                    Number(
                        getValue(
                            "number_outpatient"
                        )
                    ),


                max_glu_serum:
                    getValue(
                        "max_glu_serum"
                    ),

                A1Cresult:
                    getValue(
                        "A1Cresult"
                    ),

                change:
                    getValue(
                        "change"
                    ),

                diabetesMed:
                    getValue(
                        "diabetesMed"
                    ),

                insulin:
                    getValue(
                        "insulin"
                    )

            };


            // =========================================
            // SEND TO FASTAPI
            // =========================================

            try {

                const response =
                    await fetch(
                        "http://127.0.0.1:8000/predict",
                    {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json"
                        },
                        body: JSON.stringify(patientData)
                    }
                    );


                // =====================================
                // CHECK RESPONSE
                // =====================================

                if (!response.ok) {

                    throw new Error(
                        "Prediction request failed"
                    );

                }


                const result =
                    await response.json();


                // =====================================
                // DISPLAY RESULT
                // =====================================

                displayResult(
                    result
                );


            }

            catch(error) {

                console.error(
                    error
                );


                alert(
                    "Something went wrong while making the prediction."
                );

            }


            // =========================================
            // RESET BUTTON
            // =========================================

            button.disabled = false;

            button.innerText =
                "Predict Readmission Risk →";

        }
    );


// =====================================================
// DISPLAY RESULT
// =====================================================

function displayResult(result) {


    const resultBox =
        document.getElementById(
            "result"
        );


    const bestModelTag =
        document.getElementById(
            "bestModelTag"
        );


    const percentage =
        document.getElementById(
            "percentage"
        );


    const prediction =
        document.getElementById(
            "prediction"
        );


    const description =
        document.getElementById(
            "description"
        );


    const risk =
        document.getElementById(
            "risk"
        );


    const progressBar =
        document.getElementById(
            "progressBar"
        );


    // =========================================
    // BEST MODEL TAG
    // =========================================

    bestModelTag.innerHTML =
        "Best Model: <strong>" +
        result.best_model +
        "</strong> &middot; Accuracy: <strong>" +
        result.best_accuracy +
        "%</strong>";


    // =========================================
    // PROBABILITY
    // =========================================

    percentage.innerText =
        result.probability + "%";


    // =========================================
    // PREDICTION
    // =========================================

    prediction.innerText =
        result.prediction;


    // =========================================
    // RISK
    // =========================================

    risk.innerText =
        result.risk;


    risk.className =
        "risk";


    if (
        result.risk ===
        "High Risk"
    ) {

        risk.classList.add(
            "high"
        );

        description.innerText =
            "The model estimates a high likelihood of hospital readmission.";

    }


    else if (
        result.risk ===
        "Moderate Risk"
    ) {

        risk.classList.add(
            "moderate"
        );

        description.innerText =
            "The model estimates a moderate likelihood of hospital readmission.";

    }


    else {

        risk.classList.add(
            "low"
        );

        description.innerText =
            "The model estimates a relatively low likelihood of hospital readmission.";

    }


    // =========================================
    // PROGRESS BAR
    // =========================================

    progressBar.style.width =
        result.probability + "%";


    // =========================================
    // ALL MODELS COMPARISON
    // =========================================

    renderModelsGrid(
        result.models,
        result.best_model
    );


    // =========================================
    // SHOW RESULT
    // =========================================

    resultBox.classList.add(
        "show"
    );


    // Scroll to result

    resultBox.scrollIntoView({
        behavior: "smooth"
    });

}


// =====================================================
// RENDER ALL-MODELS COMPARISON GRID
// =====================================================

function renderModelsGrid(models, bestModelName) {

    const grid =
        document.getElementById(
            "modelsGrid"
        );


    grid.innerHTML = "";


    if (!models) {

        return;

    }


    // =========================================
    // RISK -> CSS CLASS
    // =========================================

    function riskClass(risk) {

        if (risk === "High Risk") {

            return "high";

        }

        else if (risk === "Moderate Risk") {

            return "moderate";

        }

        else {

            return "low";

        }

    }


    // =========================================
    // BUILD ONE CARD PER MODEL
    // =========================================

    Object.keys(models).forEach(
        function(modelName) {

            const info =
                models[modelName];


            const isBest =
                modelName === bestModelName;


            const card =
                document.createElement(
                    "div"
                );

            card.className =
                "model-card" +
                (isBest ? " best" : "");


            card.innerHTML = `
                <div class="model-card-header">
                    <div>
                        <span class="model-name">${modelName}</span>
                        ${isBest ? '<span class="best-badge">BEST</span>' : ""}
                    </div>
                    <div class="model-accuracy">
                        Accuracy: <strong>${info.accuracy}%</strong>
                    </div>
                </div>

                <div class="model-prediction-row">
                    <span class="model-prediction-text">${info.prediction}</span>
                    <span class="model-risk ${riskClass(info.risk)}">${info.risk}</span>
                </div>

                <div class="model-probs">

                    <div class="model-prob-row">
                        <span class="model-prob-label">Readmitted</span>
                        <div class="model-prob-track">
                            <div
                                class="model-prob-fill readmit"
                                style="width: ${info.readmitted_percentage}%"
                            ></div>
                        </div>
                        <span class="model-prob-value">${info.readmitted_percentage}%</span>
                    </div>

                    <div class="model-prob-row">
                        <span class="model-prob-label">Not Readmitted</span>
                        <div class="model-prob-track">
                            <div
                                class="model-prob-fill not-readmit"
                                style="width: ${info.not_readmitted_percentage}%"
                            ></div>
                        </div>
                        <span class="model-prob-value">${info.not_readmitted_percentage}%</span>
                    </div>

                </div>
            `;


            grid.appendChild(
                card
            );

        }

    );

}