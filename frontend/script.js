// FraudShield AI — Shared frontend logic

const API_URL = "https://creditcard-fraud-detection-2q9i.onrender.com/";
const STORAGE_KEY = "fraudshield_transaction_history";

// Get saved transaction history
function readHistory() {
    try {
        const data = JSON.parse(
            localStorage.getItem(STORAGE_KEY) || "[]"
        );

        return Array.isArray(data) ? data : [];
    } catch (error) {
        return [];
    }
}

// Save the latest 10 transactions
function saveHistory(items) {
    localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify(items.slice(0, 10))
    );
}

// Identify fraudulent transactions
function isFraud(data) {
    return (
        String(data.result || "").toUpperCase() === "FRAUDULENT" ||
        Number(data.prediction) === 1
    );
}

// Format probability
function formatProbability(value) {
    const number = Number(value);

    return Number.isFinite(number)
        ? `${number.toFixed(2)}%`
        : "N/A";
}

// Display form messages
function showMessage(message, type = "info") {
    const node = document.getElementById("form-message");

    if (!node) return;

    node.textContent = message;
    node.className = `form-message ${type}`;
}

// Read and validate transaction features
function getFeatures() {
    const input = document.getElementById("transaction-features");

    if (!input || !input.value.trim()) {
        throw new Error("Please enter the 30 transaction features.");
    }

    const values = input.value
        .trim()
        .split(/[\s,]+/)
        .filter(Boolean)
        .map(Number);

    if (values.length !== 30) {
        throw new Error(
            `You entered ${values.length} values. Exactly 30 numeric features are required.`
        );
    }

    if (!values.every(Number.isFinite)) {
        throw new Error("Please enter valid numeric values only.");
    }

    return values;
}

// Display the prediction result
function displayPrediction(data) {
    const result = document.getElementById("prediction-result");
    const placeholder = document.getElementById("result-placeholder");

    if (!result || !placeholder) return;

    const fraud = isFraud(data);

    placeholder.hidden = true;
    result.hidden = false;

    const badge = document.getElementById("result-badge");
    const label = document.getElementById("result-label");
    const symbol = document.getElementById("result-symbol");
    const probability = document.getElementById("fraud-probability");
    const progress = document.getElementById("risk-progress");

    if (badge) {
        badge.classList.toggle("fraud", fraud);
    }

    if (label) {
        label.textContent = fraud
            ? "FRAUDULENT TRANSACTION"
            : "LEGITIMATE TRANSACTION";
    }

    if (symbol) {
        symbol.textContent = fraud ? "⚠" : "✓";
    }

    if (probability) {
        probability.textContent = formatProbability(
            data.fraud_probability
        );
    }

    if (progress) {
        const percentage = Number(data.fraud_probability);

        progress.style.width = Number.isFinite(percentage)
            ? `${Math.max(0, Math.min(100, percentage))}%`
            : "0%";

        progress.classList.toggle("fraud", fraud);
    }
}

// Render transaction history and dashboard statistics
function renderHistory() {
    const history = readHistory();
    const tbody = document.getElementById("history-body");

    if (tbody) {
        tbody.replaceChildren();

        if (history.length === 0) {
            const row = document.createElement("tr");
            const cell = document.createElement("td");

            cell.colSpan = 4;
            cell.className = "empty-history";
            cell.textContent = "No transactions analyzed yet.";

            row.appendChild(cell);
            tbody.appendChild(row);
        } else {
            history.forEach(item => {
                const row = document.createElement("tr");

                const idCell = document.createElement("td");
                idCell.textContent = item.id;

                const timeCell = document.createElement("td");
                timeCell.textContent = item.time;

                const resultCell = document.createElement("td");
                const tag = document.createElement("span");

                tag.className =
                    "prediction-tag " +
                    (item.result === "FRAUDULENT"
                        ? "fraud"
                        : "legitimate");

                tag.textContent = item.result;
                resultCell.appendChild(tag);

                const probabilityCell = document.createElement("td");
                probabilityCell.textContent = item.probability;

                row.append(
                    idCell,
                    timeCell,
                    resultCell,
                    probabilityCell
                );

                tbody.appendChild(row);
            });
        }
    }

    // Dashboard statistics
    const fraudCount = history.filter(
        item => item.result === "FRAUDULENT"
    ).length;

    const legitimateCount = history.length - fraudCount;

    const totalNode = document.getElementById("total-analyzed");
    const fraudNode = document.getElementById("fraud-count");
    const legitimateNode = document.getElementById("legitimate-count");

    if (totalNode) totalNode.textContent = history.length;
    if (fraudNode) fraudNode.textContent = fraudCount;

    if (legitimateNode) {
        legitimateNode.textContent = legitimateCount;
    }

    // Latest dashboard prediction
    const latest = document.getElementById("dashboard-latest");

    if (latest) {
        latest.textContent = history.length
            ? `${history[0].id} · ${history[0].result} · Fraud probability ${history[0].probability}`
            : "No transactions analyzed yet.";
    }
}

// Handle transaction analysis
const form = document.getElementById("prediction-form");

if (form) {
    form.addEventListener("submit", async function (event) {
        event.preventDefault();

        const button = form.querySelector('button[type="submit"]');

        const originalText = button
            ? button.innerHTML
            : "Analyze Transaction";

        try {
            const features = getFeatures();

            if (button) {
                button.disabled = true;
                button.textContent = "Analyzing...";
            }

            showMessage(
                "Sending transaction to the ML model...",
                "info"
            );

            let response;

            try {
                response = await fetch(`${API_URL}/predict`, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({ features })
                });
            } catch (error) {
                throw new Error(
                    "Cannot connect to Flask. Start your backend at http://127.0.0.1:5000."
                );
            }

            let data;

            try {
                data = await response.json();
            } catch (error) {
                throw new Error(
                    "Invalid API response. Please check your Flask backend."
                );
            }

            if (!response.ok) {
                throw new Error(
                    data.error ||
                    data.message ||
                    `Request failed (${response.status}).`
                );
            }

            if (
                data.prediction === undefined ||
                data.result === undefined ||
                data.fraud_probability === undefined
            ) {
                throw new Error(
                    "The API response is missing prediction fields."
                );
            }

            // Show the result
            displayPrediction(data);

            // Save the transaction
            const history = readHistory();

            const transaction = {
                id: `TXN-${Date.now().toString().slice(-8)}`,
                time: new Date().toLocaleString(),
                result: isFraud(data)
                    ? "FRAUDULENT"
                    : "LEGITIMATE",
                probability: formatProbability(
                    data.fraud_probability
                )
            };

            saveHistory([transaction, ...history]);

            // Update history and dashboard
            renderHistory();

            showMessage(
                "Analysis completed successfully. Your result has been saved.",
                "success"
            );

        } catch (error) {
            console.error("Fraud detection error:", error);

            showMessage(
                error.message || "Something went wrong.",
                "error"
            );

        } finally {
            if (button) {
                button.disabled = false;
                button.innerHTML = originalText;
            }
        }
    });
}

// Clear transaction history
const clearButton = document.getElementById("clear-history");

if (clearButton) {
    clearButton.addEventListener("click", function () {
        if (
            confirm(
                "Clear all saved transaction history from this browser?"
            )
        ) {
            localStorage.removeItem(STORAGE_KEY);
            renderHistory();
        }
    });
}

// Update the current page when history changes in another tab
window.addEventListener("storage", function (event) {
    if (event.key === STORAGE_KEY) {
        renderHistory();
    }
});

// Initialize the current page
renderHistory();

console.log("FraudShield AI initialized.");