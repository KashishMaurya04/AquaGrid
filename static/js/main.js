
const simulateButton = document.getElementById("simulate-btn");
const leakCount = document.getElementById("leak-count");
const message = document.getElementById("simulation-message");

if (simulateButton && leakCount && message) {
    simulateButton.addEventListener("click", () => {
        const currentCount = Number(leakCount.textContent);
        leakCount.textContent = currentCount + 1;
        message.textContent =
            "Demo alert created in the page. No real sensor reading or database record was created.";
        simulateButton.textContent = "✓ Demo Alert Added";
        simulateButton.disabled = true;
    });
}
