window.onload = function () {
  const input = document.getElementById("user-input");
  input.focus();

  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      showTyping();
      document.querySelector("form").requestSubmit();
    }
  });
};

function showTyping() {
  const typingIndicator = document.getElementById("typing-indicator");
  if (typingIndicator) {
    typingIndicator.style.display = "flex";
  }
}

function validateInput(e) {
  const input = document.getElementById("user-input");
  const submitter = e.submitter;

  if (submitter.name === "ask" && input.value.trim() === "") {
    alert("Scrie o întrebare înainte de a trimite.");
    return false;
  }
  return true;
}
