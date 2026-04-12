// Shared JS helpers

/**
 * Loading overlay — shown when a slow form is submitted.
 *
 * Usage: add data-loading="true" (with optional data-loading-msg="...")
 * to any <form> that triggers a long-running server operation.
 */
(function () {
  "use strict";

  function showOverlay(msg) {
    var el = document.getElementById("loading-overlay");
    if (!el) return;
    var textEl = el.querySelector(".loading-msg");
    if (textEl) textEl.textContent = msg || "Working…";
    el.setAttribute("aria-hidden", "false");
    el.classList.add("active");
  }

  function hideOverlay() {
    var el = document.getElementById("loading-overlay");
    if (!el) return;
    el.setAttribute("aria-hidden", "true");
    el.classList.remove("active");
  }

  document.addEventListener("DOMContentLoaded", function () {
    // Attach submit handlers to all slow forms
    document.querySelectorAll("form[data-loading]").forEach(function (form) {
      form.addEventListener("submit", function () {
        var msg = form.getAttribute("data-loading-msg") || "Working…";
        showOverlay(msg);
      });
    });

    // Hide overlay on back-navigation (page restored from bfcache)
    window.addEventListener("pageshow", function (e) {
      if (e.persisted) hideOverlay();
    });
  });
})();
