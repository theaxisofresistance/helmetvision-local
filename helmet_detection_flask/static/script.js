document.addEventListener("DOMContentLoaded", () => {
  const APP_VERSION = "2.0.0";
  const FALLBACK_PREVIEW = "https://placehold.co/700x400/111111/333333?text=No+Feed+Active";
  const MAX_TOASTS = 5;
  const MAX_ACTIVITY_ITEMS = 5;
  const STORAGE_KEYS = {
    theme: "theme",
    confidenceThreshold: "confidenceThreshold",
    logs: "helmetVisionLogs"
  };

  const PROJECT_CONFIG = {
    datasetHelmet: 2500,
    datasetNoHelmet: 2500,
    validationAccuracy: [72, 78, 81, 85, 87, 89, 91, 93, 93.8, 94.2]
  };
  const FALLBACK_MODELS = [
    "SVM",
    "Random Forest",
    "Gradient Boosting",
    "Logistic Regression",
    "KNN"
  ];

  let currentActiveFileUrl = null;
  let currentImageFile = null;
  let currentMediaType = null;
  let isInferring = false;
  let accuracyChartInstance = null;
  let classChartInstance = null;

  const DOM = {
    html: document.documentElement,
    themeToggle: document.getElementById("themeToggle"),
    themeIcon: document.getElementById("themeIcon"),
    hamburger: document.getElementById("hamburger"),
    navLinks: document.getElementById("navLinks"),
    navLinksA: document.querySelectorAll(".nav-link"),
    settingsModal: document.getElementById("settingsModal"),
    openSettings: document.getElementById("openSettings"),
    closeSettings: document.getElementById("closeSettings"),
    saveSettings: document.getElementById("saveSettings"),
    confThreshold: document.getElementById("confThreshold"),
    confThreshVal: document.getElementById("confThreshVal"),
    themeRadios: document.querySelectorAll('input[name="theme"]'),
    toastContainer: document.getElementById("toastContainer"),
    tabBtns: document.querySelectorAll(".tab-btn"),
    tabContents: document.querySelectorAll(".tab-content"),
    dropZone: document.getElementById("dropZone"),
    uploadImage: document.getElementById("uploadImage"),
    uploadVideo: document.getElementById("uploadVideo"),
    uploadHint: document.getElementById("uploadHint"),
    modelSelect: document.getElementById("modelSelect"),
    uploadedPreview: document.getElementById("uploadedPreview"),
    mainMonitor: document.getElementById("mainMonitor"),
    detectionStatus: document.getElementById("detectionStatus"),
    monitorTime: document.getElementById("monitorTime"),
    startDetection: document.getElementById("startDetection"),
    pauseDetection: document.getElementById("pauseDetection"),
    stopDetection: document.getElementById("stopDetection"),
    confValue: document.getElementById("confValue"),
    confBar: document.getElementById("confBar"),
    activityFeed: document.getElementById("activityFeed"),
    logsSearch: document.getElementById("logsSearch"),
    logsFilter: document.getElementById("logsFilter"),
    exportCSV: document.getElementById("exportCSV"),
    exportExcel: document.getElementById("exportExcel"),
    refreshLogs: document.getElementById("refreshLogs"),
    logsBody: document.getElementById("logsBody"),
    reveals: document.querySelectorAll(".reveal"),
    accuracyChart: document.getElementById("accuracyChart"),
    classChart: document.getElementById("classChart"),
    resetDemo: document.getElementById("resetDemo"),
    statTotal: document.getElementById("statTotal"),
    statSafe: document.getElementById("statSafe"),
    statViolation: document.getElementById("statViolation"),
    statSafeRate: document.getElementById("statSafeRate"),
    statAvgConfidence: document.getElementById("statAvgConfidence")
  };

  const Logger = {
    log: (...args) => console.log(`[HelmetVision ${APP_VERSION}]`, ...args),
    warn: (...args) => console.warn(`[HelmetVision ${APP_VERSION}]`, ...args),
    error: (...args) => console.error(`[HelmetVision ${APP_VERSION}]`, ...args)
  };

  const StorageAdapter = {
    setItem(key, value) {
      try {
        localStorage.setItem(key, value);
      } catch (error) {
        Logger.warn("Storage write failed.", error);
      }
    },
    getItem(key) {
      try {
        return localStorage.getItem(key);
      } catch (error) {
        Logger.warn("Storage read failed.", error);
        return null;
      }
    },
    removeItem(key) {
      try {
        localStorage.removeItem(key);
      } catch (error) {
        Logger.warn("Storage remove failed.", error);
      }
    }
  };

  let logsData = loadPersistedLogs();

  function loadPersistedLogs() {
    const rawLogs = StorageAdapter.getItem(STORAGE_KEYS.logs);
    if (!rawLogs) return [];

    try {
      const parsed = JSON.parse(rawLogs);
      return Array.isArray(parsed) ? parsed : [];
    } catch (error) {
      Logger.warn("Corrupted log cache ignored.", error);
      return [];
    }
  }

  function persistLogs() {
    StorageAdapter.setItem(STORAGE_KEYS.logs, JSON.stringify(logsData));
  }

  function formatTime(date = new Date()) {
    return date.toTimeString().split(" ")[0];
  }

  function formatStatusLabel(status) {
    return status === "safe" ? "Helmet Detected" : "No Helmet Detected";
  }

  function getStatusFromPrediction(prediction) {
    const normalized = String(prediction || "").toLowerCase();
    if (normalized.includes("no helmet") || normalized.includes("without helmet")) {
      return "violation";
    }
    return "safe";
  }

  function getConfidenceFromProbabilities(probabilities) {
    if (!Array.isArray(probabilities) || probabilities.length === 0) return null;
    const topProbability = Math.max(...probabilities);
    return Math.round(topProbability * 100);
  }

  function updateDetectionStatus(state, text) {
    if (!DOM.detectionStatus) return;
    DOM.detectionStatus.className = `status-${state}`;

    const iconByState = {
      idle: "fa-pause-circle",
      running: "fa-circle-dot pulse-green",
      paused: "fa-pause-circle",
      success: "fa-circle-check",
      error: "fa-triangle-exclamation"
    };
    DOM.detectionStatus.innerHTML = `<i class="fa-solid ${iconByState[state] || iconByState.idle}"></i> ${text}`;
  }

  function updateConfidenceDisplay(confidence) {
    const safeConfidence = Number.isFinite(confidence) ? confidence : 0;
    if (DOM.confValue) DOM.confValue.textContent = `${safeConfidence}%`;
    if (DOM.confBar) DOM.confBar.style.width = `${safeConfidence}%`;
  }

  function showToast(type, title, message) {
    if (!DOM.toastContainer) return;

    const activeToasts = DOM.toastContainer.querySelectorAll(".toast");
    if (activeToasts.length >= MAX_TOASTS && activeToasts[0]) {
      activeToasts[0].remove();
    }

    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;

    let iconClass = "fa-circle-info";
    if (type === "error" || type === "violation") iconClass = "fa-triangle-exclamation";
    if (type === "success" || type === "safe") iconClass = "fa-circle-check";

    toast.innerHTML = `
      <i class="fa-solid ${iconClass} toast-icon"></i>
      <div class="toast-body">
        <div class="toast-title">${title}</div>
        <div class="toast-msg">${message}</div>
        <div class="toast-time">${formatTime()}</div>
      </div>
      <button class="toast-close" aria-label="Dismiss Alert">&times;</button>
    `;

    DOM.toastContainer.appendChild(toast);

    const closeButton = toast.querySelector(".toast-close");
    if (closeButton) {
      closeButton.addEventListener("click", () => removeToast(toast), { once: true });
    }

    setTimeout(() => {
      if (toast.parentElement) removeToast(toast);
    }, 5000);
  }

  function removeToast(toast) {
    toast.classList.add("removing");
    toast.addEventListener("animationend", () => toast.remove(), { once: true });
  }

  function clearMonitorVisualNodes() {
    if (DOM.uploadedPreview) {
      DOM.uploadedPreview.src = FALLBACK_PREVIEW;
      DOM.uploadedPreview.style.display = "block";
    }

    if (DOM.mainMonitor) {
      const dynamicVideo = DOM.mainMonitor.querySelector(".dynamic-video-node");
      if (dynamicVideo) {
        dynamicVideo.pause();
        dynamicVideo.removeAttribute("src");
        dynamicVideo.load();
        dynamicVideo.remove();
      }
    }

    if (currentActiveFileUrl) {
      URL.revokeObjectURL(currentActiveFileUrl);
      currentActiveFileUrl = null;
    }
  }

  function readFileAsDataUrl(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }

  async function createSnapshot(file) {
    if (!file.type.startsWith("image/")) return FALLBACK_PREVIEW;
    return readFileAsDataUrl(file);
  }

  function addLogEntry(entry) {
    logsData.unshift(entry);
    logsData = logsData.slice(0, 100);
    persistLogs();
    updateSummaryCards();
    renderActivityFeed();
    executeLogsFilteringOperation();
  }

  function updateSummaryCards() {
    const total = logsData.length;
    const safeCount = logsData.filter((item) => item.status === "safe").length;
    const violationCount = logsData.filter((item) => item.status === "violation").length;
    const avgConfidence = total
      ? Math.round(
          logsData.reduce((sum, item) => sum + (Number(item.confidenceValue) || 0), 0) / total
        )
      : 0;
    const safeRate = total ? Math.round((safeCount / total) * 100) : 0;
    const latestConfidence = total ? Number(logsData[0].confidenceValue) || 0 : 0;

    if (DOM.statTotal) DOM.statTotal.textContent = String(total);
    if (DOM.statSafe) DOM.statSafe.textContent = String(safeCount);
    if (DOM.statViolation) DOM.statViolation.textContent = String(violationCount);
    if (DOM.statSafeRate) DOM.statSafeRate.innerHTML = `${safeRate}<span style="font-size:1rem">%</span>`;
    if (DOM.statAvgConfidence) DOM.statAvgConfidence.innerHTML = `${avgConfidence}<span style="font-size:1rem">%</span>`;
    updateConfidenceDisplay(latestConfidence);
  }

  function renderActivityFeed() {
    if (!DOM.activityFeed) return;

    DOM.activityFeed.innerHTML = "";
    const recentItems = logsData.slice(0, MAX_ACTIVITY_ITEMS);

    if (recentItems.length === 0) {
      DOM.activityFeed.innerHTML = `
        <li class="feed-item safe">
          <span class="feed-badge safe"><i class="fa-solid fa-clock"></i></span>
          <span class="feed-text">No inference history yet</span>
          <span class="feed-time">--:--:--</span>
        </li>
      `;
      return;
    }

    const fragment = document.createDocumentFragment();
    recentItems.forEach((item) => {
      const element = document.createElement("li");
      element.className = `feed-item ${item.status}`;
      element.innerHTML = `
        <span class="feed-badge ${item.status}">
          <i class="fa-solid ${item.status === "safe" ? "fa-check" : "fa-exclamation"}"></i>
        </span>
        <span class="feed-text">${item.prediction} via ${item.model} (${item.confidence})</span>
        <span class="feed-time">${item.timestamp}</span>
      `;
      fragment.appendChild(element);
    });

    DOM.activityFeed.appendChild(fragment);
  }

  function renderLogsTableEngine(dataset) {
    if (!DOM.logsBody) return;
    DOM.logsBody.innerHTML = "";

    if (dataset.length === 0) {
      DOM.logsBody.innerHTML = `<tr><td colspan="5" style="text-align:center;color:var(--text-muted);">No matching records found.</td></tr>`;
      return;
    }

    const fragment = document.createDocumentFragment();
    dataset.forEach((row, index) => {
      const tr = document.createElement("tr");
      const badgeClass = row.status === "safe" ? "status-badge safe" : "status-badge violation";
      tr.innerHTML = `
        <td>${index + 1}</td>
        <td>${row.timestamp}</td>
        <td><img src="${row.snapshot}" class="snap-thumb" alt="Inference snapshot" /></td>
        <td><span class="${badgeClass}">${formatStatusLabel(row.status)}</span></td>
        <td><span class="conf-pill">${row.confidence}</span></td>
      `;
      fragment.appendChild(tr);
    });

    DOM.logsBody.appendChild(fragment);
  }

  function executeLogsFilteringOperation() {
    const searchString = DOM.logsSearch ? DOM.logsSearch.value.toLowerCase().trim() : "";
    const filterValue = DOM.logsFilter ? DOM.logsFilter.value : "all";

    const filteredLogs = logsData.filter((item) => {
      const matchesSearch =
        item.timestamp.toLowerCase().includes(searchString) ||
        item.status.toLowerCase().includes(searchString) ||
        item.prediction.toLowerCase().includes(searchString) ||
        item.model.toLowerCase().includes(searchString) ||
        item.confidence.toLowerCase().includes(searchString);
      const matchesFilter = filterValue === "all" || item.status === filterValue;
      return matchesSearch && matchesFilter;
    });

    renderLogsTableEngine(filteredLogs);
  }

  async function runInferenceWithFile(file) {
    if (!file || !file.type.startsWith("image/")) {
      showToast("error", "Unsupported Media", "The inference API currently supports images only.");
      return;
    }

    if (isInferring) return;

    isInferring = true;
    currentImageFile = file;
    currentMediaType = "image";
    updateDetectionStatus("running", "Running Inference");
    if (DOM.uploadHint) {
      DOM.uploadHint.textContent = `Running ${DOM.modelSelect ? DOM.modelSelect.value : "KNN"} inference for ${file.name}...`;
    }

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("model_name", DOM.modelSelect ? DOM.modelSelect.value : "KNN");

      const response = await fetch("/api/infer", {
        method: "POST",
        body: formData
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || "Inference request failed.");
      }

      const status = getStatusFromPrediction(payload.prediction);
      const confidence = getConfidenceFromProbabilities(payload.probabilities);
      const snapshot = await createSnapshot(file);

      updateConfidenceDisplay(confidence);
      updateDetectionStatus("success", `${formatStatusLabel(status)} Ready`);

      const logEntry = {
        timestamp: formatTime(),
        snapshot,
        status,
        confidence: `${Number.isFinite(confidence) ? confidence : 0}%`,
        confidenceValue: Number.isFinite(confidence) ? confidence : 0,
        model: payload.model,
        prediction: payload.prediction,
        predictionIndex: payload.prediction_index
      };

      addLogEntry(logEntry);

      if (DOM.uploadHint) {
        DOM.uploadHint.textContent = `${payload.model} predicted "${payload.prediction}" with ${logEntry.confidence} confidence.`;
      }

      showToast(
        status === "safe" ? "success" : "violation",
        payload.model,
        `${payload.prediction} detected${Number.isFinite(confidence) ? ` (${confidence}%)` : ""}.`
      );
    } catch (error) {
      updateDetectionStatus("error", "Inference Failed");
      if (DOM.uploadHint) DOM.uploadHint.textContent = error.message;
      showToast("error", "Inference Failed", error.message);
      Logger.error(error);
    } finally {
      isInferring = false;
    }
  }

  async function handleGenericInboundFile(file) {
    if (!file) return;

    try {
      clearMonitorVisualNodes();

      if (file.type.startsWith("image/")) {
        currentActiveFileUrl = URL.createObjectURL(file);
        currentMediaType = "image";
        currentImageFile = file;

        if (DOM.uploadedPreview) {
          DOM.uploadedPreview.src = currentActiveFileUrl;
          DOM.uploadedPreview.style.display = "block";
        }

        showToast("info", "Image Uploaded", `File active: "${file.name}"`);
        await runInferenceWithFile(file);
        return;
      }

      if (file.type.startsWith("video/")) {
        currentMediaType = "video";
        currentImageFile = null;
        if (DOM.uploadedPreview) DOM.uploadedPreview.style.display = "none";

        currentActiveFileUrl = URL.createObjectURL(file);
        const videoElement = document.createElement("video");
        videoElement.className = "dynamic-video-node";
        videoElement.autoplay = true;
        videoElement.muted = true;
        videoElement.loop = true;
        videoElement.playsInline = true;
        videoElement.style.cssText = "width:100%; height:100%; object-fit:cover; border-radius:8px;";
        videoElement.src = currentActiveFileUrl;

        if (DOM.mainMonitor) {
          DOM.mainMonitor.insertBefore(videoElement, DOM.detectionStatus.parentElement);
        }

        updateDetectionStatus("paused", "Video Preview Loaded");
        if (DOM.uploadHint) {
          DOM.uploadHint.textContent = "Video preview loaded. The current API only supports still-image inference.";
        }
        showToast("info", "Video Uploaded", "Video preview loaded. Upload an image to run inference.");
        return;
      }

      showToast("error", "Incompatible Format", "Please upload a standard image or video file.");
    } catch (error) {
      showToast("error", "Ingestion Error", "Failed to process the uploaded file.");
      Logger.error(error);
    }
  }

  function initLocalStorage() {
    const savedTheme = StorageAdapter.getItem(STORAGE_KEYS.theme) || "dark";
    if (DOM.html) DOM.html.setAttribute("data-theme", savedTheme);
    if (DOM.themeIcon) {
      DOM.themeIcon.className = savedTheme === "dark" ? "fa-solid fa-moon" : "fa-solid fa-sun";
    }
    DOM.themeRadios.forEach((radio) => {
      radio.checked = radio.value === savedTheme;
    });

    const savedThreshold = StorageAdapter.getItem(STORAGE_KEYS.confidenceThreshold) || "50";
    if (DOM.confThreshold) DOM.confThreshold.value = savedThreshold;
    if (DOM.confThreshVal) DOM.confThreshVal.textContent = `${savedThreshold}%`;
  }

  function populateModelDropdown(models) {
    if (!DOM.modelSelect) return;

    const selectedModel = DOM.modelSelect.value || "KNN";
    DOM.modelSelect.innerHTML = "";

    models.forEach((modelName) => {
      const option = document.createElement("option");
      option.value = modelName;
      option.textContent = modelName;
      if (modelName === selectedModel) {
        option.selected = true;
      }
      DOM.modelSelect.appendChild(option);
    });

    if (![...DOM.modelSelect.options].some((option) => option.selected) && DOM.modelSelect.options[0]) {
      DOM.modelSelect.options[0].selected = true;
    }
  }

  async function initAvailableModels() {
    populateModelDropdown(FALLBACK_MODELS);

    try {
      const response = await fetch("/api/health");
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || "Could not load models.");
      }

      if (Array.isArray(payload.available_models) && payload.available_models.length > 0) {
        populateModelDropdown(payload.available_models);
      }
    } catch (error) {
      Logger.warn("Falling back to static model list.", error);
    }
  }

  function initNavigation() {
    if (DOM.themeToggle) {
      DOM.themeToggle.addEventListener("click", () => {
        const currentTheme = DOM.html.getAttribute("data-theme");
        const nextTheme = currentTheme === "dark" ? "light" : "dark";
        DOM.html.setAttribute("data-theme", nextTheme);
        StorageAdapter.setItem(STORAGE_KEYS.theme, nextTheme);

        if (DOM.themeIcon) {
          DOM.themeIcon.className = nextTheme === "dark" ? "fa-solid fa-moon" : "fa-solid fa-sun";
        }
      });
    }

    if (DOM.hamburger && DOM.navLinks) {
      DOM.hamburger.addEventListener("click", (event) => {
        event.stopPropagation();
        DOM.navLinks.classList.toggle("open");
      });

      DOM.navLinksA.forEach((link) => {
        link.addEventListener("click", () => {
          DOM.navLinks.classList.remove("open");
          DOM.navLinksA.forEach((item) => item.classList.remove("active"));
          link.classList.add("active");
        });
      });

      document.addEventListener("click", (event) => {
        if (
          DOM.navLinks.classList.contains("open") &&
          !DOM.navLinks.contains(event.target) &&
          !DOM.hamburger.contains(event.target)
        ) {
          DOM.navLinks.classList.remove("open");
        }
      });
    }

    DOM.tabBtns.forEach((button) => {
      button.addEventListener("click", () => {
        DOM.tabBtns.forEach((btn) => btn.classList.remove("active"));
        DOM.tabContents.forEach((content) => content.classList.remove("active"));

        button.classList.add("active");
        const targetTab = document.getElementById(`tab-${button.getAttribute("data-tab")}`);
        if (targetTab) targetTab.classList.add("active");
      });
    });
  }

  function initRevealAndClock() {
    if (DOM.confBar) {
      DOM.confBar.style.transition = "width 0.5s ease";
    }

    const checkScrollReveal = () => {
      const triggerBottom = window.innerHeight * 0.85;
      DOM.reveals.forEach((reveal) => {
        if (reveal.getBoundingClientRect().top < triggerBottom) {
          reveal.classList.add("visible");
        }
      });
    };

    if (DOM.reveals.length > 0) {
      window.addEventListener("scroll", checkScrollReveal, { passive: true });
      checkScrollReveal();
    }

    const tick = () => {
      if (DOM.monitorTime) DOM.monitorTime.textContent = formatTime();
      setTimeout(tick, 1000);
    };
    tick();
  }

  function initMediaInputs() {
    if (DOM.uploadImage) {
      DOM.uploadImage.addEventListener("change", async (event) => {
        if (event.target.files && event.target.files[0]) {
          await handleGenericInboundFile(event.target.files[0]);
          event.target.value = "";
        }
      });
    }

    if (DOM.uploadVideo) {
      DOM.uploadVideo.addEventListener("change", async (event) => {
        if (event.target.files && event.target.files[0]) {
          await handleGenericInboundFile(event.target.files[0]);
          event.target.value = "";
        }
      });
    }

    if (DOM.dropZone) {
      ["dragenter", "dragover"].forEach((eventName) => {
        DOM.dropZone.addEventListener(eventName, (event) => {
          event.preventDefault();
          event.stopPropagation();
          DOM.dropZone.classList.add("drag-over");
        });
      });

      ["dragleave", "drop"].forEach((eventName) => {
        DOM.dropZone.addEventListener(eventName, (event) => {
          event.preventDefault();
          event.stopPropagation();
          DOM.dropZone.classList.remove("drag-over");
        });
      });

      DOM.dropZone.addEventListener("drop", async (event) => {
        if (event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0]) {
          await handleGenericInboundFile(event.dataTransfer.files[0]);
        }
      });
    }
  }

  function initDetectionControls() {
    if (DOM.startDetection) {
      DOM.startDetection.addEventListener("click", async () => {
        if (currentMediaType === "image" && currentImageFile) {
          await runInferenceWithFile(currentImageFile);
          return;
        }

        if (currentMediaType === "video") {
          showToast("info", "Image Only", "Inference API is currently limited to image uploads.");
          return;
        }

        showToast("error", "No Media Selected", "Upload an image first to run inference.");
      });
    }

    if (DOM.pauseDetection) {
      DOM.pauseDetection.addEventListener("click", () => {
        updateDetectionStatus("paused", "Detection Paused");
        showToast("info", "Detection Paused", "Inference is paused.");
      });
    }

    if (DOM.stopDetection) {
      DOM.stopDetection.addEventListener("click", () => {
        currentImageFile = null;
        currentMediaType = null;
        clearMonitorVisualNodes();
        updateDetectionStatus("idle", "Detection Stopped");
        if (DOM.uploadHint) {
          DOM.uploadHint.textContent = "Upload an image to run inference with the selected model. Results will be saved in your browser history.";
        }
        showToast("info", "Detection Stopped", "Preview cleared.");
      });
    }
  }

  function initLogsControls() {
    if (DOM.logsSearch) DOM.logsSearch.addEventListener("input", executeLogsFilteringOperation);
    if (DOM.logsFilter) DOM.logsFilter.addEventListener("change", executeLogsFilteringOperation);

    if (DOM.refreshLogs) {
      DOM.refreshLogs.addEventListener("click", () => {
        logsData = loadPersistedLogs();
        updateSummaryCards();
        renderActivityFeed();
        executeLogsFilteringOperation();
        showToast("success", "Logs Synced", "Logs reloaded from localStorage.");
      });
    }

    if (DOM.exportCSV) {
      DOM.exportCSV.addEventListener("click", () => {
        try {
          let csv = "Timestamp,Model,Prediction,Status,Confidence\r\n";
          logsData.forEach((row) => {
            csv += `${row.timestamp},${row.model},${row.prediction},${formatStatusLabel(row.status)},${row.confidence}\r\n`;
          });

          const blob = new Blob([new Uint8Array([0xef, 0xbb, 0xbf]), csv], {
            type: "text/csv;charset=utf-8;"
          });
          const url = URL.createObjectURL(blob);
          const link = document.createElement("a");
          link.href = url;
          link.download = `HelmetVision_Logs_${Date.now()}.csv`;
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
          URL.revokeObjectURL(url);
          showToast("success", "CSV Exported", "Log export completed.");
        } catch (error) {
          showToast("error", "Export Failed", "Unable to export CSV.");
          Logger.error(error);
        }
      });
    }

    if (DOM.exportExcel) {
      DOM.exportExcel.addEventListener("click", () => {
        try {
          let html = `<html xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:x="urn:schemas-microsoft-com:office:excel" xmlns="http://www.w3.org/TR/REC-html40">`;
          html += `<head><meta charset="utf-8"/></head><body>`;
          html += `<table border="1"><thead><tr><th>Timestamp</th><th>Model</th><th>Prediction</th><th>Status</th><th>Confidence</th></tr></thead><tbody>`;

          logsData.forEach((row) => {
            html += `<tr><td>${row.timestamp}</td><td>${row.model}</td><td>${row.prediction}</td><td>${formatStatusLabel(row.status)}</td><td>${row.confidence}</td></tr>`;
          });

          html += `</tbody></table></body></html>`;
          const blob = new Blob([html], { type: "application/vnd.ms-excel;charset=utf-8" });
          const url = URL.createObjectURL(blob);
          const link = document.createElement("a");
          link.href = url;
          link.download = `HelmetVision_Logs_${Date.now()}.xls`;
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
          URL.revokeObjectURL(url);
          showToast("success", "Excel Exported", "Log export completed.");
        } catch (error) {
          showToast("error", "Export Failed", "Unable to export Excel file.");
          Logger.error(error);
        }
      });
    }
  }

  function initSettings() {
    if (DOM.openSettings && DOM.settingsModal) {
      DOM.openSettings.addEventListener("click", () => DOM.settingsModal.classList.add("open"));
    }
    if (DOM.closeSettings && DOM.settingsModal) {
      DOM.closeSettings.addEventListener("click", () => DOM.settingsModal.classList.remove("open"));
    }
    if (DOM.settingsModal) {
      DOM.settingsModal.addEventListener("click", (event) => {
        if (event.target === DOM.settingsModal) DOM.settingsModal.classList.remove("open");
      });
    }
    if (DOM.confThreshold) {
      DOM.confThreshold.addEventListener("input", (event) => {
        if (DOM.confThreshVal) DOM.confThreshVal.textContent = `${event.target.value}%`;
      });
    }

    if (DOM.saveSettings) {
      DOM.saveSettings.addEventListener("click", () => {
        const threshold = DOM.confThreshold ? DOM.confThreshold.value : "50";
        const selectedTheme = document.querySelector('input[name="theme"]:checked');
        const theme = selectedTheme ? selectedTheme.value : "dark";

        StorageAdapter.setItem(STORAGE_KEYS.confidenceThreshold, threshold);
        StorageAdapter.setItem(STORAGE_KEYS.theme, theme);

        DOM.html.setAttribute("data-theme", theme);
        if (DOM.themeIcon) {
          DOM.themeIcon.className = theme === "dark" ? "fa-solid fa-moon" : "fa-solid fa-sun";
        }

        if (DOM.settingsModal) DOM.settingsModal.classList.remove("open");
        showToast("success", "Settings Saved", `Threshold ${threshold}% and theme ${theme} saved.`);
      });
    }
  }

  function initResetHook() {
    if (DOM.resetDemo) {
      DOM.resetDemo.addEventListener("click", () => {
        logsData = [];
        persistLogs();
        updateSummaryCards();
        renderActivityFeed();
        executeLogsFilteringOperation();
        clearMonitorVisualNodes();
        updateDetectionStatus("idle", "Detection Idle");
        showToast("success", "History Cleared", "Stored logs were removed from localStorage.");
      });
    }
  }

  function initializeChartsEngine() {
    if (typeof Chart === "undefined") return;

    if (DOM.accuracyChart) {
      accuracyChartInstance = new Chart(DOM.accuracyChart.getContext("2d"), {
        type: "line",
        data: {
          labels: ["Ep 10", "Ep 20", "Ep 30", "Ep 40", "Ep 50", "Ep 60", "Ep 70", "Ep 80", "Ep 90", "Ep 100"],
          datasets: [{
            label: "Validation Accuracy",
            data: [...PROJECT_CONFIG.validationAccuracy],
            borderColor: "#FFCC00",
            backgroundColor: "rgba(255, 204, 0, 0.08)",
            borderWidth: 2,
            fill: true,
            tension: 0.3,
            pointRadius: 3
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            y: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#888" }, min: 60, max: 100 },
            x: { grid: { display: false }, ticks: { color: "#888" } }
          }
        }
      });
    }

    if (DOM.classChart) {
      classChartInstance = new Chart(DOM.classChart.getContext("2d"), {
        type: "bar",
        data: {
          labels: ["Helmet Class", "No Helmet Class"],
          datasets: [{
            label: "Dataset Distribution",
            data: [PROJECT_CONFIG.datasetHelmet, PROJECT_CONFIG.datasetNoHelmet],
            backgroundColor: ["rgba(0, 230, 118, 0.65)", "rgba(255, 23, 68, 0.65)"],
            borderColor: ["#00e676", "#ff1744"],
            borderWidth: 1,
            barThickness: 40
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            y: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#888" }, beginAtZero: true },
            x: { grid: { display: false }, ticks: { color: "#888" } }
          }
        }
      });
    }
  }

  window.addEventListener("beforeunload", () => {
    if (currentActiveFileUrl) URL.revokeObjectURL(currentActiveFileUrl);
    if (accuracyChartInstance) accuracyChartInstance.destroy();
    if (classChartInstance) classChartInstance.destroy();
  });

  initLocalStorage();
  initAvailableModels();
  initNavigation();
  initRevealAndClock();
  initMediaInputs();
  initDetectionControls();
  initLogsControls();
  initSettings();
  initResetHook();
  renderActivityFeed();
  updateSummaryCards();
  executeLogsFilteringOperation();
  updateDetectionStatus("idle", "Detection Idle");
  initializeChartsEngine();
});
