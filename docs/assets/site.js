(function () {
  var PAGES = [
    { href: "/", label: "Home", group: "start" },
    { href: "/why", label: "Why it matters", group: "start" },
    { href: "/features", label: "Features", group: "start" },
    { href: "/blog", label: "Blog", group: "start" },
    { href: "/workbench", label: "Workbench", group: "guide" },
    { href: "/research-workflow", label: "Research workflow", group: "guide" },
    { href: "/guide", label: "How it works", group: "guide" },
    { href: "/quickstart", label: "Quickstart", group: "guide" },
    { href: "/docker-for-beginners", label: "Docker for beginners", group: "guide" },
    { href: "/docker-images", label: "Docker images", group: "guide" },
    { href: "/commands", label: "Commands", group: "guide" },
    { href: "/diagrams", label: "Diagrams", group: "deep" },
    { href: "/fine-tuning", label: "Fine-tuning", group: "deep" },
    { href: "/architecture", label: "Architecture", group: "deep" },
    { href: "/data", label: "Data", group: "deep" },
    { href: "/api", label: "API", group: "deep" },
    { href: "/compare/vs-ad-hoc-notebooks", label: "Compare", group: "meta" },
    { href: "/limits", label: "Limits", group: "meta" },
    { href: "/help", label: "Help library", group: "meta" },
    { href: "/sitemap", label: "Sitemap", group: "meta" },
    { href: "/cite", label: "Cite & feedback", group: "meta" },
  ];

  var UI_PAGES = [
    { href: "/ui-home", label: "Home", route: "/" },
    { href: "/ui-datasets", label: "Datasets", route: "/datasets" },
    { href: "/ui-train", label: "Train", route: "/train" },
    { href: "/ui-results", label: "Results", route: "/results" },
    { href: "/ui-analytics", label: "Analytics", route: "/analytics" },
    { href: "/ui-predict", label: "Predict", route: "/predict" },
    { href: "/ui-config", label: "Config", route: "/config" },
    { href: "/ui-docs", label: "Docs (UI)", route: "/docs" },
    { href: "/ui-openapi", label: "OpenAPI", route: ":8000/docs" },
  ];

  var BLURBS = {
    "/": "Leakage-safe EHR risk prediction overview",
    "/why": "Problems, audiences, and U.S. research context",
    "/features": "Capability catalog from ingest to serve",
    "/blog": "Tutorials on leakage-safe clinical ML",
    "/workbench": "Hub for every UI page with screenshots",
    "/research-workflow": "End-to-end methods-style study loop in the workbench",
    "/guide": "A–Z walkthrough with screenshots",
    "/quickstart": "Docker → first train → predict",
    "/docker-for-beginners": "Install Docker and run the stack",
    "/docker-images": "Pull Hub images: api + web",
    "/commands": "Make, curl, tests, docs recipes",
    "/diagrams": "SVG architecture and temporal design",
    "/fine-tuning": "Compare, calibrate, promote runs",
    "/architecture": "Components, jobs, config surfaces",
    "/data": "Columns, BYO CSV, index/horizon rules",
    "/api": "Predict, jobs, schema endpoints",
    "/compare/vs-ad-hoc-notebooks": "Workbench vs ad-hoc notebooks",
    "/limits": "Honest non-goals and model card",
    "/help": "Full guide library, examples, intended use",
    "/sitemap": "Full map of every documentation page",
    "/cite": "Citation and support@larucare.com",
    "/ui-home": "Workbench checklist and status",
    "/ui-datasets": "Import tabs and health gate",
    "/ui-train": "Train, compare, leakage audit",
    "/ui-results": "Metrics, SHAP, figures, ZIP",
    "/ui-analytics": "Cohort charts and filters",
    "/ui-predict": "Schema form and risk charts",
    "/ui-config": "Persona, defaults, UI prefs",
    "/ui-docs": "In-app help and intended use",
    "/ui-openapi": "Interactive Swagger API docs",
    "/blog/why-clinical-ai-label-leakage": "What 37 of 92 MIMIC studies reveal about leakage",
    "/blog/onc-safer-guides-2025-ai": "AI transparency themes in ONC's 2025 SAFER Guides",
    "/blog/fda-ai-devices-2025": "FDA AI/ML device context for research teams",
    "/blog/healthcare-ai-market-quality-gap": "Why methods quality determines whether pilots scale",
    "/blog/nih-all-of-us-reproducible-ai": "Reproducible methods for large research datasets",
    "/blog/calibration-gap-brier-ece": "Why probability calibration belongs beside AUROC",
    "/blog/hhs-ai-strategy-2025": "HHS's 2025 health AI strategy and research practice",
    "/blog/ahrq-patient-safety-ai": "Evidence-based AI methods and patient-safety research",
    "/blog/pcori-ai-methods-infrastructure": "Shared infrastructure for AI/ML methods in CER",
    "/blog/chronic-disease-trustworthy-risk-ai": "Chronic-disease research needs honest risk models",
  };

  var PAGE_LABELS = {
    "/blog/why-clinical-ai-label-leakage": "Label leakage in clinical AI",
    "/blog/onc-safer-guides-2025-ai": "ONC SAFER Guides and AI",
    "/blog/fda-ai-devices-2025": "FDA AI/ML devices in 2025",
    "/blog/healthcare-ai-market-quality-gap": "Healthcare AI's quality gap",
    "/blog/nih-all-of-us-reproducible-ai": "All of Us and reproducible AI",
    "/blog/calibration-gap-brier-ece": "The calibration gap",
    "/blog/hhs-ai-strategy-2025": "HHS AI strategy in 2025",
    "/blog/ahrq-patient-safety-ai": "AHRQ and patient-safety AI",
    "/blog/pcori-ai-methods-infrastructure": "PCORI AI/ML methods infrastructure",
    "/blog/chronic-disease-trustworthy-risk-ai": "Trustworthy chronic-disease risk AI",
  };

  var RELATED = {
    "/": ["/why", "/guide", "/quickstart", "/workbench"],
    "/why": ["/features", "/blog/prevent-data-leakage-clinical-ai", "/limits", "/cite"],
    "/features": ["/workbench", "/research-workflow", "/guide", "/fine-tuning"],
    "/blog": [
      "/blog/why-clinical-ai-label-leakage",
      "/blog/calibration-gap-brier-ece",
      "/blog/prevent-data-leakage-clinical-ai",
      "/guide",
    ],
    "/blog/why-clinical-ai-label-leakage": [
      "/blog/prevent-data-leakage-clinical-ai", "/blog/calibration-gap-brier-ece", "/data", "/limits",
    ],
    "/blog/onc-safer-guides-2025-ai": [
      "/blog/ahrq-patient-safety-ai", "/blog/fda-ai-devices-2025", "/limits", "/guide",
    ],
    "/blog/fda-ai-devices-2025": [
      "/blog/onc-safer-guides-2025-ai", "/blog/hhs-ai-strategy-2025", "/limits", "/fine-tuning",
    ],
    "/blog/healthcare-ai-market-quality-gap": [
      "/blog/why-clinical-ai-label-leakage", "/blog/calibration-gap-brier-ece", "/fine-tuning", "/guide",
    ],
    "/blog/nih-all-of-us-reproducible-ai": [
      "/blog/why-clinical-ai-label-leakage", "/blog/pcori-ai-methods-infrastructure", "/data", "/quickstart",
    ],
    "/blog/calibration-gap-brier-ece": [
      "/blog/why-clinical-ai-label-leakage", "/blog/healthcare-ai-market-quality-gap", "/fine-tuning", "/limits",
    ],
    "/blog/hhs-ai-strategy-2025": [
      "/blog/onc-safer-guides-2025-ai", "/blog/ahrq-patient-safety-ai", "/guide", "/limits",
    ],
    "/blog/ahrq-patient-safety-ai": [
      "/blog/onc-safer-guides-2025-ai", "/blog/hhs-ai-strategy-2025", "/limits", "/data",
    ],
    "/blog/pcori-ai-methods-infrastructure": [
      "/blog/nih-all-of-us-reproducible-ai", "/blog/ahrq-patient-safety-ai", "/data", "/fine-tuning",
    ],
    "/blog/chronic-disease-trustworthy-risk-ai": [
      "/blog/why-clinical-ai-label-leakage", "/blog/calibration-gap-brier-ece", "/guide", "/data",
    ],
    "/blog/prevent-data-leakage-clinical-ai": ["/data", "/diagrams", "/blog", "/limits"],
    "/blog/ehr-risk-prediction-quickstart": ["/guide", "/quickstart", "/workbench", "/ui-train"],
    "/compare/vs-ad-hoc-notebooks": ["/alternatives/opaque-clinical-automl", "/features", "/why", "/blog"],
    "/alternatives/opaque-clinical-automl": [
      "/compare/vs-ad-hoc-notebooks",
      "/listicles/open-source-clinical-risk-tools",
      "/features",
      "/limits",
    ],
    "/listicles/open-source-clinical-risk-tools": [
      "/alternatives/opaque-clinical-automl",
      "/features",
      "/cite",
      "/blog",
    ],
    "/workbench": ["/research-workflow", "/guide", "/ui-home", "/ui-train"],
    "/research-workflow": ["/workbench", "/guide", "/ui-train", "/fine-tuning"],
    "/guide": ["/workbench", "/research-workflow", "/quickstart", "/ui-train"],
    "/quickstart": ["/docker-for-beginners", "/guide", "/docker-images", "/workbench"],
    "/docker-for-beginners": ["/quickstart", "/docker-images", "/workbench", "/limits"],
    "/docker-images": ["/docker-for-beginners", "/quickstart", "/commands", "/architecture"],
    "/commands": ["/quickstart", "/docker-images", "/fine-tuning", "/api"],
    "/diagrams": ["/fine-tuning", "/architecture", "/data", "/features"],
    "/fine-tuning": ["/ui-train", "/commands", "/diagrams", "/limits"],
    "/architecture": ["/diagrams", "/features", "/api", "/data"],
    "/data": ["/ui-datasets", "/diagrams", "/fine-tuning", "/limits"],
    "/api": ["/ui-openapi", "/commands", "/ui-predict", "/features"],
    "/limits": ["/help", "/features", "/cite", "/sitemap"],
    "/help": ["/cite", "/quickstart", "/why", "/limits"],
    "/sitemap": ["/", "/help", "/blog", "/workbench"],
    "/cite": ["/help", "/why", "/limits", "/"],
    "/ui-home": ["/ui-datasets", "/ui-train", "/ui-results", "/workbench"],
    "/ui-datasets": ["/ui-train", "/data", "/ui-config", "/workbench"],
    "/ui-train": ["/ui-results", "/fine-tuning", "/ui-config", "/workbench"],
    "/ui-results": ["/ui-predict", "/ui-analytics", "/ui-train", "/workbench"],
    "/ui-analytics": ["/ui-results", "/ui-datasets", "/features", "/workbench"],
    "/ui-predict": ["/ui-results", "/api", "/ui-train", "/workbench"],
    "/ui-config": ["/ui-train", "/ui-analytics", "/ui-home", "/workbench"],
    "/ui-docs": ["/help", "/cite", "/limits", "/workbench"],
    "/ui-openapi": ["/api", "/commands", "/ui-predict", "/workbench"],
  };

  function normalizePath(pathname) {
    var path = (pathname || "/").replace(/\/index\.html$/i, "").replace(/\.html$/i, "");
    if (!path || path === "/") return "/";
    return path.replace(/\/+$/, "") || "/";
  }

  function currentPage() {
    return normalizePath(location.pathname);
  }

  function labelFor(href) {
    var key = normalizePath(href);
    var p = PAGES.find(function (x) { return x.href === key; });
    if (p) return p.label;
    var u = UI_PAGES.find(function (x) { return x.href === key; });
    return u ? u.label : PAGE_LABELS[key] || href;
  }

  function blurbFor(href) {
    return BLURBS[normalizePath(href)] || "Part of the connected documentation set";
  }

  var path = currentPage();
  var isUiDoc = path.indexOf("/ui-") === 0 || path === "/workbench";

  function isNavActive(href) {
    if (href === "/") return path === "/";
    if (href === path) return true;
    if (href === "/workbench" && isUiDoc) return true;
    // Nested docs (e.g. /blog/…) keep the parent primary item active
    if (path.indexOf(href + "/") === 0) return true;
    return false;
  }

  var nav = document.querySelector(".nav");
  if (nav) {
    nav.innerHTML =
      PAGES.map(function (p) {
        var active = isNavActive(p.href);
        var attrs = active
          ? ' class="is-active" aria-current="page"'
          : "";
        return '<a href="' + p.href + '"' + attrs + ">" + p.label + "</a>";
      }).join("") +
      '<a class="btn btn-primary" href="https://github.com/ranasl62/ehr-chronic-disease-risk-prediction">GitHub</a>';
  }

  var main = document.getElementById("main");
  if (main && isUiDoc && !document.querySelector(".ui-subnav")) {
    var sub = document.createElement("nav");
    sub.className = "ui-subnav";
    sub.setAttribute("aria-label", "Workbench pages");
    sub.innerHTML =
      "<strong>UI pages</strong>" +
      UI_PAGES.map(function (p) {
        var cur =
          p.href === path ? ' class="is-active" aria-current="page"' : "";
        return '<a href="' + p.href + '"' + cur + ">" + p.label + "</a>";
      }).join("");
    main.insertBefore(sub, main.firstChild);
  }

  if (!document.querySelector(".feedback-strip")) {
    var strip = document.createElement("aside");
    strip.className = "feedback-strip";
    strip.innerHTML =
      "Questions or feedback: " +
      '<a href="mailto:support@larucare.com?subject=%5Behr-risk-framework%5D%20feedback">support@larucare.com</a>. ' +
      "No PHI. " +
      '<a href="/sitemap">Documentation map →</a>';
    if (main) {
      var after = main.querySelector(".ui-subnav");
      if (after && after.nextSibling) main.insertBefore(strip, after.nextSibling);
      else main.insertBefore(strip, main.firstChild);
    }
  }

  var related = RELATED[path];
  if (main && related && related.length && !document.querySelector(".doc-continue")) {
    var box = document.createElement("section");
    box.className = "section doc-continue";
    box.innerHTML =
      "<h2>Continue reading</h2>" +
      '<div class="grid-2">' +
      related
        .map(function (href) {
          return (
            '<a class="card continue-card" href="' +
            href +
            '"><h3>' +
            labelFor(href) +
            '</h3><p class="muted">' +
            blurbFor(href) +
            "</p></a>"
          );
        })
        .join("") +
      "</div>";
    main.appendChild(box);
  }

  var footerInner = document.querySelector(".site-footer .inner");
  if (footerInner && !footerInner.querySelector(".footer-map")) {
    var map = document.createElement("div");
    map.className = "footer-map";
    map.innerHTML =
      "<h3>Documentation map</h3><ul>" +
      PAGES.concat(UI_PAGES).map(function (p) {
        return '<li><a href="' + p.href + '">' + p.label + "</a></li>";
      }).join("") +
      "</ul>";
    var fine = footerInner.querySelector(".fine");
    if (fine) {
      footerInner.insertBefore(map, fine);
    } else {
      footerInner.appendChild(map);
    }
  }

  // Visible note beside Live demo CTAs (free hosted server expectations)
  if (!document.querySelector(".demo-callout")) {
    var demoHref = "https://ehr-risk-framework.larucare.com/";
    var liveBtn =
      document.querySelector('a.btn[href^="' + demoHref + '"]') ||
      document.querySelector('a[href^="' + demoHref + '"]');
    if (liveBtn) {
      var note = document.createElement("p");
      note.className = "demo-callout";
      note.innerHTML =
        "<strong>Free demo server</strong> — it may be slow. Check it with a small amount of data. " +
        "For larger workloads or freer experimentation, run locally or on your own server. " +
        '<a href="' +
        demoHref +
        '" target="_blank" rel="noopener">Open live demo</a>';
      var anchor =
        liveBtn.closest(".hero-actions") ||
        liveBtn.closest("p") ||
        liveBtn.parentElement;
      if (anchor) {
        anchor.insertAdjacentElement("afterend", note);
      }
    }
  }

  // Load on-page SEO (canonical, Open Graph, JSON-LD, keyword titles)
  if (!document.getElementById("ehr-seo-script")) {
    var css = document.querySelector('link[rel="stylesheet"][href*="site.css"]');
    var seoSrc = css
      ? css.getAttribute("href").replace(/site\.css(\?.*)?$/i, "seo.js$1")
      : "/assets/seo.js";
    var seo = document.createElement("script");
    seo.id = "ehr-seo-script";
    seo.src = seoSrc;
    seo.defer = true;
    document.head.appendChild(seo);
  }
})();
