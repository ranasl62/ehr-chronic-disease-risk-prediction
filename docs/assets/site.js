(function () {
  var PAGES = [
    { href: "index.html", label: "Home", group: "start" },
    { href: "why.html", label: "Why it matters", group: "start" },
    { href: "features.html", label: "Features", group: "start" },
    { href: "workbench.html", label: "Workbench", group: "guide" },
    { href: "quickstart.html", label: "Quickstart", group: "guide" },
    { href: "docker-images.html", label: "Docker images", group: "guide" },
    { href: "commands.html", label: "Commands", group: "guide" },
    { href: "diagrams.html", label: "Diagrams", group: "deep" },
    { href: "fine-tuning.html", label: "Fine-tuning", group: "deep" },
    { href: "architecture.html", label: "Architecture", group: "deep" },
    { href: "data.html", label: "Data", group: "deep" },
    { href: "api.html", label: "API", group: "deep" },
    { href: "limits.html", label: "Limits", group: "meta" },
    { href: "help.html", label: "Help library", group: "meta" },
    { href: "sitemap.html", label: "Sitemap", group: "meta" },
    { href: "cite.html", label: "Cite & feedback", group: "meta" },
  ];

  var UI_PAGES = [
    { href: "ui-home.html", label: "Home", route: "/" },
    { href: "ui-datasets.html", label: "Datasets", route: "/datasets" },
    { href: "ui-train.html", label: "Train", route: "/train" },
    { href: "ui-results.html", label: "Results", route: "/results" },
    { href: "ui-analytics.html", label: "Analytics", route: "/analytics" },
    { href: "ui-predict.html", label: "Predict", route: "/predict" },
    { href: "ui-config.html", label: "Config", route: "/config" },
    { href: "ui-docs.html", label: "Docs (UI)", route: "/docs" },
    { href: "ui-openapi.html", label: "OpenAPI", route: ":8000/docs" },
  ];

  var BLURBS = {
    "index.html": "Overview, U.S. research context, and docs map",
    "why.html": "Problems, audiences, and U.S. research context",
    "features.html": "Capability catalog from ingest to serve",
    "workbench.html": "Hub for every UI page with screenshots",
    "quickstart.html": "Docker → first train → predict",
    "docker-images.html": "Pull Hub images: api + web",
    "commands.html": "Make, curl, tests, docs recipes",
    "diagrams.html": "SVG architecture and temporal design",
    "fine-tuning.html": "Compare, calibrate, promote runs",
    "architecture.html": "Components, jobs, config surfaces",
    "data.html": "Columns, BYO CSV, index/horizon rules",
    "api.html": "Predict, jobs, schema endpoints",
    "limits.html": "Honest non-goals and model card",
    "help.html": "Full guide library, examples, intended use",
    "sitemap.html": "Full map of every documentation page",
    "cite.html": "Citation and support@larucare.com",
    "ui-home.html": "Workbench checklist and status",
    "ui-datasets.html": "Import tabs and health gate",
    "ui-train.html": "Train, compare, leakage audit",
    "ui-results.html": "Metrics, SHAP, figures, ZIP",
    "ui-analytics.html": "Cohort charts and filters",
    "ui-predict.html": "Schema form and risk charts",
    "ui-config.html": "Persona, defaults, UI prefs",
    "ui-docs.html": "In-app help and intended use",
    "ui-openapi.html": "Interactive Swagger API docs",
  };

  var RELATED = {
    "index.html": ["why.html", "workbench.html", "quickstart.html", "diagrams.html"],
    "why.html": ["features.html", "workbench.html", "limits.html", "cite.html"],
    "features.html": ["workbench.html", "diagrams.html", "fine-tuning.html", "limits.html"],
    "workbench.html": ["ui-home.html", "ui-train.html", "ui-predict.html", "quickstart.html"],
    "quickstart.html": ["docker-images.html", "workbench.html", "commands.html", "data.html"],
    "docker-images.html": ["quickstart.html", "commands.html", "architecture.html", "limits.html"],
    "commands.html": ["quickstart.html", "docker-images.html", "fine-tuning.html", "api.html"],
    "diagrams.html": ["fine-tuning.html", "architecture.html", "data.html", "features.html"],
    "fine-tuning.html": ["ui-train.html", "commands.html", "diagrams.html", "limits.html"],
    "architecture.html": ["diagrams.html", "features.html", "api.html", "data.html"],
    "data.html": ["ui-datasets.html", "diagrams.html", "fine-tuning.html", "limits.html"],
    "api.html": ["ui-openapi.html", "commands.html", "ui-predict.html", "features.html"],
    "limits.html": ["help.html", "features.html", "cite.html", "sitemap.html"],
    "help.html": ["cite.html", "quickstart.html", "why.html", "limits.html"],
    "sitemap.html": ["index.html", "help.html", "workbench.html", "limits.html"],
    "cite.html": ["help.html", "why.html", "limits.html", "index.html"],
    "ui-home.html": ["ui-datasets.html", "ui-train.html", "ui-results.html", "workbench.html"],
    "ui-datasets.html": ["ui-train.html", "data.html", "ui-config.html", "workbench.html"],
    "ui-train.html": ["ui-results.html", "fine-tuning.html", "ui-config.html", "workbench.html"],
    "ui-results.html": ["ui-predict.html", "ui-analytics.html", "ui-train.html", "workbench.html"],
    "ui-analytics.html": ["ui-results.html", "ui-datasets.html", "features.html", "workbench.html"],
    "ui-predict.html": ["ui-results.html", "api.html", "ui-train.html", "workbench.html"],
    "ui-config.html": ["ui-train.html", "ui-analytics.html", "ui-home.html", "workbench.html"],
    "ui-docs.html": ["help.html", "cite.html", "limits.html", "workbench.html"],
    "ui-openapi.html": ["api.html", "commands.html", "ui-predict.html", "workbench.html"],
  };

  function currentPage() {
    var path = (location.pathname.split("/").pop() || "index.html").replace(/\/$/, "");
    if (!path) path = "index.html";
    return path;
  }

  function labelFor(href) {
    var p = PAGES.find(function (x) { return x.href === href; });
    if (p) return p.label;
    var u = UI_PAGES.find(function (x) { return x.href === href; });
    return u ? u.label : href;
  }

  function blurbFor(href) {
    return BLURBS[href] || "Part of the connected documentation set";
  }

  var path = currentPage();
  var isUiDoc = path.indexOf("ui-") === 0 || path === "workbench.html";

  var nav = document.querySelector(".nav");
  if (nav) {
    nav.innerHTML =
      PAGES.map(function (p) {
        var cur =
          p.href === path || (p.href === "workbench.html" && isUiDoc)
            ? ' aria-current="page"'
            : "";
        return '<a href="' + p.href + '"' + cur + ">" + p.label + "</a>";
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
        var cur = p.href === path ? ' aria-current="page"' : "";
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
      '<a href="sitemap.html">Documentation map →</a>';
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
})();
