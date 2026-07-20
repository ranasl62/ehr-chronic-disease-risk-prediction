(function () {
  var PAGES = [
    { href: "/", label: "Home", group: "start" },
    { href: "/why", label: "Why it matters", group: "start" },
    { href: "/features", label: "Features", group: "start" },
    { href: "/workbench", label: "Workbench", group: "guide" },
    { href: "/quickstart", label: "Quickstart", group: "guide" },
    { href: "/docker-images", label: "Docker images", group: "guide" },
    { href: "/commands", label: "Commands", group: "guide" },
    { href: "/diagrams", label: "Diagrams", group: "deep" },
    { href: "/fine-tuning", label: "Fine-tuning", group: "deep" },
    { href: "/architecture", label: "Architecture", group: "deep" },
    { href: "/data", label: "Data", group: "deep" },
    { href: "/api", label: "API", group: "deep" },
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
    "/": "Overview, U.S. research context, and docs map",
    "/why": "Problems, audiences, and U.S. research context",
    "/features": "Capability catalog from ingest to serve",
    "/workbench": "Hub for every UI page with screenshots",
    "/quickstart": "Docker → first train → predict",
    "/docker-images": "Pull Hub images: api + web",
    "/commands": "Make, curl, tests, docs recipes",
    "/diagrams": "SVG architecture and temporal design",
    "/fine-tuning": "Compare, calibrate, promote runs",
    "/architecture": "Components, jobs, config surfaces",
    "/data": "Columns, BYO CSV, index/horizon rules",
    "/api": "Predict, jobs, schema endpoints",
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
  };

  var RELATED = {
    "/": ["/why", "/workbench", "/quickstart", "/diagrams"],
    "/why": ["/features", "/workbench", "/limits", "/cite"],
    "/features": ["/workbench", "/diagrams", "/fine-tuning", "/limits"],
    "/workbench": ["/ui-home", "/ui-train", "/ui-predict", "/quickstart"],
    "/quickstart": ["/docker-images", "/workbench", "/commands", "/data"],
    "/docker-images": ["/quickstart", "/commands", "/architecture", "/limits"],
    "/commands": ["/quickstart", "/docker-images", "/fine-tuning", "/api"],
    "/diagrams": ["/fine-tuning", "/architecture", "/data", "/features"],
    "/fine-tuning": ["/ui-train", "/commands", "/diagrams", "/limits"],
    "/architecture": ["/diagrams", "/features", "/api", "/data"],
    "/data": ["/ui-datasets", "/diagrams", "/fine-tuning", "/limits"],
    "/api": ["/ui-openapi", "/commands", "/ui-predict", "/features"],
    "/limits": ["/help", "/features", "/cite", "/sitemap"],
    "/help": ["/cite", "/quickstart", "/why", "/limits"],
    "/sitemap": ["/", "/help", "/workbench", "/limits"],
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
    return u ? u.label : href;
  }

  function blurbFor(href) {
    return BLURBS[normalizePath(href)] || "Part of the connected documentation set";
  }

  var path = currentPage();
  var isUiDoc = path.indexOf("/ui-") === 0 || path === "/workbench";

  var nav = document.querySelector(".nav");
  if (nav) {
    nav.innerHTML =
      PAGES.map(function (p) {
        var cur =
          p.href === path || (p.href === "/workbench" && isUiDoc)
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
})();
