/**
 * On-page SEO helpers for the documentation site (ehr.larucare.com).
 * Loaded after site.js from each page (or inject via site.js).
 */
(function () {
  var SITE_ORIGIN = "https://ehr.larucare.com";
  var SITE_NAME = "EHR Risk Framework";
  var CATCHPHRASE =
    "An open-source framework for leakage-safe, calibrated, and explainable EHR risk prediction";
  var GITHUB = "https://github.com/ranasl62/ehr-chronic-disease-risk-prediction";
  var OG_IMAGE = SITE_ORIGIN + "/assets/og-card.png";

  /** Unique title + description per path (path without trailing slash, "/" for home). */
  var PAGE_META = {
    "/": {
      title: "EHR Risk Framework — Leakage-Safe Clinical Machine Learning Workbench",
      description:
        "Open-source EHR risk prediction for research: leakage-aware splits, calibration (Brier/ECE), SHAP explainability, Docker workbench. Not a medical device.",
    },
    "/why": {
      title: "Why Leakage-Safe EHR Risk Prediction Matters — EHR Risk Framework",
      description:
        "Why clinical machine learning needs index-time integrity, calibration, and explainability—and how this open workbench helps labs and students.",
    },
    "/features": {
      title: "Features — Leakage Audit, Calibration, SHAP & Docker Workbench",
      description:
        "Capability catalog for EHR risk modeling: ingest, temporal splits, leakage audit, isotonic calibration, fairness jobs, FastAPI, and Angular UI.",
    },
    "/workbench": {
      title: "Research Workbench UI Tour — EHR Risk Framework",
      description:
        "Tour the Angular researcher workbench for EHR risk prediction: datasets, train, results, predict, config, and OpenAPI.",
    },
    "/research-workflow": {
      title: "Research Workflow — Methods-Style EHR Risk Study Loop",
      description:
        "End-to-end research workflow: health gate, train, trust pack, leakage audit, external validation, curves, and ZIP export.",
    },
    "/guide": {
      title: "How it Works A–Z — EHR Risk Workbench Guide with Screenshots",
      description:
        "Complete A–Z walkthrough: Docker start, datasets, train, leakage audit, results, SHAP, predict — with screenshots. Research/education only.",
    },
    "/quickstart": {
      title: "Quickstart: Train an EHR Risk Model in Minutes (Docker)",
      description:
        "Docker Compose quickstart for leakage-aware EHR risk prediction—train, audit, and review calibrated metrics on synthetic teaching data.",
    },
    "/docker-for-beginners": {
      title: "Docker for Beginners — Run the EHR Risk Framework",
      description:
        "Plain-language Docker installation and Docker Compose guide for running the EHR Risk Framework API and web workbench.",
    },
    "/docker-images": {
      title: "Docker Hub Images for EHR Risk Prediction — API & Web",
      description:
        "Pull ranasl62/ehr-risk-api and ehr-risk-web for a one-command clinical ML research stack.",
    },
    "/commands": {
      title: "CLI & Make Commands for EHR Risk Modeling",
      description:
        "Copy-paste recipes: ehr-ai CLI, Make targets, curl jobs, leakage audit, and research-paper locks.",
    },
    "/diagrams": {
      title: "Architecture Diagrams — Temporal EHR Risk Pipelines",
      description:
        "SVG and ASCII diagrams for leakage-aware longitudinal EHR risk modeling and the research workbench stack.",
    },
    "/fine-tuning": {
      title: "Fine-Tuning & Model Comparison for EHR Risk Models",
      description:
        "Compare models, light HPO, promote runs, and iterate on calibrated EHR risk prediction experiments.",
    },
    "/architecture": {
      title: "Software Architecture — Open EHR Risk Prediction Framework",
      description:
        "Components of the EHR Risk Framework: openhealth package, FastAPI jobs, Angular UI, task YAML, and reports.",
    },
    "/data": {
      title: "EHR Data Guide — Schema, Demo CSVs & Index/Horizon Rules",
      description:
        "How to format longitudinal EHR-style CSVs for leakage-safe risk prediction; teaching demos under data/demo/.",
    },
    "/api": {
      title: "Researcher API — Train, Predict & Audit Jobs",
      description:
        "FastAPI endpoints for EHR risk prediction research: datasets, training jobs, leakage audit, fairness, and reports.",
    },
    "/limits": {
      title: "Limitations & Model Card — Research-Only EHR Risk Software",
      description:
        "Honest non-goals: not a medical device, not clinical AutoML. Model card pointers for leakage-aware EHR risk research.",
    },
    "/help": {
      title: "Help Library — EHR Risk Framework Documentation",
      description:
        "Full documentation library for leakage-safe EHR risk prediction: guides, examples, and intended use.",
    },
    "/sitemap": {
      title: "Documentation Sitemap — EHR Risk Framework",
      description: "Complete map of documentation pages for the open-source EHR risk prediction workbench.",
    },
    "/cite": {
      title: "Cite the EHR Risk Framework — DOI & Feedback",
      description:
        "Citation (CITATION.cff, Zenodo DOI) and feedback for the open-source leakage-aware EHR risk modeling software.",
    },
    "/blog": {
      title: "Blog — Clinical Machine Learning & Leakage-Safe EHR Risk",
      description:
        "Tutorials on preventing data leakage in clinical AI, EHR risk prediction quickstarts, and open research tooling.",
    },
    "/blog/prevent-data-leakage-clinical-ai": {
      title: "How to Prevent Data Leakage in Clinical AI (EHR Risk Models)",
      description:
        "Practical guide to index-time integrity, temporal splits, and leakage audits for EHR risk prediction research.",
    },
    "/blog/why-clinical-ai-label-leakage": {
      title: "Why Many Same-Admission AI Models Look Strong — Label Leakage (JAMA 2025)",
      description:
        "JAMA Network Open found 40.2% of reviewed MIMIC same-admission studies used post-discharge ICD codes as features. Leakage-aware research methods.",
    },
    "/blog/onc-safer-guides-2025-ai": {
      title: "2025 ONC SAFER Guides and AI Transparency — Research Methods Notes",
      description:
        "ONC SAFER Guides emphasize AI inventory, human oversight, and monitoring. How research tooling supports transparency practice.",
    },
    "/blog/fda-ai-devices-2025": {
      title: "FDA AI/ML Device Clearances in 2025 — What Research Teams Should Learn",
      description:
        "Record AI/ML device clearances in 2025. Research software is not an FDA pathway; methods hygiene still matters.",
    },
    "/blog/healthcare-ai-market-quality-gap": {
      title: "Healthcare AI Market Growth vs Methods Quality Gap",
      description:
        "Market forecasts surge while pilots fail to scale. Leakage, calibration, and explainability remain bottlenecks.",
    },
    "/blog/nih-all-of-us-reproducible-ai": {
      title: "NIH All of Us and Reproducible Leakage-Safe ML Methods",
      description:
        "All of Us enables large-scale EHR research; predictive ML still needs index-time integrity and reproducible tooling.",
    },
    "/blog/calibration-gap-brier-ece": {
      title: "Calibration Gap — Brier Score and ECE for EHR Risk Models",
      description:
        "Why AUROC is not enough. Brier, ECE, and isotonic calibration for honest probability reporting.",
    },
    "/blog/hhs-ai-strategy-2025": {
      title: "HHS 2025 AI Strategic Plan — Trustworthy Reproducible Health AI",
      description:
        "HHS AI plan goals: innovation, trust, access, workforce. Aligning open research workbenches with trustworthy methods.",
    },
    "/blog/ahrq-patient-safety-ai": {
      title: "AHRQ Digital Healthcare Research and Evidence-Based AI Methods",
      description:
        "AHRQ digital healthcare research and AI safety questions—methods infrastructure for honest evaluation.",
    },
    "/blog/pcori-ai-methods-infrastructure": {
      title: "PCORI AI/ML Methods for CER — Shared Open Infrastructure",
      description:
        "PCORI funds AI/ML methods for patient-centered CER; open leakage-safe tooling helps labs execute.",
    },
    "/blog/chronic-disease-trustworthy-risk-ai": {
      title: "Chronic Disease Burden and Trustworthy Risk Prediction Research",
      description:
        "CDC chronic disease burden meets leakage-safe, calibrated, explainable EHR risk modeling for research.",
    },
    "/blog/ehr-risk-prediction-quickstart": {
      title: "How to Build an EHR Risk Prediction Model (Open-Source Quickstart)",
      description:
        "Step-by-step tutorial: Docker workbench, teaching CSV, train a leakage-aware risk model, review calibration and SHAP.",
    },
    "/compare/vs-ad-hoc-notebooks": {
      title: "EHR Risk Framework vs Ad-Hoc Clinical ML Notebooks",
      description:
        "Compare a leakage-aware open workbench to one-off notebooks for EHR risk prediction research and teaching.",
    },
    "/alternatives/opaque-clinical-automl": {
      title: "Alternatives to Opaque Clinical AutoML for Research Labs",
      description:
        "Open, auditable alternatives for EHR risk prediction: leakage audits, calibration, and explainability without black-box AutoML.",
    },
    "/listicles/open-source-clinical-risk-tools": {
      title: "Open-Source Tools for Clinical Risk Prediction Research",
      description:
        "A research-focused list of open tools for EHR risk modeling—including the EHR Risk Framework—with honest scope notes.",
    },
  };

  function normalizePath(pathname) {
    var path = (pathname || "/").replace(/\/index\.html$/i, "").replace(/\.html$/i, "");
    if (!path || path === "/") return "/";
    return path.replace(/\/+$/, "") || "/";
  }

  function upsertMeta(attr, key, content) {
    if (!content) return;
    var sel = attr === "property" ? 'meta[property="' + key + '"]' : 'meta[name="' + key + '"]';
    var el = document.head.querySelector(sel);
    if (!el) {
      el = document.createElement("meta");
      el.setAttribute(attr, key);
      document.head.appendChild(el);
    }
    el.setAttribute("content", content);
  }

  function upsertLink(rel, href) {
    var el = document.head.querySelector('link[rel="' + rel + '"]');
    if (!el) {
      el = document.createElement("link");
      el.setAttribute("rel", rel);
      document.head.appendChild(el);
    }
    el.setAttribute("href", href);
  }

  var path = normalizePath(location.pathname);
  var meta = PAGE_META[path];
  var canonical = SITE_ORIGIN + (path === "/" ? "/" : path + "/");
  var title = meta ? meta.title : document.title || SITE_NAME;
  var description =
    meta && meta.description
      ? meta.description
      : (
          document.querySelector('meta[name="description"]') || { content: CATCHPHRASE }
        ).content;

  if (meta) {
    document.title = title;
    upsertMeta("name", "description", description);
  }

  upsertMeta("name", "keywords", [
    "EHR risk prediction",
    "clinical machine learning",
    "leakage-safe AI",
    "electronic health records",
    "calibration ECE Brier",
    "SHAP explainability",
    "temporal validation",
    "open source healthcare AI",
    "health informatics research",
  ].join(", "));

  upsertMeta("name", "author", "Md Rana Hossain");
  upsertMeta("name", "robots", "index,follow");
  upsertMeta("property", "og:site_name", SITE_NAME);
  upsertMeta("property", "og:title", title);
  upsertMeta("property", "og:description", description);
  upsertMeta("property", "og:type", path.indexOf("/blog/") === 0 ? "article" : "website");
  upsertMeta("property", "og:url", canonical);
  upsertMeta("property", "og:image", OG_IMAGE);
  upsertMeta("name", "twitter:card", "summary_large_image");
  upsertMeta("name", "twitter:title", title);
  upsertMeta("name", "twitter:description", description);
  upsertMeta("name", "twitter:image", OG_IMAGE);
  upsertLink("canonical", canonical);

  // Brand subtitle once
  var brand = document.querySelector(".site-header .brand");
  if (brand && !document.querySelector(".brand-tagline")) {
    var tag = document.createElement("p");
    tag.className = "brand-tagline";
    tag.textContent = CATCHPHRASE;
    var inner = document.querySelector(".site-header .inner");
    if (inner) inner.appendChild(tag);
  }

  // JSON-LD
  if (!document.getElementById("seo-jsonld")) {
    var ld = {
      "@context": "https://schema.org",
      "@graph": [
        {
          "@type": "WebSite",
          "@id": SITE_ORIGIN + "/#website",
          url: SITE_ORIGIN + "/",
          name: SITE_NAME,
          description: CATCHPHRASE,
          publisher: { "@id": SITE_ORIGIN + "/#org" },
        },
        {
          "@type": "SoftwareApplication",
          "@id": SITE_ORIGIN + "/#software",
          name: SITE_NAME,
          applicationCategory: "HealthApplication",
          operatingSystem: "Linux, macOS, Windows (Docker)",
          url: SITE_ORIGIN + "/",
          downloadUrl: GITHUB,
          license: "https://opensource.org/licenses/MIT",
          description: CATCHPHRASE + ". For research and education only. Outputs are not clinical recommendations and are not intended for patient care. We are working toward broader general-purpose use in the future.",
          author: {
            "@type": "Person",
            name: "Md Rana Hossain",
            email: "support@larucare.com",
          },
          offers: {
            "@type": "Offer",
            price: "0",
            priceCurrency: "USD",
          },
        },
        {
          "@type": "Organization",
          "@id": SITE_ORIGIN + "/#org",
          name: "LaruCare / EHR Risk Framework",
          url: SITE_ORIGIN + "/",
          email: "support@larucare.com",
          sameAs: [GITHUB, "https://www.linkedin.com/in/mdranahossain/"],
        },
        {
          "@type": "WebPage",
          "@id": canonical + "#webpage",
          url: canonical,
          name: title,
          description: description,
          isPartOf: { "@id": SITE_ORIGIN + "/#website" },
          about: { "@id": SITE_ORIGIN + "/#software" },
        },
      ],
    };
    var script = document.createElement("script");
    script.type = "application/ld+json";
    script.id = "seo-jsonld";
    script.textContent = JSON.stringify(ld);
    document.head.appendChild(script);
  }

  window.EHR_SEO = { SITE_ORIGIN: SITE_ORIGIN, CATCHPHRASE: CATCHPHRASE, path: path };
})();
