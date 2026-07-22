# Angular workbench → nginx (proxies /v1 and /health to Compose service `api`).
# Local:      docker compose up --build  → ehr-risk-web:local
# Docker Hub: ranasl62/ehr-risk-web
#
# Optional remote API (bake at image build):
#   docker build --build-arg API_ENDPOINT=https://ehr-risk-framework.onrender.com -t ehr-risk-web ./web
# Leave unset for same-origin nginx → Compose `api`.

FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .

ARG API_ENDPOINT=
ENV API_ENDPOINT=$API_ENDPOINT

RUN npm run build -- --configuration production

FROM nginx:1.27-alpine

LABEL org.opencontainers.image.title="ehr-risk-web" \
      org.opencontainers.image.description="EHR Risk Framework Angular workbench (nginx)" \
      org.opencontainers.image.source="https://github.com/ranasl62/ehr-chronic-disease-risk-prediction" \
      org.opencontainers.image.licenses="MIT"

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist/web/browser /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
  CMD wget -qO- http://127.0.0.1/health || exit 1
