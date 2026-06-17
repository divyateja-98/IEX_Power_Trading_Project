FROM node:20-alpine AS build

WORKDIR /app

ARG VITE_FASTAPI_BASE_URL=http://127.0.0.1:8002
ENV VITE_FASTAPI_BASE_URL=$VITE_FASTAPI_BASE_URL

COPY frontend/package*.json ./
RUN npm install

COPY frontend/ ./
RUN npm run build

FROM nginx:1.27-alpine

COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD wget --quiet --tries=1 --spider http://127.0.0.1:8080/ || exit 1
