FROM nginx:1.16

COPY deployment/alanno.nginx.conf /etc/nginx/conf.d/default.conf
