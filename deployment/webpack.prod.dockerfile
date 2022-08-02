FROM node:14.15.0

RUN npm install webpack -g
COPY ./server/package.json /package.json
WORKDIR /
RUN npm install

COPY ./server .
CMD ["webpack", "--config", "webpack.config.js"]