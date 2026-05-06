FROM node:22-alpine
WORKDIR /app
COPY package.json /app/package.json
RUN npm install
COPY index.html /app/index.html
COPY tsconfig.json /app/tsconfig.json
COPY src /app/src
CMD ["npm", "run", "dev"]
