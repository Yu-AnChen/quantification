FROM ghcr.io/prefix-dev/pixi:0.63.2-noble AS build

WORKDIR /app
COPY . .
RUN pixi install --locked
# Replace editable install with a proper wheel so the production stage
# doesn't need the source tree (editable installs write a .pth back to /app/src).
RUN pixi run pip install --no-deps --force-reinstall .
RUN pixi shell-hook -s bash > /shell-hook
RUN echo "#!/bin/bash" > /app/entrypoint.sh
RUN cat /shell-hook >> /app/entrypoint.sh
RUN echo 'exec "$@"' >> /app/entrypoint.sh

FROM ubuntu:24.04 AS production
WORKDIR /app
COPY --from=build /app/.pixi/envs/default /app/.pixi/envs/default
COPY --from=build --chmod=0755 /app/entrypoint.sh /app/entrypoint.sh

ENTRYPOINT [ "/app/entrypoint.sh" ]
CMD [ "chunkprop", "--help" ]
