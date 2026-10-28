FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
# HoverFast is installed from its git repository.
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

RUN mkdir /app
WORKDIR /app

# Outside /app, so copying the project below cannot replace it with a host .venv.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
COPY pyproject.toml uv.lock /app/
# torch for a CPU: the cuda flavour installs it for a GPU instead.
RUN uv sync --frozen --no-dev --no-install-project --extra cpu

ENV PATH="/opt/venv/bin:$PATH"

# The whole project: the app target may be any module in it, not only app.py.
COPY . /app
# And the project itself, so `hoverfast_app` (in src/) is importable.
RUN uv sync --frozen --no-dev --extra cpu

# The version of this build, when it is not the one the source declares: a build from
# a branch is passed it by `arkitekt plugin release`. Empty for a release proper.
ARG ARKITEKT_APP_VERSION=""
ENV ARKITEKT_APP_VERSION=$ARKITEKT_APP_VERSION
