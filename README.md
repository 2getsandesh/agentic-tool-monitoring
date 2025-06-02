# Agentic-Application-Monitoring

## Environment Setup
### Virtual Environment
1. Creating - `conda create --name monitoring`. This will create a virtual environment.
2. Activating - `conda activate monitoring`

### Install dependencies
Run `pip install -r requirements.txt`

### Setting up Traceloop
1. Clone the traceloop github : https://github.com/traceloop/openllmetry

Install the required packages
1. Traceloop-SDK
```
cd openllmetry/packages/traceloop-sdk
pip install -e .
cd ..
```
2. MCP Instrumentation
```
cd opentelemetry-instrumentation-mcp
pip install -e .
cd ..
```
3. Scemantic Conventions
```
cd opentelemetry-semantic-conventions-ai
pip install -e .
```

## Start Instana Agent / Jaeger
### Jaeger
```
docker run --rm --name jaeger \
  -e COLLECTOR_ZIPKIN_HOST_PORT=:9411 \
  -p 16686:16686 \
  -p 4317:4317 \
  -p 4318:4318 \
  -p 14250:14250 \
  -p 14268:14268 \
  -p 14269:14269 \
  -p 9411:9411 \
  jaegertracing/all-in-one:1.67.0
```
### Instana
1. Download and unzip Instana agent : https://ibmdevsandbox-instanaibm.instana.io/#/agents/installation
2. Install java 1.8 in system (Instana requirement)
3. Set `JAVA_HOME`
4. Run Instana by `./bin/karaf` You might need to provide access in system settings
5. Configure the port of agent or mcp server by setting logging endpoint to 4317(grpc) or 4318(http)
*Now Instana Agent should be running*

## Setup Environment Variables
```
export TRACELOOP_BASE_URL=localhost:4317
export TRACELOOP_LOGGING_ENABLED=true
export TRACELOOP_LOGGING_ENDPOINT=$TRACELOOP_BASE_URL
export TRACELOOP_METRICS_ENABLED=false
export OTEL_EXPORTER_OTLP_INSECURE=true
```

## Start MCP Server
command : `python weather.py`

## RUN Agent
Set creadentials for `ChatOpenAI` in environment variables.

command : `python agent.py`
