#!/usr/bin/env python3
"""
Data pipeline instrumentation example for 254Carbon

This example simulates a streaming ingestion/normalization pipeline with rich
observability:
- Traces wrap Kafka consume/produce, DB access, cache lookups, and processing.
- Metrics expose RED-style counters and latency histograms for key stages.
- Logs are structured JSON correlated with traces via `trace_id`/`span_id`.

Guiding principles:
- Keep metric label cardinality low: use operation/table/topic rather than IDs.
- Log details in the event payload, not as metric labels, to control cost.
- Ensure every error path records exceptions on the active span and emits
  an error log with minimal, actionable context.
"""

import time
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

import aiokafka
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
import asyncpg
import redis.asyncio as redis
from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize OpenTelemetry
def setup_observability():
    """Initialize OpenTelemetry tracing and metrics.

    - Spans exported to the OpenTelemetry Collector via OTLP/gRPC
    - Metrics exported via OTLP/HTTP using a periodic reader
    - Auto-instrumentation enabled for Redis and AsyncPG
    """
    
    # Configure tracing
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)
    
    # Configure OTLP trace exporter
    otlp_trace_exporter = OTLPSpanExporter(endpoint="http://otel-collector:4317")
    span_processor = BatchSpanProcessor(otlp_trace_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)
    
    # Configure metrics
    otlp_metric_exporter = OTLPMetricExporter(endpoint="http://otel-collector:4318")
    metric_reader = PeriodicExportingMetricReader(otlp_metric_exporter)
    meter_provider = MeterProvider(metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)
    
    # Auto-instrument Redis and PostgreSQL
    RedisInstrumentor().instrument()
    AsyncPGInstrumentor().instrument()
    
    return tracer, metrics.get_meter(__name__)

# Initialize observability
tracer, meter = setup_observability()

# Create metrics
# Names follow a stable convention to make dashboards and alerts predictable.
records_processed = meter.create_counter(
    name="ingestion_records_processed_total",
    description="Total records processed",
    unit="1"
)

processing_duration = meter.create_histogram(
    name="ingestion_processing_duration_seconds",
    description="Record processing duration",
    unit="s"
)

connector_runs = meter.create_counter(
    name="ingestion_connector_runs_total",
    description="Total connector runs",
    unit="1"
)

connector_duration = meter.create_histogram(
    name="ingestion_connector_run_duration_seconds",
    description="Connector run duration",
    unit="s"
)

kafka_messages_consumed = meter.create_counter(
    name="kafka_messages_consumed_total",
    description="Total Kafka messages consumed",
    unit="1"
)

kafka_messages_produced = meter.create_counter(
    name="kafka_messages_produced_total",
    description="Total Kafka messages produced",
    unit="1"
)

db_queries = meter.create_counter(
    name="database_queries_total",
    description="Total database queries",
    unit="1"
)

db_query_duration = meter.create_histogram(
    name="database_query_duration_seconds",
    description="Database query duration",
    unit="s"
)

cache_hits = meter.create_counter(
    name="cache_hits_total",
    description="Total cache hits",
    unit="1"
)

cache_misses = meter.create_counter(
    name="cache_misses_total",
    description="Total cache misses",
    unit="1"
)

# Data models
@dataclass
class MarketTick:
    instrument_id: str
    price: float
    volume: int
    timestamp: datetime
    market: str

@dataclass
class NormalizedTick:
    instrument_id: str
    price: float
    volume: int
    timestamp: datetime
    market: str
    normalized_price: float
    currency: str

# Structured logging function
def log_with_trace(level: str, message: str, **kwargs):
    """Emit a structured JSON log and enrich with active span context.

    Keep payloads concise; put identifiers that help correlate evolution over
    time (e.g., connector, operation) and avoid PII or high-cardinality fields.
    """
    span = trace.get_current_span()
    
    log_data = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "level": level.upper(),
        "service": "254carbon-ingestion",
        "message": message,
        **kwargs
    }
    
    # Add trace context if available
    if span and span.is_recording():
        span_context = span.get_span_context()
        log_data.update({
            "trace_id": format(span_context.trace_id, '032x'),
            "span_id": format(span_context.span_id, '016x'),
        })
    
    logger.info(json.dumps(log_data))

# Database operations
class DatabaseManager:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool = None
    
    async def initialize(self):
        """Initialize database connection pool.

        Uses a bounded pool (min:5, max:20) to avoid thundering herds and to
        set predictable limits for DB resource planning.
        """
        with tracer.start_as_current_span("db.initialize") as span:
            try:
                self.pool = await asyncpg.create_pool(
                    self.connection_string,
                    min_size=5,
                    max_size=20
                )
                
                span.set_attributes({
                    "db.operation": "initialize",
                    "db.pool_size": 20,
                    "operation.success": True
                })
                
                log_with_trace("info", "Database pool initialized",
                             operation="db.initialize",
                             pool_size=20,
                             event_type="business_event")
                
            except Exception as e:
                span.set_attributes({
                    "db.operation": "initialize",
                    "operation.success": False,
                    "error.type": type(e).__name__,
                    "error.message": str(e)
                })
                
                log_with_trace("error", "Failed to initialize database pool",
                             operation="db.initialize",
                             error_type=type(e).__name__,
                             error_message=str(e),
                             event_type="error")
                
                raise
    
    async def insert_tick(self, tick: NormalizedTick):
        """Insert a normalized tick into the database.

        - Records span attributes for table/operation and a duration metric
        - Avoids putting the full row into labels; logs carry details
        """
        with tracer.start_as_current_span("db.insert_tick") as span:
            start_time = time.time()
            
            span.set_attributes({
                "db.operation": "insert",
                "db.table": "normalized_ticks",
                "instrument.id": tick.instrument_id,
                "instrument.price": tick.price
            })
            
            try:
                async with self.pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO normalized_ticks 
                        (instrument_id, price, volume, timestamp, market, normalized_price, currency)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """, tick.instrument_id, tick.price, tick.volume, tick.timestamp,
                    tick.market, tick.normalized_price, tick.currency)
                
                duration = time.time() - start_time
                
                span.set_attributes({
                    "operation.success": True,
                    "db.query.duration": duration
                })
                
                # Record metrics
                db_queries.add(1, {
                    "operation": "insert",
                    "table": "normalized_ticks",
                    "service": "ingestion"
                })
                
                db_query_duration.record(duration, {
                    "operation": "insert",
                    "table": "normalized_ticks",
                    "service": "ingestion"
                })
                
                log_with_trace("info", "Tick inserted successfully",
                             operation="db.insert_tick",
                             instrument_id=tick.instrument_id,
                             price=tick.price,
                             duration_ms=duration*1000,
                             event_type="business_event")
                
            except Exception as e:
                duration = time.time() - start_time
                
                span.set_attributes({
                    "operation.success": False,
                    "error.type": type(e).__name__,
                    "error.message": str(e),
                    "db.query.duration": duration
                })
                
                log_with_trace("error", "Failed to insert tick",
                             operation="db.insert_tick",
                             instrument_id=tick.instrument_id,
                             error_type=type(e).__name__,
                             error_message=str(e),
                             duration_ms=duration*1000,
                             event_type="error")
                
                span.record_exception(e)
                raise
    
    async def get_instrument_reference(self, instrument_id: str) -> Optional[Dict[str, Any]]:
        """Fetch instrument reference data by ID.

        Returns None when the instrument does not exist, logging a WARN rather
        than error to reflect expected absence in some flows.
        """
        with tracer.start_as_current_span("db.get_instrument_reference") as span:
            start_time = time.time()
            
            span.set_attributes({
                "db.operation": "select",
                "db.table": "instruments",
                "instrument.id": instrument_id
            })
            
            try:
                async with self.pool.acquire() as conn:
                    row = await conn.fetchrow("""
                        SELECT id, name, type, currency, market
                        FROM instruments
                        WHERE id = $1
                    """, instrument_id)
                
                duration = time.time() - start_time
                
                if row:
                    instrument_data = dict(row)
                    
                    span.set_attributes({
                        "operation.success": True,
                        "db.query.duration": duration,
                        "instrument.name": instrument_data.get("name", ""),
                        "instrument.type": instrument_data.get("type", "")
                    })
                    
                    # Record metrics
                    db_queries.add(1, {
                        "operation": "select",
                        "table": "instruments",
                        "service": "ingestion"
                    })
                    
                    db_query_duration.record(duration, {
                        "operation": "select",
                        "table": "instruments",
                        "service": "ingestion"
                    })
                    
                    log_with_trace("info", "Instrument reference retrieved",
                                 operation="db.get_instrument_reference",
                                 instrument_id=instrument_id,
                                 instrument_name=instrument_data.get("name", ""),
                                 duration_ms=duration*1000,
                                 event_type="business_event")
                    
                    return instrument_data
                else:
                    span.set_attributes({
                        "operation.success": False,
                        "db.query.duration": duration,
                        "error.type": "NotFoundError",
                        "error.message": "Instrument not found"
                    })
                    
                    log_with_trace("warn", "Instrument reference not found",
                                 operation="db.get_instrument_reference",
                                 instrument_id=instrument_id,
                                 duration_ms=duration*1000,
                                 event_type="business_event")
                    
                    return None
                    
            except Exception as e:
                duration = time.time() - start_time
                
                span.set_attributes({
                    "operation.success": False,
                    "error.type": type(e).__name__,
                    "error.message": str(e),
                    "db.query.duration": duration
                })
                
                log_with_trace("error", "Failed to get instrument reference",
                             operation="db.get_instrument_reference",
                             instrument_id=instrument_id,
                             error_type=type(e).__name__,
                             error_message=str(e),
                             duration_ms=duration*1000,
                             event_type="error")
                
                span.record_exception(e)
                raise

# Cache operations
class CacheManager:
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis = None
    
    async def initialize(self):
        """Initialize Redis connection"""
        with tracer.start_as_current_span("cache.initialize") as span:
            try:
                self.redis = redis.from_url(self.redis_url)
                
                # Test connection
                await self.redis.ping()
                
                span.set_attributes({
                    "cache.operation": "initialize",
                    "cache.type": "redis",
                    "operation.success": True
                })
                
                log_with_trace("info", "Cache connection initialized",
                             operation="cache.initialize",
                             cache_type="redis",
                             event_type="business_event")
                
            except Exception as e:
                span.set_attributes({
                    "cache.operation": "initialize",
                    "operation.success": False,
                    "error.type": type(e).__name__,
                    "error.message": str(e)
                })
                
                log_with_trace("error", "Failed to initialize cache connection",
                             operation="cache.initialize",
                             error_type=type(e).__name__,
                             error_message=str(e),
                             event_type="error")
                
                raise
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from cache"""
        with tracer.start_as_current_span("cache.get") as span:
            span.set_attributes({
                "cache.operation": "get",
                "cache.key": key
            })
            
            try:
                value = await self.redis.get(key)
                
                if value:
                    span.set_attributes({
                        "operation.success": True,
                        "cache.hit": True
                    })
                    
                    # Record cache hit
                    cache_hits.add(1, {
                        "operation": "get",
                        "service": "ingestion"
                    })
                    
                    log_with_trace("info", "Cache hit",
                                 operation="cache.get",
                                 cache_key=key,
                                 event_type="business_event")
                    
                    return value
                else:
                    span.set_attributes({
                        "operation.success": True,
                        "cache.hit": False
                    })
                    
                    # Record cache miss
                    cache_misses.add(1, {
                        "operation": "get",
                        "service": "ingestion"
                    })
                    
                    log_with_trace("info", "Cache miss",
                                 operation="cache.get",
                                 cache_key=key,
                                 event_type="business_event")
                    
                    return None
                    
            except Exception as e:
                span.set_attributes({
                    "operation.success": False,
                    "error.type": type(e).__name__,
                    "error.message": str(e)
                })
                
                log_with_trace("error", "Cache get operation failed",
                             operation="cache.get",
                             cache_key=key,
                             error_type=type(e).__name__,
                             error_message=str(e),
                             event_type="error")
                
                span.record_exception(e)
                raise
    
    async def set(self, key: str, value: str, ttl: int = 3600):
        """Set value in cache"""
        with tracer.start_as_current_span("cache.set") as span:
            span.set_attributes({
                "cache.operation": "set",
                "cache.key": key,
                "cache.ttl": ttl
            })
            
            try:
                await self.redis.setex(key, ttl, value)
                
                span.set_attributes({
                    "operation.success": True
                })
                
                log_with_trace("info", "Cache set operation successful",
                             operation="cache.set",
                             cache_key=key,
                             cache_ttl=ttl,
                             event_type="business_event")
                
            except Exception as e:
                span.set_attributes({
                    "operation.success": False,
                    "error.type": type(e).__name__,
                    "error.message": str(e)
                })
                
                log_with_trace("error", "Cache set operation failed",
                             operation="cache.set",
                             cache_key=key,
                             error_type=type(e).__name__,
                             error_message=str(e),
                             event_type="error")
                
                span.record_exception(e)
                raise

# Kafka operations
class KafkaManager:
    def __init__(self, bootstrap_servers: str):
        self.bootstrap_servers = bootstrap_servers
        self.consumer = None
        self.producer = None
    
    async def initialize(self):
        """Initialize Kafka consumer and producer"""
        with tracer.start_as_current_span("kafka.initialize") as span:
            try:
                # Initialize consumer
                self.consumer = AIOKafkaConsumer(
                    'market.ticks.raw',
                    bootstrap_servers=self.bootstrap_servers,
                    group_id='ingestion-group',
                    auto_offset_reset='latest'
                )
                
                # Initialize producer
                self.producer = AIOKafkaProducer(
                    bootstrap_servers=self.bootstrap_servers
                )
                
                # Start consumer and producer
                await self.consumer.start()
                await self.producer.start()
                
                span.set_attributes({
                    "kafka.operation": "initialize",
                    "kafka.consumer.group": "ingestion-group",
                    "kafka.consumer.topic": "market.ticks.raw",
                    "operation.success": True
                })
                
                log_with_trace("info", "Kafka consumer and producer initialized",
                             operation="kafka.initialize",
                             consumer_group="ingestion-group",
                             consumer_topic="market.ticks.raw",
                             event_type="business_event")
                
            except Exception as e:
                span.set_attributes({
                    "kafka.operation": "initialize",
                    "operation.success": False,
                    "error.type": type(e).__name__,
                    "error.message": str(e)
                })
                
                log_with_trace("error", "Failed to initialize Kafka",
                             operation="kafka.initialize",
                             error_type=type(e).__name__,
                             error_message=str(e),
                             event_type="error")
                
                raise
    
    async def consume_messages(self):
        """Consume messages from Kafka"""
        with tracer.start_as_current_span("kafka.consume") as span:
            span.set_attributes({
                "kafka.operation": "consume",
                "kafka.topic": "market.ticks.raw"
            })
            
            try:
                async for message in self.consumer:
                    with tracer.start_as_current_span("kafka.message.process") as msg_span:
                        msg_span.set_attributes({
                            "kafka.topic": message.topic,
                            "kafka.partition": message.partition,
                            "kafka.offset": message.offset,
                            "kafka.timestamp": message.timestamp
                        })
                        
                        try:
                            # Parse message
                            tick_data = json.loads(message.value.decode())
                            
                            # Create MarketTick object
                            tick = MarketTick(
                                instrument_id=tick_data["instrument_id"],
                                price=tick_data["price"],
                                volume=tick_data["volume"],
                                timestamp=datetime.fromisoformat(tick_data["timestamp"]),
                                market=tick_data["market"]
                            )
                            
                            msg_span.set_attributes({
                                "instrument.id": tick.instrument_id,
                                "instrument.price": tick.price,
                                "instrument.volume": tick.volume,
                                "instrument.market": tick.market
                            })
                            
                            # Process the tick
                            await self.process_tick(tick)
                            
                            # Record metrics
                            kafka_messages_consumed.add(1, {
                                "topic": message.topic,
                                "service": "ingestion"
                            })
                            
                            log_with_trace("info", "Message processed successfully",
                                         operation="kafka.message.process",
                                         instrument_id=tick.instrument_id,
                                         price=tick.price,
                                         market=tick.market,
                                         event_type="business_event")
                            
                        except Exception as e:
                            msg_span.set_attributes({
                                "error": True,
                                "error.type": type(e).__name__,
                                "error.message": str(e)
                            })
                            
                            log_with_trace("error", "Failed to process message",
                                         operation="kafka.message.process",
                                         error_type=type(e).__name__,
                                         error_message=str(e),
                                         event_type="error")
                            
                            msg_span.record_exception(e)
                            
            except Exception as e:
                span.set_attributes({
                    "error": True,
                    "error.type": type(e).__name__,
                    "error.message": str(e)
                })
                
                log_with_trace("error", "Failed to consume messages",
                             operation="kafka.consume",
                             error_type=type(e).__name__,
                             error_message=str(e),
                             event_type="error")
                
                span.record_exception(e)
                raise
    
    async def produce_message(self, topic: str, message: Dict[str, Any]):
        """Produce message to Kafka"""
        with tracer.start_as_current_span("kafka.produce") as span:
            span.set_attributes({
                "kafka.operation": "produce",
                "kafka.topic": topic
            })
            
            try:
                # Serialize message
                message_bytes = json.dumps(message).encode()
                
                # Send message
                await self.producer.send_and_wait(topic, message_bytes)
                
                span.set_attributes({
                    "operation.success": True,
                    "kafka.message.size": len(message_bytes)
                })
                
                # Record metrics
                kafka_messages_produced.add(1, {
                    "topic": topic,
                    "service": "ingestion"
                })
                
                log_with_trace("info", "Message produced successfully",
                             operation="kafka.produce",
                             topic=topic,
                             message_size=len(message_bytes),
                             event_type="business_event")
                
            except Exception as e:
                span.set_attributes({
                    "operation.success": False,
                    "error.type": type(e).__name__,
                    "error.message": str(e)
                })
                
                log_with_trace("error", "Failed to produce message",
                             operation="kafka.produce",
                             topic=topic,
                             error_type=type(e).__name__,
                             error_message=str(e),
                             event_type="error")
                
                span.record_exception(e)
                raise
    
    async def process_tick(self, tick: MarketTick):
        """Process a market tick"""
        with tracer.start_as_current_span("process_tick") as span:
            start_time = time.time()
            
            span.set_attributes({
                "operation": "process_tick",
                "instrument.id": tick.instrument_id,
                "instrument.price": tick.price,
                "instrument.volume": tick.volume,
                "instrument.market": tick.market
            })
            
            try:
                # Get instrument reference data
                instrument_ref = await db_manager.get_instrument_reference(tick.instrument_id)
                
                if not instrument_ref:
                    # Create default reference data
                    instrument_ref = {
                        "id": tick.instrument_id,
                        "name": f"{tick.instrument_id} Corporation",
                        "type": "equity",
                        "currency": "USD",
                        "market": tick.market
                    }
                
                # Normalize price (simplified example)
                normalized_price = tick.price
                if instrument_ref["currency"] != "USD":
                    # Simple currency conversion (in real implementation, use proper rates)
                    normalized_price = tick.price * 1.2  # Mock conversion rate
                
                # Create normalized tick
                normalized_tick = NormalizedTick(
                    instrument_id=tick.instrument_id,
                    price=tick.price,
                    volume=tick.volume,
                    timestamp=tick.timestamp,
                    market=tick.market,
                    normalized_price=normalized_price,
                    currency=instrument_ref["currency"]
                )
                
                # Store in database
                await db_manager.insert_tick(normalized_tick)
                
                # Produce normalized tick
                await self.produce_message("market.ticks.normalized", {
                    "instrument_id": normalized_tick.instrument_id,
                    "price": normalized_tick.price,
                    "volume": normalized_tick.volume,
                    "timestamp": normalized_tick.timestamp.isoformat(),
                    "market": normalized_tick.market,
                    "normalized_price": normalized_tick.normalized_price,
                    "currency": normalized_tick.currency
                })
                
                duration = time.time() - start_time
                
                span.set_attributes({
                    "operation.success": True,
                    "processing.duration": duration,
                    "normalized.price": normalized_price,
                    "instrument.currency": instrument_ref["currency"]
                })
                
                # Record metrics
                records_processed.add(1, {
                    "instrument": tick.instrument_id,
                    "market": tick.market,
                    "service": "ingestion"
                })
                
                processing_duration.record(duration, {
                    "instrument": tick.instrument_id,
                    "market": tick.market,
                    "service": "ingestion"
                })
                
                log_with_trace("info", "Tick processed successfully",
                             operation="process_tick",
                             instrument_id=tick.instrument_id,
                             price=tick.price,
                             normalized_price=normalized_price,
                             duration_ms=duration*1000,
                             event_type="business_event")
                
            except Exception as e:
                duration = time.time() - start_time
                
                span.set_attributes({
                    "operation.success": False,
                    "error.type": type(e).__name__,
                    "error.message": str(e),
                    "processing.duration": duration
                })
                
                log_with_trace("error", "Failed to process tick",
                             operation="process_tick",
                             instrument_id=tick.instrument_id,
                             error_type=type(e).__name__,
                             error_message=str(e),
                             duration_ms=duration*1000,
                             event_type="error")
                
                span.record_exception(e)
                raise

# Main application
class DataPipelineApp:
    def __init__(self):
        self.db_manager = DatabaseManager("postgresql://user:password@localhost:5432/market_data")
        self.cache_manager = CacheManager("redis://localhost:6379")
        self.kafka_manager = KafkaManager("localhost:9092")
        self.running = False
    
    async def initialize(self):
        """Initialize all components"""
        with tracer.start_as_current_span("app.initialize") as span:
            try:
                # Initialize components
                await self.db_manager.initialize()
                await self.cache_manager.initialize()
                await self.kafka_manager.initialize()
                
                span.set_attributes({
                    "operation": "initialize",
                    "operation.success": True
                })
                
                log_with_trace("info", "Data pipeline application initialized",
                             operation="app.initialize",
                             event_type="business_event")
                
            except Exception as e:
                span.set_attributes({
                    "operation": "initialize",
                    "operation.success": False,
                    "error.type": type(e).__name__,
                    "error.message": str(e)
                })
                
                log_with_trace("error", "Failed to initialize application",
                             operation="app.initialize",
                             error_type=type(e).__name__,
                             error_message=str(e),
                             event_type="error")
                
                span.record_exception(e)
                raise
    
    async def run(self):
        """Run the data pipeline"""
        with tracer.start_as_current_span("app.run") as span:
            try:
                self.running = True
                
                span.set_attributes({
                    "operation": "run",
                    "operation.success": True
                })
                
                log_with_trace("info", "Data pipeline started",
                             operation="app.run",
                             event_type="business_event")
                
                # Start consuming messages
                await self.kafka_manager.consume_messages()
                
            except Exception as e:
                span.set_attributes({
                    "operation": "run",
                    "operation.success": False,
                    "error.type": type(e).__name__,
                    "error.message": str(e)
                })
                
                log_with_trace("error", "Data pipeline failed",
                             operation="app.run",
                             error_type=type(e).__name__,
                             error_message=str(e),
                             event_type="error")
                
                span.record_exception(e)
                raise
    
    async def shutdown(self):
        """Shutdown the application"""
        with tracer.start_as_current_span("app.shutdown") as span:
            try:
                self.running = False
                
                # Close Kafka connections
                if self.kafka_manager.consumer:
                    await self.kafka_manager.consumer.stop()
                if self.kafka_manager.producer:
                    await self.kafka_manager.producer.stop()
                
                # Close database pool
                if self.db_manager.pool:
                    await self.db_manager.pool.close()
                
                # Close Redis connection
                if self.cache_manager.redis:
                    await self.cache_manager.redis.close()
                
                span.set_attributes({
                    "operation": "shutdown",
                    "operation.success": True
                })
                
                log_with_trace("info", "Data pipeline shutdown completed",
                             operation="app.shutdown",
                             event_type="business_event")
                
            except Exception as e:
                span.set_attributes({
                    "operation": "shutdown",
                    "operation.success": False,
                    "error.type": type(e).__name__,
                    "error.message": str(e)
                })
                
                log_with_trace("error", "Failed to shutdown application",
                             operation="app.shutdown",
                             error_type=type(e).__name__,
                             error_message=str(e),
                             event_type="error")
                
                span.record_exception(e)
                raise

# Global instances
db_manager = DatabaseManager("postgresql://user:password@localhost:5432/market_data")
cache_manager = CacheManager("redis://localhost:6379")
kafka_manager = KafkaManager("localhost:9092")

# Main function
async def main():
    """Main function"""
    app = DataPipelineApp()
    
    try:
        # Initialize application
        await app.initialize()
        
        # Run application
        await app.run()
        
    except KeyboardInterrupt:
        print("Shutting down...")
    except Exception as e:
        print(f"Application failed: {e}")
    finally:
        # Shutdown application
        await app.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
