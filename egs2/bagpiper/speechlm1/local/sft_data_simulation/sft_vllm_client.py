#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Shared async vLLM client for SFT data simulation."""

import asyncio
import itertools
import json
import re
import time
from typing import Any, Callable, Dict, List, Optional, Set

import aiohttp


def get_processed_indices(output_file: str, idx_key: str = "idx") -> Set[Any]:
    """Get set of indices/IDs already processed in output file.

    Args:
        output_file: Path to the output JSONL file.
        idx_key: Key name for the index field in output records.

    Returns:
        Set of indices/IDs that have been successfully processed.
        Can contain integers or strings depending on the key type.
    """
    processed: Set[Any] = set()
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    if idx_key in record and record[idx_key] is not None:
                        processed.add(record[idx_key])
    except FileNotFoundError:
        pass
    return processed


def parse_vllm_urls(url_string: str) -> List[str]:
    """Parse colon-separated vLLM URLs.

    Args:
        url_string: Single URL or colon-separated URLs.
            Example: "http://host1:8000/v1:http://host2:8000/v1"

    Returns:
        List of parsed URLs.
    """
    # Split by "://" first to identify URL boundaries, then rejoin
    # This handles the case where URLs contain ":" in the port
    urls = []
    current = ""
    for part in url_string.split(":"):
        if part.startswith("//"):
            # This is the start of a new URL (after http or https)
            if current and not current.endswith("http") and \
               not current.endswith("https"):
                urls.append(current.rstrip(":"))
                current = ""
            current += ":" + part
        elif part in ("http", "https"):
            if current:
                urls.append(current.rstrip(":"))
            current = part
        else:
            current += ":" + part

    if current:
        urls.append(current)

    # Clean up URLs
    return [url.rstrip("/") for url in urls if url]


# Default model name
DEFAULT_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"


class AsyncVLLMClient:
    """Async client for vLLM OpenAI-compatible API with multi-URL support."""

    def __init__(
        self,
        base_url: str,
        model: Optional[str] = None,
        max_concurrent: int = 256,
        timeout: int = 120,
        max_retries: int = 3,
    ):
        """Initialize the async vLLM client.

        Args:
            base_url: Base URL(s) for the vLLM API. Can be a single URL or
                multiple URLs separated by colons.
                Example: "http://host1:8000/v1:http://host2:8000/v1"
            model: Model name to use. Defaults to DEFAULT_MODEL.
            max_concurrent: Maximum number of concurrent requests (total).
            timeout: Request timeout in seconds.
            max_retries: Number of retry attempts on failure.
        """
        self.base_urls = parse_vllm_urls(base_url)
        if not self.base_urls:
            raise ValueError(f"No valid URLs found in: {base_url}")

        print(f"Initialized vLLM client with {len(self.base_urls)} endpoint(s):")
        for url in self.base_urls:
            print(f"  - {url}")

        self.model = model or DEFAULT_MODEL
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries

        # Round-robin URL selector (thread-safe via itertools.cycle)
        self._url_cycle = itertools.cycle(self.base_urls)
        self._url_lock = asyncio.Lock()

    async def _get_next_url(self) -> str:
        """Get next URL in round-robin fashion."""
        async with self._url_lock:
            return next(self._url_cycle)

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
        json_mode: bool = False,
    ) -> Optional[str]:
        """Send a chat completion request with retry logic.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            json_mode: If True, enforce JSON output format.

        Returns:
            Generated text content, or None if all retries failed.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        for attempt in range(self.max_retries):
            base_url = await self._get_next_url()
            url = f"{base_url}/chat/completions"

            try:
                async with self.semaphore:
                    async with aiohttp.ClientSession(
                        timeout=self.timeout
                    ) as session:
                        async with session.post(url, json=payload) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                return data["choices"][0]["message"]["content"]
                            else:
                                error_text = await resp.text()
                                print(
                                    f"API error (attempt {attempt + 1}, "
                                    f"{base_url}): {resp.status} - "
                                    f"{error_text[:200]}"
                                )
            except asyncio.TimeoutError:
                print(f"Timeout (attempt {attempt + 1}, {base_url})")
            except Exception as e:
                print(f"Error (attempt {attempt + 1}, {base_url}): {e}")

            if attempt < self.max_retries - 1:
                await asyncio.sleep(2 ** attempt)

        return None


class MassiveQueryProcessor:
    """Producer-consumer processor with per-server worker pools.

    Provides dynamic load balancing through shared queue architecture:
    - All queries go into a shared queue
    - Each server has a pool of workers pulling from the queue
    - Faster servers naturally process more queries
    - Periodic checkpointing for fault tolerance
    """

    def __init__(
        self,
        base_urls: List[str],
        model: str,
        workers_per_server: int = 64,
        timeout: int = 120,
        max_retries: int = 3,
        checkpoint_interval: int = 10000,
    ):
        """Initialize the massive query processor.

        Args:
            base_urls: List of vLLM API base URLs.
            model: Model name to use.
            workers_per_server: Number of concurrent workers per server.
            timeout: Request timeout in seconds.
            max_retries: Number of retry attempts on failure.
            checkpoint_interval: Save results every N items.
        """
        self.base_urls = base_urls
        self.model = model
        self.workers_per_server = workers_per_server
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.checkpoint_interval = checkpoint_interval

        # Statistics
        self._total_processed = 0
        self._total_success = 0
        self._start_time = None
        self._server_stats: Dict[str, Dict[str, int]] = {}

        print(f"MassiveQueryProcessor initialized:")
        print(f"  - {len(base_urls)} server(s)")
        print(f"  - {workers_per_server} workers per server")
        print(f"  - {workers_per_server * len(base_urls)} total workers")
        print(f"  - Checkpoint every {checkpoint_interval} results")

    def _parse_metric(self, text: str, metric_name: str) -> int:
        """Parse a Prometheus metric value from /metrics output."""
        # vLLM metrics format: metric_name{labels} value
        # Also handle: metric_name value
        pattern = rf'^{re.escape(metric_name)}(?:\{{[^}}]*\}})?\s+(\d+(?:\.\d+)?)'
        for line in text.split('\n'):
            match = re.match(pattern, line.strip())
            if match:
                return int(float(match.group(1)))
        return -1

    async def get_server_load(self, session: aiohttp.ClientSession, url: str) -> Dict:
        """Query /metrics endpoint for server load."""
        metrics_url = url.replace('/v1', '/metrics')
        try:
            async with session.get(metrics_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    text = await resp.text()
                    waiting = self._parse_metric(text, 'vllm:num_requests_waiting')
                    running = self._parse_metric(text, 'vllm:num_requests_running')
                    return {'waiting': waiting, 'running': running}
        except Exception:
            pass
        return {'waiting': -1, 'running': -1}

    async def monitor_loads(self, stop_event: asyncio.Event):
        """Periodically log server loads for visibility."""
        async with aiohttp.ClientSession() as session:
            while not stop_event.is_set():
                loads = {}
                for url in self.base_urls:
                    loads[url] = await self.get_server_load(session, url)

                # Build status string
                parts = []
                for url in self.base_urls:
                    # Extract hostname from URL
                    host = url.split('//')[1].split(':')[0] if '//' in url else url
                    load = loads[url]
                    stats = self._server_stats.get(url, {'processed': 0, 'success': 0})
                    parts.append(
                        f"{host}: W={load['waiting']} R={load['running']} "
                        f"done={stats['processed']}"
                    )

                elapsed = time.time() - self._start_time if self._start_time else 0
                rate = self._total_processed / elapsed if elapsed > 0 else 0

                print(
                    f"[Monitor] {self._total_processed} processed "
                    f"({self._total_success} success) | "
                    f"Rate: {rate:.1f}/sec | {' | '.join(parts)}"
                )

                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=30)
                    break
                except asyncio.TimeoutError:
                    pass

    async def _execute_query(
        self,
        session: aiohttp.ClientSession,
        server_url: str,
        query: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Execute a single query against a server with retries."""
        messages = query['messages']
        temperature = query.get('temperature', 0.7)
        max_tokens = query.get('max_tokens', 512)
        json_mode = query.get('json_mode', False)

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        url = f"{server_url}/chat/completions"

        for attempt in range(self.max_retries):
            try:
                async with session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        response_text = data["choices"][0]["message"]["content"]
                        return {
                            'idx': query['idx'],
                            'response': response_text,
                            'metadata': query.get('metadata', {}),
                        }
                    else:
                        error_text = await resp.text()
                        if attempt == self.max_retries - 1:
                            print(
                                f"API error (idx={query['idx']}, {server_url}): "
                                f"{resp.status} - {error_text[:100]}"
                            )
            except asyncio.TimeoutError:
                if attempt == self.max_retries - 1:
                    print(f"Timeout (idx={query['idx']}, {server_url})")
            except Exception as e:
                if attempt == self.max_retries - 1:
                    print(f"Error (idx={query['idx']}, {server_url}): {e}")

            if attempt < self.max_retries - 1:
                await asyncio.sleep(2 ** attempt)

        return None

    async def server_worker(
        self,
        server_url: str,
        query_queue: asyncio.Queue,
        results_queue: asyncio.Queue,
    ):
        """Worker that pulls from shared queue and sends to specific server."""
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            while True:
                query = await query_queue.get()
                if query is None:  # Poison pill
                    query_queue.task_done()
                    break

                result = await self._execute_query(session, server_url, query)

                # Update statistics
                self._total_processed += 1
                if server_url not in self._server_stats:
                    self._server_stats[server_url] = {'processed': 0, 'success': 0}
                self._server_stats[server_url]['processed'] += 1

                if result is not None:
                    self._total_success += 1
                    self._server_stats[server_url]['success'] += 1
                    await results_queue.put(result)

                query_queue.task_done()

    async def checkpoint_writer(
        self,
        results_queue: asyncio.Queue,
        output_file: str,
        process_fn: Optional[Callable[[Dict], Optional[Dict]]] = None,
    ):
        """Collect results and write checkpoints every N items.

        Args:
            results_queue: Queue to receive results from workers.
            output_file: Path to output JSONL file.
            process_fn: Optional function to process/transform results before saving.
                If it returns None, the result is skipped.
        """
        buffer = []
        total_written = 0

        while True:
            result = await results_queue.get()
            if result is None:  # Final signal
                break

            # Apply processing function if provided
            if process_fn is not None:
                processed = process_fn(result)
                if processed is None:
                    continue
                result = processed

            buffer.append(result)

            if len(buffer) >= self.checkpoint_interval:
                # Write checkpoint (sync I/O in executor to avoid blocking)
                await asyncio.get_event_loop().run_in_executor(
                    None, self._write_buffer, buffer, output_file
                )
                total_written += len(buffer)
                print(f"[Checkpoint] {total_written} results saved to {output_file}")
                buffer = []

        # Write remaining
        if buffer:
            await asyncio.get_event_loop().run_in_executor(
                None, self._write_buffer, buffer, output_file
            )
            total_written += len(buffer)
            print(f"[Final] {total_written} total results saved to {output_file}")

    def _write_buffer(self, buffer: List[Dict], output_file: str):
        """Write buffer to file (sync operation)."""
        with open(output_file, 'a', encoding='utf-8') as f:
            for item in buffer:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    async def process_all(
        self,
        queries: List[Dict[str, Any]],
        output_file: str,
        processed_ids: Optional[Set] = None,
        process_fn: Optional[Callable[[Dict], Optional[Dict]]] = None,
    ) -> Dict[str, Any]:
        """Process all queries with dynamic load balancing.

        Args:
            queries: List of query dicts, each with:
                - 'idx': Unique identifier
                - 'messages': List of message dicts for chat completion
                - 'metadata': Optional dict with additional data to preserve
                - 'temperature': Optional (default 0.7)
                - 'max_tokens': Optional (default 512)
                - 'json_mode': Optional (default False)
            output_file: Path to output JSONL file.
            processed_ids: Set of IDs to skip (for resume).
            process_fn: Optional function to process results before saving.
                Receives dict with 'idx', 'response', 'metadata'.
                Should return processed dict to save, or None to skip.

        Returns:
            Dict with statistics: total, processed, success, skipped.
        """
        self._start_time = time.time()
        self._total_processed = 0
        self._total_success = 0
        self._server_stats = {}

        # Filter out already processed
        if processed_ids:
            pending = [q for q in queries if q['idx'] not in processed_ids]
            print(f"Skipping {len(queries) - len(pending)} already processed queries")
        else:
            pending = queries

        if not pending:
            print("No queries to process")
            return {
                'total': len(queries),
                'processed': 0,
                'success': 0,
                'skipped': len(queries),
            }

        print(f"Processing {len(pending)} queries across {len(self.base_urls)} servers...")

        # Create shared queue with all pending queries
        query_queue: asyncio.Queue = asyncio.Queue()
        for q in pending:
            await query_queue.put(q)

        # Create results queue for checkpointing
        results_queue: asyncio.Queue = asyncio.Queue()

        # Start per-server worker pools
        workers = []
        for url in self.base_urls:
            for _ in range(self.workers_per_server):
                worker = asyncio.create_task(
                    self.server_worker(url, query_queue, results_queue)
                )
                workers.append(worker)

        # Start checkpoint writer
        checkpoint_task = asyncio.create_task(
            self.checkpoint_writer(results_queue, output_file, process_fn)
        )

        # Start load monitor
        stop_monitor = asyncio.Event()
        monitor_task = asyncio.create_task(self.monitor_loads(stop_monitor))

        # Wait for all queries to be processed
        await query_queue.join()

        # Signal workers to stop (poison pills)
        for _ in workers:
            await query_queue.put(None)
        await asyncio.gather(*workers)

        # Signal checkpoint writer to finish
        await results_queue.put(None)
        await checkpoint_task

        # Stop monitor
        stop_monitor.set()
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass

        elapsed = time.time() - self._start_time
        print(f"\nCompleted in {elapsed:.1f}s")
        print(f"Total: {self._total_processed}, Success: {self._total_success}")
        print(f"Rate: {self._total_processed / elapsed:.1f} queries/sec")

        # Print per-server stats
        print("\nPer-server statistics:")
        for url in self.base_urls:
            stats = self._server_stats.get(url, {'processed': 0, 'success': 0})
            host = url.split('//')[1].split(':')[0] if '//' in url else url
            pct = stats['processed'] / self._total_processed * 100 if self._total_processed > 0 else 0
            print(f"  {host}: {stats['processed']} processed ({pct:.1f}%), {stats['success']} success")

        return {
            'total': len(queries),
            'processed': self._total_processed,
            'success': self._total_success,
            'skipped': len(queries) - len(pending),
        }
