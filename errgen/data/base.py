"""
Base class and shared utilities for all data provider clients.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

# HTTP timeout for all external API calls (seconds)
REQUEST_TIMEOUT = 30
# Retry on transient HTTP errors
HTTP_RETRY_ATTEMPTS = 3
HTTP_RETRY_DELAY = 2.0


class BaseDataClient:
    """
    Thin base class that provides a safe HTTP GET helper with retry logic.

    All provider-specific clients inherit from this.
    """

    def _get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """
        Perform a GET request with retry on 429/5xx responses.

        Returns the parsed JSON body on success.
        Raises requests.HTTPError on non-retriable HTTP failures.
        Raises RuntimeError if all retry attempts are exhausted.
        """
        last_exc: Exception | None = None

        for attempt in range(HTTP_RETRY_ATTEMPTS):
            try:
                resp = requests.get(
                    url,
                    params=params,
                    headers=self._merge_headers(headers),
                    timeout=REQUEST_TIMEOUT,
                )
                if resp.status_code == 429:
                    wait = HTTP_RETRY_DELAY * (2 ** attempt)
                    logger.warning(
                        "HTTP 429 from %s (attempt %d/%d). Waiting %.1fs.",
                        url,
                        attempt + 1,
                        HTTP_RETRY_ATTEMPTS,
                        wait,
                    )
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                logger.debug("HTTP GET %s → %s", url, resp.status_code)
                return resp.json()

            except requests.HTTPError as exc:
                # Non-transient HTTP error – do not retry
                raise exc
            except (requests.ConnectionError, requests.Timeout) as exc:
                wait = HTTP_RETRY_DELAY * (2 ** attempt)
                logger.warning(
                    "Connection error to %s (attempt %d/%d). Waiting %.1fs. %s",
                    url,
                    attempt + 1,
                    HTTP_RETRY_ATTEMPTS,
                    wait,
                    exc,
                )
                last_exc = exc
                time.sleep(wait)

        raise RuntimeError(
            f"HTTP GET {url} failed after {HTTP_RETRY_ATTEMPTS} attempts. "
            f"Last error: {last_exc}"
        )

    def _get_text(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> str:
        """
        Perform a GET request and return the raw text body.

        Useful for HTML pages such as SEC filing documents that are not JSON.
        """
        last_exc: Exception | None = None

        for attempt in range(HTTP_RETRY_ATTEMPTS):
            try:
                resp = requests.get(
                    url,
                    params=params,
                    headers=self._merge_headers(headers),
                    timeout=REQUEST_TIMEOUT,
                )
                if resp.status_code == 429:
                    wait = HTTP_RETRY_DELAY * (2 ** attempt)
                    logger.warning(
                        "HTTP 429 from %s (attempt %d/%d). Waiting %.1fs.",
                        url,
                        attempt + 1,
                        HTTP_RETRY_ATTEMPTS,
                        wait,
                    )
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                logger.debug("HTTP GET TEXT %s → %s", url, resp.status_code)
                return resp.text

            except requests.HTTPError as exc:
                raise exc
            except (requests.ConnectionError, requests.Timeout) as exc:
                wait = HTTP_RETRY_DELAY * (2 ** attempt)
                logger.warning(
                    "Connection error to %s (attempt %d/%d). Waiting %.1fs. %s",
                    url,
                    attempt + 1,
                    HTTP_RETRY_ATTEMPTS,
                    wait,
                    exc,
                )
                last_exc = exc
                time.sleep(wait)

        raise RuntimeError(
            f"HTTP GET text {url} failed after {HTTP_RETRY_ATTEMPTS} attempts. "
            f"Last error: {last_exc}"
        )

    @staticmethod
    def _merge_headers(headers: dict[str, str] | None = None) -> dict[str, str]:
        merged = {
            "User-Agent": (
                "ERRGen/1.0 research pipeline "
                "(contact: support@example.com)"
            )
        }
        if headers:
            merged.update(headers)
        return merged
