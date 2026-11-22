"""A second API key for when the first one is out of quota.

A host can run on a free-tier key and keep a paid one for when it runs dry.
The free one is always preferred: a call goes to the fallback only after the
primary answered with a quota error (HTTP 429, or a body naming
RESOURCE_EXHAUSTED or insufficient_quota). Any other failure is the call's own
and is raised as before, because a second key cannot fix a bad request.

After a quota error the primary key rests for ``fallback_key_cooldown`` seconds:
calls in that window go straight to the fallback rather than each paying for a
refused request first. When the window ends the primary is tried again, so a
per-minute limit moves a minute of traffic to the paid key, not a day's. The
memory of which keys are resting is process-wide, because the chat and
embedding clients are built per call and per store, and kept per purpose: one
key often serves both, and a provider's quota is per model, so chat running dry
says nothing about embeddings.
"""

import logging
import threading
import time
from collections.abc import Awaitable, Callable

import httpx

logger = logging.getLogger(__name__)

_QUOTA_MARKERS = ("resource_exhausted", "insufficient_quota")


def is_quota_error(error: BaseException) -> bool:
    """Whether ``error`` is an endpoint refusing the key for lack of quota.

    Works for httpx's ``HTTPStatusError`` and for the OpenAI SDK's errors (which
    Mem0's embedder raises): both carry the ``httpx.Response`` they came from.
    """
    response = getattr(error, "response", None)
    if not isinstance(response, httpx.Response):
        return False
    if response.status_code == 429:
        return True
    try:
        text = response.text
    except httpx.ResponseNotRead:
        text = ""
    haystack = f"{text} {error}".casefold()
    return any(marker in haystack for marker in _QUOTA_MARKERS)


class KeyPool:
    """Which of a primary and a fallback key to try first."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._resting_until: dict[tuple[str, str], float] = {}
        self._lock = threading.Lock()

    def order(self, purpose: str, primary: str, fallback: str | None) -> list[str]:
        """The keys to try, in order. Just the primary when there is no fallback."""
        if not fallback or fallback == primary:
            return [primary]
        with self._lock:
            resting = self._resting_until.get((purpose, primary), 0.0) > self._clock()
        return [fallback, primary] if resting else [primary, fallback]

    def rest(self, purpose: str, key: str, seconds: float) -> None:
        with self._lock:
            self._resting_until[(purpose, key)] = self._clock() + seconds

    def recovered(self, purpose: str, key: str) -> None:
        with self._lock:
            self._resting_until.pop((purpose, key), None)

    def clear(self) -> None:
        with self._lock:
            self._resting_until.clear()


KEY_POOL = KeyPool()


def _absorb(
    error: BaseException,
    *,
    key: str,
    primary: str,
    is_last: bool,
    cooldown: float,
    purpose: str,
    pool: KeyPool,
) -> bool:
    """Whether to try the next key after ``error``; rests the primary if so."""
    if is_last or not is_quota_error(error):
        return False
    if key == primary:
        pool.rest(purpose, primary, cooldown)
        logger.warning(
            "%s API key is out of quota; using the fallback key for the next %ss",
            purpose,
            cooldown,
        )
    else:
        logger.warning("%s fallback API key is out of quota; trying the primary", purpose)
    return True


async def with_fallback_key[T](
    primary: str,
    fallback: str | None,
    call: Callable[[str], Awaitable[T]],
    *,
    cooldown: float,
    purpose: str,
    pool: KeyPool = KEY_POOL,
) -> T:
    """Run ``call`` with the preferred key, moving to the other on a quota error."""
    keys = pool.order(purpose, primary, fallback)
    for index, key in enumerate(keys):
        try:
            result = await call(key)
        except Exception as error:
            if _absorb(
                error,
                key=key,
                primary=primary,
                is_last=index == len(keys) - 1,
                cooldown=cooldown,
                purpose=purpose,
                pool=pool,
            ):
                continue
            raise
        if key == primary:
            pool.recovered(purpose, primary)
        return result
    raise AssertionError("unreachable")


def with_fallback_key_sync[T](
    primary: str,
    fallback: str | None,
    call: Callable[[str], T],
    *,
    cooldown: float,
    purpose: str,
    pool: KeyPool = KEY_POOL,
) -> T:
    """``with_fallback_key`` for a blocking call, such as Mem0's embedder."""
    keys = pool.order(purpose, primary, fallback)
    for index, key in enumerate(keys):
        try:
            result = call(key)
        except Exception as error:
            if _absorb(
                error,
                key=key,
                primary=primary,
                is_last=index == len(keys) - 1,
                cooldown=cooldown,
                purpose=purpose,
                pool=pool,
            ):
                continue
            raise
        if key == primary:
            pool.recovered(purpose, primary)
        return result
    raise AssertionError("unreachable")
