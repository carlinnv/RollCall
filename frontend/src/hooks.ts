import { useCallback, useEffect, useMemo, useRef, useState } from "react";

/** How often a live page asks the API for fresh data. */
const POLL_MS = 2000;

/**
 * Load data from the API and keep it fresh.
 *
 * With `live` (the default) the data is refetched every 2 seconds, so a family's
 * text shows up without reloading the page. `reload` refetches immediately, for
 * use right after an action.
 *
 * - `data` only changes when the response actually differs, so polling does not
 *   re-render the page or disturb what the user is typing.
 * - Polling pauses while the browser tab is hidden and catches up when it returns.
 * - If a request fails, the last good data stays on screen and `error` is set.
 */
export function useLoad<T>(fn: () => Promise<T>, deps: unknown[] = [], live = true) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const load = useCallback(fn, deps);

  const shown = useRef<string | null>(null); // JSON of the data currently on screen
  const latest = useRef(0); // id of the newest request; older responses are discarded
  const inFlight = useRef(0);

  const request = useCallback(
    (force: boolean) => {
      // A poll waits for the previous request. An explicit reload never waits, because
      // a request already in flight may have been answered before the user's action.
      if (!force && inFlight.current > 0) return;
      const mine = ++latest.current;
      inFlight.current++;
      load()
        .then((d) => {
          if (mine !== latest.current) return;
          const text = JSON.stringify(d);
          if (text !== shown.current) {
            shown.current = text;
            setData(d);
          }
          setError(null);
        })
        .catch((e: Error) => {
          if (mine === latest.current) setError(e.message);
        })
        .finally(() => {
          inFlight.current--;
        });
    },
    [load],
  );

  const reload = useCallback(() => request(true), [request]);

  useEffect(() => {
    request(true);
    if (!live) return () => void latest.current++;
    const poll = () => {
      if (!document.hidden) request(false);
    };
    const timer = window.setInterval(poll, POLL_MS);
    document.addEventListener("visibilitychange", poll);
    return () => {
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", poll);
      latest.current++; // ignore any response that arrives after unmount or a deps change
    };
  }, [request, live]);

  return { data, error, reload };
}

/** How long a changed item stays highlighted. Matches the `flash` animation in styles.css. */
const FLASH_MS = 6000;

/**
 * Ids of items that just appeared or changed, for a highlight that fades.
 *
 * An item counts as changed when `signatureOf` returns a different string than it
 * did on the previous update. Nothing is highlighted on the first load.
 */
export function useChanged<T>(items: T[] | null | undefined, idOf: (item: T) => string, signatureOf: (item: T) => string) {
  const previous = useRef<Map<string, string> | null>(null);
  const [changed, setChanged] = useState<Set<string>>(new Set());

  useEffect(() => {
    if (!items) return;
    const next = new Map(items.map((item) => [idOf(item), signatureOf(item)]));
    const before = previous.current;
    previous.current = next;
    if (!before) return;
    const ids = [...next].filter(([id, signature]) => before.get(id) !== signature).map(([id]) => id);
    if (!ids.length) return;
    setChanged((current) => new Set([...current, ...ids]));
    // Not cancelled when `items` changes again: each highlight must still expire on its own schedule.
    window.setTimeout(() => {
      setChanged((current) => {
        const rest = new Set(current);
        ids.forEach((id) => rest.delete(id));
        return rest;
      });
    }, FLASH_MS);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [items]);

  return changed;
}

/**
 * Keep rows where they are while live updates arrive, so a row that changes does
 * not jump away from the reader.
 *
 * The server's order is used on first load, whenever rows are added, and when
 * `resort` is called. In between, existing rows hold their position, except that a
 * row which starts matching `pinFirst` (a new safety case) moves to the top.
 * `held` is true while the order on screen differs from the server's.
 */
export function useStableOrder<T>(items: T[] | null | undefined, idOf: (item: T) => string, pinFirst?: (item: T) => boolean) {
  const order = useRef<string[]>([]);
  const pinned = useRef<Set<string>>(new Set());
  const [resortCount, setResortCount] = useState(0);

  const result = useMemo(() => {
    if (!items) return { rows: [] as T[], held: false };
    const byId = new Map(items.map((item) => [idOf(item), item]));
    const server = items.map(idOf);
    const known = new Set(order.current);
    const hasNewRows = server.some((id) => !known.has(id));

    if (hasNewRows || order.current.length === 0) {
      order.current = server;
    } else {
      let kept = order.current.filter((id) => byId.has(id));
      if (pinFirst) {
        const newlyPinned = kept.filter((id) => pinFirst(byId.get(id)!) && !pinned.current.has(id));
        if (newlyPinned.length) kept = [...newlyPinned, ...kept.filter((id) => !newlyPinned.includes(id))];
      }
      order.current = kept;
    }
    if (pinFirst) pinned.current = new Set(server.filter((id) => pinFirst(byId.get(id)!)));

    return { rows: order.current.map((id) => byId.get(id)!), held: order.current.join() !== server.join() };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [items, resortCount]);

  const resort = useCallback(() => {
    order.current = [];
    setResortCount((n) => n + 1);
  }, []);

  return { ...result, resort };
}
