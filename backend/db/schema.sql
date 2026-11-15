CREATE TABLE IF NOT EXISTS absence_events (
  time        timestamptz NOT NULL,
  absence_id  text NOT NULL,
  student_id  text NOT NULL,
  barrier     text,
  bus_route   text,
  classroom   text,
  status      text
);
SELECT create_hypertable('absence_events', by_range('time'), if_not_exists => TRUE);
CREATE MATERIALIZED VIEW IF NOT EXISTS absences_daily
WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
SELECT time_bucket('1 day', time) AS day, barrier, bus_route, classroom, count(*) AS absences
FROM absence_events
GROUP BY day, barrier, bus_route, classroom
WITH NO DATA;
SELECT add_continuous_aggregate_policy('absences_daily', start_offset => INTERVAL '60 days', end_offset => NULL, schedule_interval => INTERVAL '5 minutes', if_not_exists => TRUE);
CREATE UNIQUE INDEX IF NOT EXISTS absence_events_key ON absence_events (absence_id, time);
-- The app's records (students, absences, escalations, plans, messages, ...), one JSON row each.
-- Written by app/services/persist.py, which also creates it on startup.
CREATE TABLE IF NOT EXISTS app_records (
  collection  text        NOT NULL,
  id          text        NOT NULL,
  pos         integer     NOT NULL,
  data        jsonb       NOT NULL,
  updated_at  timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (collection, id)
);
