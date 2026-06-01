//! Crash-resilient WS outbox — Rust mirror of `kanad_compute/outbox.py`.
//!
//! Same contract:
//!   * `record` inserts BEFORE the frame goes on the wire (durable).
//!   * `ack(exp_id, last_seq)` deletes rows ≤ `last_seq` for that exp.
//!   * `pending()` returns every row ordered by `(experiment_id, seq)`
//!     for replay on reconnect.
//!   * `gc(older_than)` is defensive cleanup; default age is 24 h.
//!
//! WAL journal mode + `synchronous = NORMAL` mirrors the Python version's
//! durability/latency trade-off.

use chrono::Utc;
use parking_lot::Mutex;
use rusqlite::{params, Connection};
use std::path::{Path, PathBuf};
use std::time::Duration;
use thiserror::Error;

pub const GC_DEFAULT_AGE: Duration = Duration::from_secs(86_400);

#[derive(Debug, Error)]
pub enum OutboxError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("sqlite error: {0}")]
    Sql(#[from] rusqlite::Error),
}

#[derive(Debug, Clone, PartialEq)]
pub struct OutboxRow {
    pub experiment_id: String,
    pub seq: i64,
    pub kind: String,
    pub payload_json: String,
}

pub struct Outbox {
    path: PathBuf,
    conn: Mutex<Connection>,
}

impl Outbox {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, OutboxError> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(&path)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        let me = Self {
            path,
            conn: Mutex::new(conn),
        };
        me.init_schema()?;
        Ok(me)
    }

    fn init_schema(&self) -> Result<(), OutboxError> {
        let conn = self.conn.lock();
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                kind TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_events_exp_seq
                ON events(experiment_id, seq);
            CREATE INDEX IF NOT EXISTS idx_events_created
                ON events(created_at);
            "#,
        )?;
        Ok(())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn record(
        &self,
        experiment_id: &str,
        seq: i64,
        kind: &str,
        payload_json: &str,
    ) -> Result<(), OutboxError> {
        let now = Utc::now().timestamp_millis() as f64 / 1000.0;
        let conn = self.conn.lock();
        conn.execute(
            "INSERT INTO events(experiment_id, seq, kind, payload_json, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![experiment_id, seq, kind, payload_json, now],
        )?;
        Ok(())
    }

    /// Delete every row for `experiment_id` with seq ≤ `last_seq`.
    /// Returns the count of rows removed.
    pub fn ack(&self, experiment_id: &str, last_seq: i64) -> Result<usize, OutboxError> {
        let conn = self.conn.lock();
        let n = conn.execute(
            "DELETE FROM events WHERE experiment_id = ?1 AND seq <= ?2",
            params![experiment_id, last_seq],
        )?;
        Ok(n)
    }

    /// Materialized snapshot of every unacked row, ordered by
    /// `(experiment_id, seq)`. Caller iterates without holding the DB lock.
    pub fn pending(&self) -> Result<Vec<OutboxRow>, OutboxError> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT experiment_id, seq, kind, payload_json FROM events
             ORDER BY experiment_id, seq",
        )?;
        let rows = stmt
            .query_map([], |r| {
                Ok(OutboxRow {
                    experiment_id: r.get(0)?,
                    seq: r.get(1)?,
                    kind: r.get(2)?,
                    payload_json: r.get(3)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    pub fn pending_count(&self) -> Result<usize, OutboxError> {
        let conn = self.conn.lock();
        let n: i64 = conn.query_row("SELECT COUNT(*) FROM events", [], |r| r.get(0))?;
        Ok(n as usize)
    }

    pub fn gc(&self, older_than: Duration) -> Result<usize, OutboxError> {
        let cutoff = (Utc::now().timestamp_millis() as f64 / 1000.0) - older_than.as_secs_f64();
        let conn = self.conn.lock();
        let n = conn.execute("DELETE FROM events WHERE created_at < ?1", params![cutoff])?;
        if n > 0 {
            tracing::info!(removed = n, ?older_than, "outbox.gc");
        }
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn tmp_outbox() -> (tempfile::TempDir, Outbox) {
        let d = tempdir().unwrap();
        let p = d.path().join("outbox.sqlite");
        let ob = Outbox::open(&p).unwrap();
        (d, ob)
    }

    #[test]
    fn record_and_pending_ordered() {
        let (_d, ob) = tmp_outbox();
        ob.record("exp-A", 2, "Progress", "{}").unwrap();
        ob.record("exp-A", 1, "Log", "{}").unwrap();
        ob.record("exp-B", 1, "Log", "{}").unwrap();
        let p = ob.pending().unwrap();
        assert_eq!(p.len(), 3);
        assert_eq!(p[0].experiment_id, "exp-A");
        assert_eq!(p[0].seq, 1);
        assert_eq!(p[1].seq, 2);
        assert_eq!(p[2].experiment_id, "exp-B");
    }

    #[test]
    fn ack_drops_up_to_seq() {
        let (_d, ob) = tmp_outbox();
        for s in 1..=5 {
            ob.record("e", s, "Progress", "{}").unwrap();
        }
        let n = ob.ack("e", 3).unwrap();
        assert_eq!(n, 3);
        assert_eq!(ob.pending_count().unwrap(), 2);
    }

    #[test]
    fn ack_other_experiment_no_op() {
        let (_d, ob) = tmp_outbox();
        ob.record("a", 1, "Log", "{}").unwrap();
        ob.record("b", 1, "Log", "{}").unwrap();
        assert_eq!(ob.ack("c", 999).unwrap(), 0);
        assert_eq!(ob.pending_count().unwrap(), 2);
    }

    #[test]
    fn durable_across_reopen() {
        let d = tempdir().unwrap();
        let p = d.path().join("ob.sqlite");
        {
            let ob = Outbox::open(&p).unwrap();
            ob.record("e", 1, "FinalResult", "{\"energy\":-1.1}")
                .unwrap();
        }
        // Drop the previous handle, reopen — rows must still be there.
        let ob2 = Outbox::open(&p).unwrap();
        let rows = ob2.pending().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].kind, "FinalResult");
    }

    #[test]
    fn gc_removes_old_rows() {
        let (_d, ob) = tmp_outbox();
        ob.record("e", 1, "Log", "{}").unwrap();
        // Backdate row to 2 days ago by direct sql.
        {
            let conn = ob.conn.lock();
            let old = (Utc::now().timestamp_millis() as f64 / 1000.0) - 2.0 * 86_400.0;
            conn.execute("UPDATE events SET created_at = ?1", params![old])
                .unwrap();
        }
        let removed = ob.gc(GC_DEFAULT_AGE).unwrap();
        assert_eq!(removed, 1);
        assert_eq!(ob.pending_count().unwrap(), 0);
    }
}
