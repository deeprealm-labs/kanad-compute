//! Per-experiment monotonic seq state, persisted to disk via atomic
//! tmpfile + rename. Mirror of the `seq.json` file maintained by the
//! Python `ComputeWSClient`.

use parking_lot::Mutex;
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExperimentBuffer {
    pub next_seq: i64,
    pub last_ack_seq: i64,
}

impl ExperimentBuffer {
    pub fn from_last_ack(last_ack: i64) -> Self {
        Self {
            next_seq: last_ack + 1,
            last_ack_seq: last_ack,
        }
    }
}

pub struct SeqState {
    path: PathBuf,
    buffers: Mutex<HashMap<String, ExperimentBuffer>>,
}

impl SeqState {
    /// Open the state file. Missing file → empty state, that's fine.
    pub fn open(path: impl Into<PathBuf>) -> std::io::Result<Self> {
        let path = path.into();
        let buffers = match std::fs::read_to_string(&path) {
            Ok(s) if !s.trim().is_empty() => parse(&s),
            _ => HashMap::new(),
        };
        Ok(Self {
            path,
            buffers: Mutex::new(buffers),
        })
    }

    /// Snapshot of `Hello.last_ack_seq` — only experiments with positive acks.
    pub fn last_ack_seq_map(&self) -> HashMap<String, i64> {
        self.buffers
            .lock()
            .iter()
            .filter(|(_, b)| b.last_ack_seq > 0)
            .map(|(k, b)| (k.clone(), b.last_ack_seq))
            .collect()
    }

    /// Assign the next monotonic seq for an experiment; creates the buffer
    /// on first use.
    pub fn next_seq(&self, experiment_id: &str) -> i64 {
        let mut g = self.buffers.lock();
        let buf = g
            .entry(experiment_id.to_owned())
            .or_insert_with(|| ExperimentBuffer::from_last_ack(0));
        // First call yields 1 (matches Python: next_seq starts at 1).
        if buf.next_seq == 0 {
            buf.next_seq = 1;
        }
        let s = buf.next_seq;
        buf.next_seq = s + 1;
        s
    }

    /// Record a server ack. Returns true if the on-disk file should be
    /// rewritten (i.e. last_ack_seq moved forward).
    pub fn record_ack(&self, experiment_id: &str, last_seq: i64) -> bool {
        let mut g = self.buffers.lock();
        let buf = g.entry(experiment_id.to_owned()).or_default();
        if last_seq > buf.last_ack_seq {
            buf.last_ack_seq = last_seq;
            if buf.next_seq <= last_seq {
                buf.next_seq = last_seq + 1;
            }
            true
        } else {
            false
        }
    }

    /// Persist the current state to disk atomically (tmpfile + rename).
    pub fn save(&self) -> std::io::Result<()> {
        let snapshot: Map<String, Value> = self
            .buffers
            .lock()
            .iter()
            .filter(|(_, b)| b.last_ack_seq > 0)
            .map(|(k, b)| (k.clone(), Value::from(b.last_ack_seq)))
            .collect();
        let body =
            serde_json::to_string(&Value::Object(snapshot)).map_err(std::io::Error::other)?;
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = self.path.with_extension("json.tmp");
        std::fs::write(&tmp, body)?;
        std::fs::rename(&tmp, &self.path)?;
        Ok(())
    }

    #[cfg(test)]
    fn snapshot(&self) -> HashMap<String, ExperimentBuffer> {
        self.buffers.lock().clone()
    }
}

fn parse(s: &str) -> HashMap<String, ExperimentBuffer> {
    serde_json::from_str::<HashMap<String, i64>>(s)
        .unwrap_or_default()
        .into_iter()
        .map(|(k, v)| (k, ExperimentBuffer::from_last_ack(v)))
        .collect()
}

pub fn path_in(state_dir: &Path) -> PathBuf {
    state_dir.join("seq.json")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn next_seq_monotonic_per_experiment() {
        let d = tempdir().unwrap();
        let s = SeqState::open(d.path().join("seq.json")).unwrap();
        assert_eq!(s.next_seq("a"), 1);
        assert_eq!(s.next_seq("a"), 2);
        assert_eq!(s.next_seq("b"), 1);
        assert_eq!(s.next_seq("a"), 3);
    }

    #[test]
    fn record_ack_advances_state_and_save_roundtrips() {
        let d = tempdir().unwrap();
        let p = d.path().join("seq.json");
        let s = SeqState::open(&p).unwrap();
        s.next_seq("a"); // 1
        s.next_seq("a"); // 2
        assert!(s.record_ack("a", 2));
        assert!(!s.record_ack("a", 1)); // regressing ack ignored
        s.save().unwrap();

        // Reopen and verify last_ack_seq survived.
        let s2 = SeqState::open(&p).unwrap();
        let snap = s2.snapshot();
        assert_eq!(snap["a"].last_ack_seq, 2);
        // next_seq is rebuilt from last_ack so we keep monotonicity.
        assert_eq!(s2.next_seq("a"), 3);
    }

    #[test]
    fn last_ack_map_skips_zero_entries() {
        let d = tempdir().unwrap();
        let s = SeqState::open(d.path().join("seq.json")).unwrap();
        s.next_seq("a");
        let m = s.last_ack_seq_map();
        assert!(m.is_empty()); // ack still 0, omitted from Hello payload
        s.record_ack("a", 5);
        let m = s.last_ack_seq_map();
        assert_eq!(m["a"], 5);
    }

    #[test]
    fn empty_file_is_fine() {
        let d = tempdir().unwrap();
        let p = d.path().join("seq.json");
        std::fs::write(&p, "").unwrap();
        let s = SeqState::open(&p).unwrap();
        assert!(s.snapshot().is_empty());
    }
}
