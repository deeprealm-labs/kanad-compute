//! End-to-end smoke test: spin up an in-process WebSocket server,
//! connect the gateway client, and assert the handshake + dispatch +
//! ack lifecycle works exactly the way the Python client does.
//!
//! The "solver" here is `UnimplementedSolver`, so the experiment turns
//! into one Log frame followed by one Error frame with code=not_implemented.
//! That's enough to prove the wire is honest end-to-end.

use futures_util::{SinkExt, StreamExt};
use kanad_gateway::client::unimplemented_factory;
use kanad_gateway::{ClientConfig, GatewayClient};
use kanad_protocol::{
    parse_client_message, Ack, ClientMessage, ExperimentRequest, MoleculeSpec, Registered,
    ServerMessage, SolverSpec,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tempfile::tempdir;
use tokio::net::TcpListener;
use tokio_tungstenite::accept_async;
use tokio_tungstenite::tungstenite::protocol::Message;

#[tokio::test]
async fn handshake_dispatch_ack_roundtrip() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let url = format!("http://{}", addr);

    // ── server: accept one client and act like kanad-app ─────────────────
    let server = tokio::spawn(async move {
        let (sock, _peer) = listener.accept().await.unwrap();
        let mut ws = accept_async(sock).await.unwrap();

        // 1) Expect Hello.
        let hello_frame = ws.next().await.unwrap().unwrap();
        let hello_text = match hello_frame {
            Message::Text(t) => t.to_string(),
            other => panic!("expected text Hello, got {other:?}"),
        };
        let parsed = parse_client_message(&hello_text).expect("Hello parses");
        match parsed {
            ClientMessage::Hello(h) => {
                assert_eq!(h.node_id, "test-node");
                assert_eq!(h.protocol_version, kanad_protocol::PROTOCOL_VERSION);
            }
            other => panic!("expected Hello, got {other:?}"),
        };

        // 2) Send Registered.
        let registered = ServerMessage::Registered(Registered {
            protocol_version: kanad_protocol::PROTOCOL_VERSION.into(),
            node_id: "test-node".into(),
            session_id: "sess-1".into(),
            server_version: "0.1.0".into(),
        });
        ws.send(Message::Text(serde_json::to_string(&registered).unwrap()))
            .await
            .unwrap();

        // 3) Send an ExperimentRequest. UnimplementedSolver will reject it
        //    via an Error event with code=not_implemented.
        let req = ExperimentRequest {
            experiment_id: "exp-1".into(),
            user_id: "u-1".into(),
            molecule: MoleculeSpec {
                atoms: vec![],
                basis: "sto-3g".into(),
                charge: 0,
                multiplicity: 1,
            },
            solver: SolverSpec {
                type_: "vqe".into(),
                ansatz_type: "hardware_efficient".into(),
                max_iterations: 10,
                max_excitations: 5,
                optimizer: None,
                mapper_type: None,
                convergence_threshold: None,
                n_layers: None,
                shots: None,
                frozen_core: false,
                include_singles: true,
                include_doubles: true,
                extra: HashMap::new(),
            },
            backend: "kanad_compute".into(),
            backend_credentials: None,
            deadline_ms: 600_000,
        };
        let req_frame =
            serde_json::to_string(&ServerMessage::ExperimentRequest(Box::new(req))).unwrap();
        ws.send(Message::Text(req_frame)).await.unwrap();

        // 4) Collect events until we see the Error frame.
        let mut last_seq: i64 = 0;
        let mut saw_log = false;
        let mut saw_error = false;
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        while tokio::time::Instant::now() < deadline {
            let next = tokio::time::timeout(Duration::from_millis(500), ws.next()).await;
            let frame = match next {
                Ok(Some(Ok(Message::Text(t)))) => t.to_string(),
                Ok(Some(Ok(_other))) => continue,
                Ok(Some(Err(_))) | Ok(None) => break,
                Err(_) => continue,
            };
            let msg = parse_client_message(&frame).expect("client message parses");
            match msg {
                ClientMessage::ExperimentEvent(ev) => {
                    last_seq = ev.seq;
                    match ev.payload {
                        kanad_protocol::EventPayload::Log(_) => saw_log = true,
                        kanad_protocol::EventPayload::Error(e) => {
                            saw_error = true;
                            assert_eq!(e.code.as_deref(), Some("not_implemented"));
                        }
                        _ => {}
                    }
                    if saw_error {
                        break;
                    }
                }
                ClientMessage::Ping(_) | ClientMessage::Pong(_) | ClientMessage::Hello(_) => {}
            }
        }
        assert!(saw_log, "expected Log start event");
        assert!(saw_error, "expected Error event from UnimplementedSolver");

        // 5) Ack the experiment — client should trim its outbox.
        let ack = ServerMessage::Ack(Ack {
            experiment_id: "exp-1".into(),
            last_seq,
        });
        ws.send(Message::Text(serde_json::to_string(&ack).unwrap()))
            .await
            .unwrap();

        // Give the client a tick to handle the Ack before we shut down.
        tokio::time::sleep(Duration::from_millis(100)).await;
        let _ = ws.close(None).await;
        last_seq
    });

    // ── client ───────────────────────────────────────────────────────────
    let state = tempdir().unwrap();
    let cfg = ClientConfig {
        kanad_url: url,
        api_key: "test-token".into(),
        node_id: "test-node".into(),
        client_version: "0.1.0".into(),
        kanad_core_version: None,
        state_dir: state.path().to_path_buf(),
    };
    let client = Arc::new(GatewayClient::new(cfg, unimplemented_factory()).unwrap());

    let client_task = {
        let c = client.clone();
        tokio::spawn(async move {
            // connect_once returns when the server closes the WS.
            let _ = c.connect_once().await;
        })
    };

    let last_seq = tokio::time::timeout(Duration::from_secs(10), server)
        .await
        .expect("server task timed out")
        .expect("server task panicked");
    assert!(last_seq >= 2, "expected at least Log + Error seqs");

    let _ = tokio::time::timeout(Duration::from_secs(2), client_task).await;

    // Outbox should be empty after the Ack trim.
    let state_db = state.path().join("outbox.db");
    let ob = kanad_gateway::Outbox::open(&state_db).unwrap();
    assert_eq!(
        ob.pending_count().unwrap(),
        0,
        "outbox should be empty after Ack"
    );

    // seq.json should record the last ack.
    let seq_file = state.path().join("seq.json");
    let text = std::fs::read_to_string(&seq_file).expect("seq.json written");
    assert!(
        text.contains("exp-1"),
        "seq.json must contain experiment_id"
    );
}
