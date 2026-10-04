//! End-to-end smoke tests: a real UI server driving the real `tp3-rust` binary.

use std::{
    io::{Read, Write},
    net::{SocketAddr, TcpStream},
    path::PathBuf,
    time::{Duration, Instant},
};

use serde_json::{json, Value};
use tp3_rust::ui::{start, UiConfig, UiServer};

fn skip_without_data() -> bool {
    if std::path::Path::new("data/digits.csv").is_file() {
        false
    } else {
        eprintln!("skipping: data/digits.csv is not available in this checkout");
        true
    }
}

fn output_root(tag: &str) -> PathBuf {
    PathBuf::from(format!("target/ui-smoke/{tag}-{}", std::process::id()))
}

fn start_server(tag: &str) -> (UiServer, PathBuf) {
    let root = output_root(tag);
    let server = start(UiConfig {
        addr: "127.0.0.1:0".parse().unwrap(),
        exe: PathBuf::from(env!("CARGO_BIN_EXE_tp3-rust")),
        output_root: root.clone(),
        digits: PathBuf::from("data/digits.csv"),
        more_digits: PathBuf::from("data/more_digits.csv"),
        baseline_model: PathBuf::from("output/exercise2/selected_model.toml"),
        max_parallel: 2,
    })
    .expect("server starts");
    (server, root)
}

type Reply = (u16, Vec<(String, String)>, String);

fn http(
    addr: SocketAddr,
    method: &str,
    path: &str,
    extra_headers: &[(&str, &str)],
    body: Option<&str>,
) -> Reply {
    let mut stream = TcpStream::connect(addr).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(30)))
        .unwrap();
    let mut request = format!("{method} {path} HTTP/1.1\r\nConnection: close\r\n");
    if !extra_headers
        .iter()
        .any(|(name, _)| name.eq_ignore_ascii_case("host"))
    {
        request.push_str(&format!("Host: 127.0.0.1:{}\r\n", addr.port()));
    }
    if !extra_headers
        .iter()
        .any(|(name, _)| name.eq_ignore_ascii_case("content-type"))
    {
        request.push_str("Content-Type: application/json\r\n");
    }
    for (name, value) in extra_headers {
        request.push_str(&format!("{name}: {value}\r\n"));
    }
    if let Some(body) = body {
        request.push_str(&format!("Content-Length: {}\r\n", body.len()));
    }
    request.push_str("\r\n");
    if let Some(body) = body {
        request.push_str(body);
    }
    stream.write_all(request.as_bytes()).unwrap();
    let mut raw = Vec::new();
    let _ = stream.read_to_end(&mut raw);
    let text = String::from_utf8_lossy(&raw).into_owned();
    let (head, body) = text.split_once("\r\n\r\n").unwrap_or((&text, ""));
    let mut lines = head.split("\r\n");
    let status = lines
        .next()
        .and_then(|line| line.split(' ').nth(1))
        .and_then(|code| code.parse().ok())
        .unwrap_or(0);
    let headers = lines
        .filter_map(|line| line.split_once(':'))
        .map(|(name, value)| (name.to_ascii_lowercase(), value.trim().to_owned()))
        .collect();
    (status, headers, body.to_owned())
}

fn json_of(reply: &Reply) -> Value {
    serde_json::from_str(&reply.2).unwrap_or_else(|_| panic!("not JSON: {}", reply.2))
}

fn tiny_candidate(name: &str) -> Value {
    json!({
        "name": name,
        "topology": [784, 8, 10],
        "hidden_activation": "relu",
        "optimizer": "adam",
        "learning_rate": 0.01,
        "batch_size": 256,
        "max_epochs": 2,
        "patience": 2,
        "min_delta": 0.0,
        "seed": 7
    })
}

fn wait_for_runs(
    addr: SocketAddr,
    ids: &[String],
    poll: Duration,
    mut observe: impl FnMut(&Value),
) -> Vec<Value> {
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        let list = json_of(&http(addr, "GET", "/api/runs", &[], None));
        observe(&list);
        let mine: Vec<Value> = list
            .as_array()
            .unwrap()
            .iter()
            .filter(|run| ids.iter().any(|id| run["id"] == *id))
            .cloned()
            .collect();
        let finished = mine
            .iter()
            .all(|run| run["status"] == "done" || run["status"] == "failed");
        if finished && mine.len() == ids.len() {
            return mine;
        }
        assert!(
            Instant::now() < deadline,
            "runs did not finish in 120 s: {mine:?}"
        );
        std::thread::sleep(poll);
    }
}

#[test]
fn single_run_end_to_end() {
    if skip_without_data() {
        return;
    }
    let (server, root) = start_server("single");
    let addr = server.local_addr();

    let defaults = json_of(&http(addr, "GET", "/api/defaults", &[], None));
    assert_eq!(defaults["candidate"]["topology"], json!([784, 64, 10]));

    let body = json!({ "exercise": "exercise2", "candidate": tiny_candidate("smoke") }).to_string();
    let reply = http(addr, "POST", "/api/runs", &[], Some(&body));
    assert_eq!(reply.0, 202, "{}", reply.2);
    let created = json_of(&reply);
    let id = created["runs"][0]["id"].as_str().unwrap().to_owned();

    let runs = wait_for_runs(
        addr,
        std::slice::from_ref(&id),
        Duration::from_millis(500),
        |_| {},
    );
    let run = &runs[0];
    assert_eq!(run["status"], "done", "{run}");
    let accuracy = run["result"]["validation_accuracy"].as_f64().unwrap();
    assert!(accuracy > 0.1 && accuracy <= 1.0, "accuracy {accuracy}");

    let metrics = json_of(&http(
        addr,
        "GET",
        &format!("/api/runs/{id}/metrics"),
        &[],
        None,
    ));
    let points = metrics.as_array().unwrap();
    assert!(!points.is_empty());
    assert!(points[0]["epoch"].as_u64().unwrap() >= 1);

    let run_dir = root.join(&id);
    assert!(run_dir.join("config.toml").is_file());
    assert!(run_dir.join("live_metrics.csv").is_file());
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn hostile_requests_are_refused() {
    let (server, root) = start_server("hostile");
    let addr = server.local_addr();

    let wrong_host = http(
        addr,
        "GET",
        "/api/defaults",
        &[("Host", "evil.example")],
        None,
    );
    assert_eq!(wrong_host.0, 403);
    let wrong_origin = http(
        addr,
        "GET",
        "/api/defaults",
        &[("Origin", "http://evil.example")],
        None,
    );
    assert_eq!(wrong_origin.0, 403);
    let plain = http(
        addr,
        "POST",
        "/api/runs",
        &[("Content-Type", "text/plain")],
        Some("{}"),
    );
    assert_eq!(plain.0, 415);

    let mut candidate = tiny_candidate("bad");
    candidate["learning_rate"] = json!(-1);
    let body = json!({ "exercise": "exercise2", "candidate": candidate }).to_string();
    assert_eq!(http(addr, "POST", "/api/runs", &[], Some(&body)).0, 400);

    let mut candidate = tiny_candidate("bad");
    candidate["topology"] = json!([100, 8, 10]);
    let body = json!({ "exercise": "exercise2", "candidate": candidate }).to_string();
    assert_eq!(http(addr, "POST", "/api/runs", &[], Some(&body)).0, 400);

    assert_eq!(
        http(addr, "POST", "/api/runs", &[], Some("not json")).0,
        400
    );
    assert_eq!(http(addr, "GET", "/api/nope", &[], None).0, 404);
    assert_eq!(http(addr, "GET", "/api/runs/../etc", &[], None).0, 404);
    assert_eq!(http(addr, "GET", "/api/runs/abc/metrics", &[], None).0, 404);
    assert_eq!(http(addr, "DELETE", "/api/runs", &[], None).0, 405);
    assert_eq!(http(addr, "OPTIONS", "/api/runs", &[], None).0, 405);

    let refused = start(UiConfig {
        addr: "0.0.0.0:0".parse().unwrap(),
        exe: PathBuf::from(env!("CARGO_BIN_EXE_tp3-rust")),
        output_root: root.clone(),
        digits: PathBuf::from("data/digits.csv"),
        more_digits: PathBuf::from("data/more_digits.csv"),
        baseline_model: PathBuf::from("output/exercise2/selected_model.toml"),
        max_parallel: 1,
    });
    assert!(refused.is_err());
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn sweep_runs_with_cap_and_exports_best() {
    if skip_without_data() {
        return;
    }
    let (server, root) = start_server("sweep");
    let addr = server.local_addr();

    let mut candidate = tiny_candidate("sweep");
    candidate["max_epochs"] = json!(1);
    let body = json!({
        "exercise": "exercise2",
        "candidate": candidate,
        "sweep": { "field": "learning_rate", "values": ["0.01", "0.02", "0.03"] },
        "parallel": 2
    })
    .to_string();
    let reply = http(addr, "POST", "/api/runs", &[], Some(&body));
    assert_eq!(reply.0, 202, "{}", reply.2);
    let created = json_of(&reply);
    let group = created["group"].as_str().unwrap().to_owned();
    let created_runs = created["runs"].as_array().unwrap();
    assert_eq!(created_runs.len(), 3);
    assert!(created_runs
        .iter()
        .all(|run| run["group"] == group.as_str()));
    let ids: Vec<String> = created_runs
        .iter()
        .map(|run| run["id"].as_str().unwrap().to_owned())
        .collect();

    let mut max_running = 0;
    let runs = wait_for_runs(addr, &ids, Duration::from_millis(250), |list| {
        let running = list
            .as_array()
            .unwrap()
            .iter()
            .filter(|run| run["group"] == group.as_str() && run["status"] == "running")
            .count();
        max_running = max_running.max(running);
    });
    assert!(max_running <= 2, "saw {max_running} runs at once");
    assert!(runs.iter().all(|run| run["status"] == "done"), "{runs:?}");

    let best_name = runs
        .iter()
        .max_by(|a, b| {
            let accuracy = |run: &Value| run["result"]["validation_accuracy"].as_f64().unwrap();
            accuracy(a).total_cmp(&accuracy(b))
        })
        .map(|run| run["name"].as_str().unwrap().to_owned())
        .unwrap();
    let best = http(
        addr,
        "GET",
        &format!("/api/runs/best/config.toml?group={group}"),
        &[],
        None,
    );
    assert_eq!(best.0, 200, "{}", best.2);
    let study: tp3_rust::digits::config::DigitStudyConfig = toml::from_str(&best.2).unwrap();
    study.validate().unwrap();
    assert_eq!(study.candidates.len(), 1);
    assert_eq!(study.candidates[0].name, best_name);

    let one = http(
        addr,
        "GET",
        &format!("/api/runs/{}/config.toml", ids[0]),
        &[],
        None,
    );
    assert_eq!(one.0, 200);
    let disposition = one
        .1
        .iter()
        .find(|(name, _)| name == "content-disposition")
        .map(|(_, value)| value.as_str())
        .unwrap_or("");
    assert!(
        disposition.starts_with("attachment; filename=\""),
        "{disposition}"
    );

    assert_eq!(
        http(
            addr,
            "GET",
            "/api/runs/best/config.toml?group=../x",
            &[],
            None
        )
        .0,
        400
    );
    assert_eq!(
        http(
            addr,
            "GET",
            "/api/runs/best/config.toml?exercise=nope",
            &[],
            None
        )
        .0,
        400
    );
    assert_eq!(
        http(
            addr,
            "GET",
            "/api/runs/best/config.toml?group=g1-1",
            &[],
            None
        )
        .0,
        404
    );
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn index_page_has_a_control_for_every_candidate_field() {
    let (server, root) = start_server("index");
    let addr = server.local_addr();

    let page = http(addr, "GET", "/", &[], None);
    assert_eq!(page.0, 200);
    let header = |name: &str| {
        page.1
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.clone())
    };
    assert!(header("content-type").unwrap().starts_with("text/html"));
    assert!(header("content-security-policy").is_some());

    let defaults =
        serde_json::to_value(tp3_rust::digits::config::CandidateConfig::default()).unwrap();
    for key in defaults.as_object().unwrap().keys() {
        assert!(
            page.2.contains(&format!("data-field=\"{key}\"")),
            "the page has no control for CandidateConfig::{key}"
        );
    }
    for key in ["validation_ratio", "split_seed"] {
        assert!(page.2.contains(&format!("data-field=\"{key}\"")), "{key}");
    }

    let script = http(addr, "GET", "/app.js", &[], None);
    assert_eq!(script.0, 200);
    let content_type = script
        .1
        .iter()
        .find(|(key, _)| key == "content-type")
        .map(|(_, value)| value.as_str())
        .unwrap_or("");
    assert!(content_type.contains("javascript"), "{content_type}");
    assert!(script.2.contains("/api/runs"));
    let _ = std::fs::remove_dir_all(root);
}
