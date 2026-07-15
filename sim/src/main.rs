use terminal_sim::replay;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("diff") => {
            let path = args.get(2).expect("usage: tsim diff <replay> [verbose]");
            let verbose: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(8);
            let ds = replay::diff_replay(path, verbose);
            println!(
                "== {}: turns {}/{} ok, frames {}/{} ok, restore {}/{} ok",
                path, ds.turns_ok, ds.turns, ds.frames_ok, ds.frames, ds.restore_ok,
                ds.restore_checked
            );
            if ds.turns_ok == ds.turns && ds.restore_ok == ds.restore_checked {
                println!("PASS");
            } else {
                println!("FAIL (first divergence: {:?})", ds.first_bad);
                std::process::exit(1);
            }
        }
        Some("bench") => {
            let path = args.get(2).expect("usage: tsim bench <replay> [iters]");
            let iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(200);
            bench(path, iters);
        }
        _ => {
            eprintln!("usage: tsim diff <replay> [verbose] | tsim bench <replay> [iters]");
            std::process::exit(2);
        }
    }
}

fn bench(path: &str, iters: usize) {
    use std::sync::Arc;
    use terminal_sim::config::Config;
    let text = std::fs::read_to_string(path).expect("replay");
    let raw: Vec<serde_json::Value> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let rep = replay::parse(path);
    let cfg = Arc::new(Config::from_json(&rep.config));

    // Pre-extract per-turn (turn_frame_idx, frame0_raw_idx)
    let mut turns: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < rep.frames.len() {
        if rep.frames[i].phase == 0
            && i + 1 < rep.frames.len()
            && rep.frames[i + 1].phase == 1
        {
            turns.push((i, i + 1));
        }
        i += 1;
    }

    let t0 = std::time::Instant::now();
    let mut total_frames = 0u64;
    for _ in 0..iters {
        for &(ti, f0) in turns.iter() {
            let mut st = replay::state_from_turn_frame(cfg.clone(), &rep.frames[ti]);
            replay::attach_ids(&mut st, &raw[ti + 1]);
            let cmds = replay::commands_from_frame0(&raw[f0 + 1]);
            let mut frames = Vec::new();
            terminal_sim::engine::play_turn(&mut st, cmds, false, &mut frames);
            total_frames += frames.len() as u64;
        }
    }
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "{} iters x {} turns: {:.2}s -> {:.0} turns/sec, {:.0} frames/sec",
        iters,
        turns.len(),
        dt,
        (iters * turns.len()) as f64 / dt,
        total_frames as f64 / dt
    );
}
