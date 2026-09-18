//! Local-validator matrix for the sell-off window + surge fee.
//!
//! Reads `scripts/local-stand/stand.json` (built by `scripts/local-stand/up.sh`)
//! and, against the REAL Coffer program running in `solana-test-validator`:
//!
//! - `shared_suite_*` runs Titan's shared suite (`tests/common/mod.rs`) on every
//!   static pool of the configuration matrix and prints a pool × test table;
//! - `real_routed_transactions` sends real `swap_route_v3` transactions through
//!   the Titan router program for every direction of every static pool at
//!   several sizes and compares the user's ATA delta AND the post-swap pool
//!   account with the venue's quote / `apply_swap`;
//! - `dynamic_*` fills windows in chunks (direct swaps and routed swaps),
//!   proves the venue reports a full window as unavailable while the program
//!   reverts with `MaxSelloffExceeded`, waits for a real-time rotation on a
//!   20-second window, toggles `set_token_active` / `set_swaps_enabled` /
//!   `set_pool_enabled`, checks the buy side is unaffected and that the surge
//!   fee accrues in the output token's protocol bucket exactly as predicted.
//!
//! Every test SKIPs when the stand is not up. Run through
//! `scripts/local-stand/run-matrix.sh` (order matters: the suite reads the
//! static pools before the real transactions move them).

#![allow(clippy::result_large_err)]
#![allow(dead_code)] // diagnostic fields kept for the Debug output of failures

mod common;

use std::str::FromStr;
use std::time::{Duration, Instant};

use common::SuiteConfig;
use solana_client::nonblocking::rpc_client::RpcClient;
use solana_client::rpc_config::RpcTransactionConfig;
use solana_instruction::Instruction;
use solana_program_pack::Pack;
use solana_pubkey::Pubkey;
use solana_sdk::commitment_config::CommitmentConfig;
use solana_sdk::compute_budget::ComputeBudgetInstruction;
use solana_sdk::instruction::InstructionError;
use solana_sdk::signature::Signature;
use solana_sdk::signature::{Keypair, Signer, read_keypair_file};
use solana_sdk::transaction::{Transaction, TransactionError};
use solana_transaction_status_client_types::{
    UiTransactionEncoding, option_serializer::OptionSerializer,
};
use spl_token::state::Account as TokenAccount;

use titan_integration_template::account_caching::rpc_cache::RpcClientCache;
use titan_integration_template::coffer::errors::ErrorCode;
use titan_integration_template::coffer::state::CofferPool;
use titan_integration_template::coffer::swap::{apply_swap, quote_exact_in, selloff_headroom};
use titan_integration_template::coffer_venue::{COFFER_PROGRAM_ID, CofferVenue};
use titan_integration_template::local_stand::*;
use titan_integration_template::trading_venue::error::TradingVenueError;
use titan_integration_template::trading_venue::{
    FromAccount, QuoteRequest, QuoteResult, SwapType, TradingVenue,
};

// Allocation guard for the `construction` run under `--profile release-debug`.
#[cfg(debug_assertions)]
#[global_allocator]
static A: assert_no_alloc::AllocDisabler = assert_no_alloc::AllocDisabler;

// ---------------------------------------------------------------------------
// Stand access
// ---------------------------------------------------------------------------

struct Stand {
    manifest: StandManifest,
    rpc: RpcClient,
    wallet: Keypair,
}

async fn stand_or_skip(test: &str) -> Option<Stand> {
    let Some(manifest) = StandManifest::load() else {
        eprintln!(
            "SKIP {test}: no {} — run scripts/local-stand/up.sh",
            StandManifest::path().display()
        );
        return None;
    };
    let Ok(url) = std::env::var("SOLANA_RPC_URL") else {
        eprintln!("SKIP {test}: set SOLANA_RPC_URL={}", manifest.rpc);
        return None;
    };
    let rpc = RpcClient::new_with_commitment(url, CommitmentConfig::confirmed());
    if rpc.get_health().await.is_err() {
        eprintln!("SKIP {test}: local validator not reachable");
        return None;
    }
    let wallet_path = std::env::var("LOCAL_STAND_WALLET")
        .unwrap_or_else(|_| format!("{}/.config/solana/id.json", std::env::var("HOME").unwrap()));
    let wallet = read_keypair_file(&wallet_path).expect("wallet keypair");
    assert_eq!(
        wallet.pubkey().to_string(),
        manifest.wallet,
        "wallet differs from the stand's pool admin"
    );
    Some(Stand {
        manifest,
        rpc,
        wallet,
    })
}

impl Stand {
    fn pool(&self, case: &str) -> &StandPool {
        self.manifest
            .pools
            .iter()
            .find(|p| p.case == case)
            .unwrap_or_else(|| panic!("pool {case} in manifest"))
    }

    fn mint(&self, name: &str) -> (Pubkey, Pubkey) {
        let m = self.manifest.mint(name);
        (
            Pubkey::from_str(&m.address).unwrap(),
            if m.token_2022 {
                spl_token_2022::ID
            } else {
                spl_token::ID
            },
        )
    }

    /// Fresh venue from live state (fresh cache: no stale accounts).
    async fn venue(&self, pool: Pubkey) -> CofferVenue {
        let account = self.rpc.get_account(&pool).await.expect("pool account");
        let mut venue = CofferVenue::from_account(&pool, &account).expect("from_account");
        let cache = RpcClientCache::new(RpcClient::new_with_commitment(
            self.manifest.rpc.clone(),
            CommitmentConfig::confirmed(),
        ));
        venue.update_state(&cache).await.expect("update_state");
        venue
    }

    async fn pool_state(&self, pool: Pubkey) -> CofferPool {
        CofferPool::from_account_data(&self.rpc.get_account(&pool).await.unwrap().data).unwrap()
    }

    async fn token_balance(&self, mint: &Pubkey, tp: &Pubkey) -> u64 {
        let key = ata(&self.wallet.pubkey(), mint, tp);
        match self.rpc.get_account(&key).await {
            Ok(acc) => TokenAccount::unpack_from_slice(&acc.data)
                .map(|a| a.amount)
                .unwrap_or(0),
            Err(_) => 0,
        }
    }

    /// Send and confirm; on failure return the program's custom error code.
    async fn send(&self, ixs: &[Instruction]) -> Result<Signature, u32> {
        let bh = self.rpc.get_latest_blockhash().await.unwrap();
        let tx = Transaction::new_signed_with_payer(
            ixs,
            Some(&self.wallet.pubkey()),
            &[&self.wallet],
            bh,
        );
        match self.rpc.send_and_confirm_transaction(&tx).await {
            Ok(sig) => Ok(sig),
            Err(e) => match e.get_transaction_error() {
                Some(TransactionError::InstructionError(_, InstructionError::Custom(code))) => {
                    Err(code)
                }
                other => panic!("transaction failed without a custom code: {other:?} / {e}"),
            },
        }
    }

    /// `(block_time, compute_units_consumed)` of a confirmed transaction.
    async fn tx_meta(&self, sig: &Signature) -> (i64, u64) {
        let tx = self
            .rpc
            .get_transaction_with_config(
                sig,
                RpcTransactionConfig {
                    encoding: Some(UiTransactionEncoding::Json),
                    commitment: Some(CommitmentConfig::confirmed()),
                    max_supported_transaction_version: Some(0),
                },
            )
            .await
            .expect("confirmed transaction");
        let cu = tx
            .transaction
            .meta
            .and_then(|m| match m.compute_units_consumed {
                OptionSerializer::Some(v) => Some(v),
                _ => None,
            })
            .unwrap_or_default();
        (tx.block_time.unwrap_or(0), cu)
    }

    fn request(&self, venue: &CofferVenue, i: u8, j: u8, amount: u64) -> QuoteRequest {
        QuoteRequest {
            input_mint: venue.get_token(i as usize).unwrap().pubkey,
            output_mint: venue.get_token(j as usize).unwrap().pubkey,
            amount,
            swap_type: SwapType::ExactIn,
        }
    }

    /// Execute one swap for real (routed through the Titan router when
    /// `routed`, else the venue's own `swap` instruction) and compare the
    /// user's output delta and the post-swap pool account with the venue's
    /// prediction. When they differ, the quote is recomputed with the venue's
    /// clock set to the transaction's block time: the sell-off window is a
    /// function of the clock, so a rotation or carry-over decay between the
    /// quote and the landing block explains a difference exactly.
    async fn parity_swap(
        &self,
        venue: &CofferVenue,
        i: u8,
        j: u8,
        amount: u64,
        routed: bool,
    ) -> Parity {
        let request = self.request(venue, i, j, amount);
        let quote = venue.quote(request.clone()).expect("quote");
        if quote.amount == 0 {
            // Nothing fillable in this direction right now (window exhausted
            // by the previous sample): nothing to execute or compare.
            return Parity {
                amount_requested: amount,
                amount_executed: 0,
                partial: true,
                expected: 0,
                received: 0,
                error: None,
                pool_matches: true,
                surge_fee: 0,
                compute_units: 0,
                block_time: 0,
                clock_adjusted: None,
            };
        }
        let (dec_in, dec_out) = (
            venue.get_token(i as usize).unwrap().decimals as u8,
            venue.get_token(j as usize).unwrap().decimals as u8,
        );
        let predict = |now: i64| -> (u64, u64, CofferPool) {
            let outcome = quote_exact_in(venue.pool(), quote.amount, i, j, dec_in, dec_out, now)
                .expect("exact quote at the fillable amount");
            let mut predicted = *venue.pool();
            apply_swap(&mut predicted, i, j, &outcome).unwrap();
            (outcome.amount_out_user, outcome.surge_fee_amount, predicted)
        };
        let (expected, surge_fee, predicted) = predict(venue.now);

        let out_tp = venue.get_token(j as usize).unwrap().get_token_program();
        let before = self.token_balance(&request.output_mint, &out_tp).await;
        let executed = QuoteRequest {
            amount: quote.amount,
            ..request.clone()
        };
        let ixs = if routed {
            // A routed coffer leg needs more than the router's 200k default:
            // the router spends ~45k CU creating the TitanPDA ATAs and the
            // 4-segment surge fee costs the swap four extra curve evaluations.
            vec![
                ComputeBudgetInstruction::set_compute_unit_limit(600_000),
                build_route_ix(venue, &executed, self.wallet.pubkey()).unwrap(),
            ]
        } else {
            vec![
                venue
                    .generate_swap_instruction(executed, self.wallet.pubkey())
                    .unwrap(),
            ]
        };
        let sent = self.send(&ixs).await;
        let after = self.token_balance(&request.output_mint, &out_tp).await;
        let chain = self.pool_state(venue.market_id()).await;
        let received = after - before;
        let (mut block_time, mut compute_units) = (0, 0);
        let mut clock_adjusted = None;
        if let Ok(sig) = &sent {
            let (bt, cu) = self.tx_meta(sig).await;
            block_time = bt;
            compute_units = cu;
            if (received != expected || chain != predicted) && bt != venue.now {
                let (e2, _, p2) = predict(bt);
                clock_adjusted = Some(ClockAdjusted {
                    block_time_minus_now: bt - venue.now,
                    expected: e2,
                    matches: received == e2 && chain == p2,
                });
            }
        }
        Parity {
            amount_requested: amount,
            amount_executed: quote.amount,
            partial: quote.not_enough_liquidity,
            expected,
            received,
            error: sent.err(),
            pool_matches: chain == predicted,
            surge_fee,
            compute_units,
            block_time,
            clock_adjusted,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ClockAdjusted {
    block_time_minus_now: i64,
    expected: u64,
    matches: bool,
}

#[derive(Debug, Clone, Copy)]
struct Parity {
    amount_requested: u64,
    amount_executed: u64,
    partial: bool,
    expected: u64,
    received: u64,
    error: Option<u32>,
    pool_matches: bool,
    surge_fee: u64,
    compute_units: u64,
    block_time: i64,
    /// Set when the plain comparison failed and the venue's clock differed
    /// from the block time: the re-quote at the block time.
    clock_adjusted: Option<ClockAdjusted>,
}

impl Parity {
    /// Exact at the venue's clock.
    fn exact(&self) -> bool {
        self.error.is_none() && self.received == self.expected && self.pool_matches
    }
    /// Exact at the venue's clock, or exact once the venue clock is set to the
    /// block time (the window moved between quote and landing).
    fn ok(&self) -> bool {
        self.exact() || self.clock_adjusted.is_some_and(|c| c.matches)
    }
}

fn geometric(lb: u64, ub: u64, n: usize) -> Vec<u64> {
    let mut v: Vec<u64> = (0..n)
        .map(|k| {
            let t = k as f64 / (n - 1) as f64;
            (((lb as f64).ln() + t * ((ub as f64).ln() - (lb as f64).ln())).exp() as u64)
                .clamp(lb, ub)
        })
        .collect();
    v.dedup();
    v
}

fn headroom_of(venue: &CofferVenue, i: u8, j: u8) -> QuoteResult {
    venue
        .quote(QuoteRequest {
            input_mint: venue.get_token(i as usize).unwrap().pubkey,
            output_mint: venue.get_token(j as usize).unwrap().pubkey,
            amount: u64::MAX / 8,
            swap_type: SwapType::ExactIn,
        })
        .expect("probe quote")
}

// ---------------------------------------------------------------------------
// 1. Titan's shared suite on every static pool
// ---------------------------------------------------------------------------

const SUITE_TESTS: [&str; 8] = [
    "construction",
    "zero_input_spot_price",
    "bound_simulation",
    "random_samples",
    "monotone",
    "quoting_speed",
    "price_monotone",
    "mean_value_theorem",
];

/// Run one shared-suite function on its own OS thread with a private runtime
/// (the suite's futures hold a LiteSVM and are not `Send`), capturing a panic
/// as the failure message.
async fn run_suite_test(name: &str, pool: Pubkey) -> Result<(), String> {
    let name_owned = name.to_string();
    let handle = std::thread::spawn(move || {
        let config = SuiteConfig {
            pool,
            programs: vec![COFFER_PROGRAM_ID],
        };
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            match name_owned.as_str() {
                "construction" => common::construction::<CofferVenue>(&config).await,
                "zero_input_spot_price" => {
                    common::zero_input_spot_price::<CofferVenue>(&config).await
                }
                "bound_simulation" => common::bound_simulation::<CofferVenue>(&config).await,
                "random_samples" => common::random_samples::<CofferVenue>(&config).await,
                "monotone" => common::monotone::<CofferVenue>(&config).await,
                "quoting_speed" => common::quoting_speed::<CofferVenue>(&config).await,
                "price_monotone" => common::price_monotone::<CofferVenue>(&config).await,
                "mean_value_theorem" => common::mean_value_theorem::<CofferVenue>(&config).await,
                other => panic!("unknown suite test {other}"),
            }
        })
    });
    tokio::task::spawn_blocking(move || match handle.join() {
        Ok(()) => Ok(()),
        Err(payload) => Err(payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|| "panic".into())),
    })
    .await
    .unwrap()
}

/// The per-direction MVT residual with the input-atom-aware slack, for the
/// table (see README: the shared MVT tolerance is in output atoms only).
fn mvt_residual(venue: &CofferVenue) -> (usize, usize, f64, f64) {
    let mut template_violations = 0;
    let mut input_aware_violations = 0;
    let mut worst: f64 = 0.0;
    let mut worst_surge: f64 = 0.0;
    for (i, j) in venue.directions_num() {
        let Ok((lb, ub)) = venue.bounds(i, j) else {
            continue;
        };
        let (im, om) = (
            venue.get_token(i as usize).unwrap().pubkey,
            venue.get_token(j as usize).unwrap().pubkey,
        );
        let q = |x: u64| {
            venue
                .quote(QuoteRequest {
                    input_mint: im,
                    output_mint: om,
                    amount: x,
                    swap_type: SwapType::ExactIn,
                })
                .unwrap()
        };
        let grid = geometric(lb, ub, 64);
        for w in grid.windows(2) {
            let (a, b) = (w[0], w[1]);
            if b <= a {
                continue;
            }
            let (qa, qb) = (q(a), q(b));
            if qb.expected_output <= qa.expected_output {
                continue;
            }
            let chord = (qb.expected_output - qa.expected_output) as f64 / (b - a) as f64;
            let atol = 2.0 / (b - a) as f64;
            if chord > qa.price * (1.0 + 1e-5) + atol || chord < qb.price * (1.0 - 1e-5) - atol {
                template_violations += 1;
                let rel = ((chord / qa.price) - 1.0).max(1.0 - chord / qb.price);
                worst = worst.max(rel);
            }
            let atol2 = (2.0 + qa.price) / (b - a) as f64;
            let over = (chord - (qa.price * (1.0 + 1e-5) + atol2)) / qa.price;
            let under = ((qb.price * (1.0 - 1e-5) - atol2) - chord) / qb.price;
            if over > 0.0 || under > 0.0 {
                input_aware_violations += 1;
                worst_surge = worst_surge.max(over).max(under);
            }
        }
    }
    (
        template_violations,
        input_aware_violations,
        worst,
        worst_surge,
    )
}

async fn shared_suite(only: Option<&str>) {
    let Some(stand) = stand_or_skip("shared_suite").await else {
        return;
    };
    let tests: Vec<&str> = match only {
        Some(t) => vec![t],
        None => SUITE_TESTS.to_vec(),
    };
    type Row = (String, Vec<Result<(), String>>, String);
    let mut table: Vec<Row> = Vec::new();
    for pool in stand.manifest.pools.iter().filter(|p| !p.dynamic) {
        let address: Pubkey = pool.address.parse().unwrap();
        let venue = stand.venue(address).await;
        let dirs = venue.directions_num();
        let mut results = Vec::new();
        for t in &tests {
            results.push(run_suite_test(t, address).await);
        }
        let (tv, iv, worst, worst_surge) = mvt_residual(&venue);
        let note = format!(
            "dirs={} mvt: template_violations={tv} (worst {worst:.2e}) beyond_input_atom_slack={iv} (worst {worst_surge:.2e})",
            dirs.len()
        );
        table.push((pool.case.clone(), results, note));
    }

    println!(
        "\n== Titan shared suite over the local stand (SOLANA_RPC_URL={}) ==",
        stand.manifest.rpc
    );
    print!("  {:<20}", "case");
    for t in &tests {
        print!(" {:<10}", &t[..t.len().min(10)]);
    }
    println!("  notes");
    let mut failures = Vec::new();
    for (case, results, note) in &table {
        print!("  {case:<20}");
        for (t, r) in tests.iter().zip(results) {
            match r {
                Ok(()) => print!(" {:<10}", "ok"),
                Err(msg) => {
                    print!(" {:<10}", "FAIL");
                    failures.push(format!("{case}/{t}: {}", msg.lines().next().unwrap_or("")));
                }
            }
        }
        println!("  {note}");
    }
    for f in &failures {
        println!("  ! {f}");
    }
    assert!(
        failures.is_empty(),
        "{} suite failure(s) (see table)",
        failures.len()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shared_suite_all_tests() {
    shared_suite(None).await;
}

/// `construction` alone, meant for `--profile release-debug` where the
/// no-allocation guard is active.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shared_suite_construction_no_alloc() {
    shared_suite(Some("construction")).await;
}

// ---------------------------------------------------------------------------
// 2. Real routed transactions through the Titan router
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn real_routed_transactions() {
    let Some(stand) = stand_or_skip("real_routed_transactions").await else {
        return;
    };
    let mut rows = Vec::new();
    let mut bad = 0;
    for pool in stand.manifest.pools.iter().filter(|p| !p.dynamic) {
        let address: Pubkey = pool.address.parse().unwrap();
        let mut venue = stand.venue(address).await;
        let mut sent = 0;
        let mut mismatches = Vec::new();
        let mut partial_fills = 0;
        let mut surged = 0;
        let mut clock_adjusted = 0;
        let mut max_cu = 0;

        if !pool.pool_enabled || !pool.swaps_enabled {
            // Every direction is unavailable: quoting refuses, the program reverts.
            let q = venue.quote(stand.request(&venue, 0, 1, 1_000_000));
            assert!(
                matches!(q, Err(TradingVenueError::InactivePoolError(..))),
                "{}: {q:?}",
                pool.case
            );
            let ix = venue
                .generate_swap_instruction(
                    stand.request(&venue, 0, 1, 1_000_000),
                    stand.wallet.pubkey(),
                )
                .unwrap();
            let expected = if pool.pool_enabled {
                ErrorCode::SwapsDisabled
            } else {
                ErrorCode::PoolDisabled
            };
            assert_eq!(
                stand.send(&[ix]).await,
                Err(expected.code()),
                "{}",
                pool.case
            );
            rows.push((
                pool.case.clone(),
                format!(
                    "quote refused (InactivePoolError), program reverts {} ({})",
                    expected.name(),
                    expected.code()
                ),
            ));
            continue;
        }

        eprintln!("real_routed_transactions: {} ...", pool.case);
        let mut nothing_fillable = 0;
        let mut exhausted_directions = 0;
        for (i, j) in venue.directions_num() {
            // Earlier samples may have filled this input token's window.
            venue = stand.venue(address).await;
            if !venue.directions_num().contains(&(i, j)) {
                exhausted_directions += 1;
                continue;
            }
            let (lb, ub) = venue.bounds(i, j).unwrap();
            let (in_mint, in_tp) = (
                venue.get_token(i as usize).unwrap().pubkey,
                venue.get_token(i as usize).unwrap().get_token_program(),
            );
            let wallet_has = stand.token_balance(&in_mint, &in_tp).await;
            let ub = ub.min(wallet_has / 2);
            for amount in geometric(lb.max(1), ub.max(lb.max(1)), 3) {
                venue = stand.venue(address).await;
                let p = stand.parity_swap(&venue, i, j, amount, true).await;
                if p.amount_executed == 0 {
                    nothing_fillable += 1;
                    continue;
                }
                sent += 1;
                if p.partial {
                    partial_fills += 1;
                }
                if p.surge_fee > 0 {
                    surged += 1;
                }
                if !p.exact() && p.ok() {
                    clock_adjusted += 1;
                }
                max_cu = max_cu.max(p.compute_units);
                if !p.ok() {
                    mismatches.push(format!("{i}->{j} {p:?}"));
                }
            }
        }
        // Deactivated input token: the direction is absent, the program reverts.
        for &slot in &pool.inactive_tokens {
            assert!(
                venue.directions_num().iter().all(|(i, _)| *i != slot),
                "{}: inactive slot listed",
                pool.case
            );
            let other = if slot == 0 { 1 } else { 0 };
            let q = venue.quote(stand.request(&venue, slot, other, 1_000_000));
            assert!(
                matches!(q, Err(TradingVenueError::AmmMethodError(_))),
                "{}: {q:?}",
                pool.case
            );
            let ix = venue
                .generate_swap_instruction(
                    stand.request(&venue, slot, other, 1_000_000),
                    stand.wallet.pubkey(),
                )
                .unwrap();
            assert_eq!(
                stand.send(&[ix]).await,
                Err(ErrorCode::TokenInactive.code()),
                "{}",
                pool.case
            );
        }
        if !mismatches.is_empty() {
            bad += 1;
        }
        rows.push((
            pool.case.clone(),
            format!(
                "{sent} routed txs, {} exact, {clock_adjusted} exact only at block time, {partial_fills} partial-fill quotes executed at the reported amount, {nothing_fillable} samples / {exhausted_directions} directions with nothing fillable (window exhausted by earlier samples), {surged} with surge fee, max {max_cu} CU{}",
                sent - mismatches.len() - clock_adjusted,
                if mismatches.is_empty() { String::new() } else { format!("; MISMATCH {}", mismatches.join(" | ")) }
            ),
        ));
    }
    println!("\n== real routed transactions (swap_route_v3 on the validator) ==");
    for (case, r) in &rows {
        println!("  {case:<20} {r}");
    }
    assert_eq!(bad, 0, "{bad} pool(s) with on-chain/quote mismatches");
}

// ---------------------------------------------------------------------------
// 3. Dynamic sequences
// ---------------------------------------------------------------------------

/// Sell BONK in chunks until the window is full; each chunk must match
/// on-chain (payout and full pool state), the full window must be reported
/// as unavailable and reverted on-chain, and the buy side must keep working.
/// Returns (sold, per-chunk parity, output-token protocol bucket right after the sells).
async fn fill_window(
    stand: &Stand,
    case: &str,
    routed: bool,
    chunks: u64,
) -> (u64, Vec<Parity>, u64) {
    let pool: Pubkey = stand.pool(case).address.parse().unwrap();
    let mut venue = stand.venue(pool).await;
    let probe = headroom_of(&venue, 0, 1);
    assert!(
        probe.not_enough_liquidity,
        "{case}: probe should exceed the cap"
    );
    let cap = probe.amount;
    assert!(cap > 0);
    let chunk = cap / chunks;
    let mut log = Vec::new();
    let mut sold = 0u64;
    loop {
        venue = stand.venue(pool).await;
        let head = headroom_of(&venue, 0, 1).amount;
        if head == 0 {
            break;
        }
        let amount = chunk.min(head);
        let p = stand.parity_swap(&venue, 0, 1, amount, routed).await;
        if p.amount_executed == 0 {
            break; // the venue refuses the remaining dust (zero output)
        }
        assert!(p.ok(), "{case}: chunk {amount} mismatch: {p:?}");
        if !p.exact() {
            println!(
                "  {case}: chunk {amount} exact only at block time ({:?})",
                p.clock_adjusted.unwrap()
            );
        }
        sold += p.amount_executed;
        log.push(p);
    }
    // Window full (or only dust the venue refuses because it would pay 0).
    venue = stand.venue(pool).await;
    let bucket_after_sells = venue.pool().tokens[1].dynamics.protocol_fees_owed;
    let raw_head = selloff_headroom(venue.pool(), 0, venue.now)
        .unwrap()
        .unwrap_or(0);
    assert!(
        !venue.directions_num().contains(&(0, 1)),
        "{case}: full window still listed"
    );
    assert!(
        venue.directions_num().contains(&(1, 0)),
        "{case}: buy side must stay listed"
    );
    assert!(
        venue.bounds(0, 1).is_err(),
        "{case}: bounds must report no quotable range"
    );
    let q = venue.quote(stand.request(&venue, 0, 1, 1)).unwrap();
    assert!(
        q.not_enough_liquidity && q.amount == 0 && q.expected_output == 0,
        "{case}: {q:?}"
    );
    // One atom past the cap reverts on-chain.
    let ix = venue
        .generate_swap_instruction(
            stand.request(&venue, 0, 1, raw_head + 1),
            stand.wallet.pubkey(),
        )
        .unwrap();
    assert_eq!(
        stand.send(&[ix]).await,
        Err(ErrorCode::MaxSelloffExceeded.code()),
        "{case}"
    );
    if raw_head > 0 {
        // The dust the venue refuses is accepted by the program but pays 0
        // (100% surge at full fill); after it the cap is exactly reached.
        let (out_mint, out_tp) = (venue.get_token(1).unwrap().pubkey, spl_token::ID);
        let before = stand.token_balance(&out_mint, &out_tp).await;
        let ix = venue
            .generate_swap_instruction(stand.request(&venue, 0, 1, raw_head), stand.wallet.pubkey())
            .unwrap();
        stand.send(&[ix]).await.unwrap();
        let paid = stand.token_balance(&out_mint, &out_tp).await - before;
        println!(
            "  {case}: remaining {raw_head} atoms refused by the venue (zero output); on-chain they paid {paid} atoms"
        );
        venue = stand.venue(pool).await;
        let ix = venue
            .generate_swap_instruction(stand.request(&venue, 0, 1, 1), stand.wallet.pubkey())
            .unwrap();
        assert_eq!(
            stand.send(&[ix]).await,
            Err(ErrorCode::MaxSelloffExceeded.code()),
            "{case}"
        );
        sold += raw_head;
    }
    // Buy side unaffected by BONK's cap.
    let (lb, ub) = venue.bounds(1, 0).unwrap();
    let p = stand
        .parity_swap(&venue, 1, 0, geometric(lb, ub, 3)[1], routed)
        .await;
    assert!(p.ok(), "{case}: buy side mismatch {p:?}");
    (sold, log, bucket_after_sells)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dynamic_window_filling_direct_swaps() {
    let Some(stand) = stand_or_skip("dynamic_window_filling_direct_swaps").await else {
        return;
    };
    println!("\n== dynamic: window filling with direct swaps ==");
    for case in ["dyn_b", "dyn_c", "dyn_8020_c"] {
        let pool: Pubkey = stand.pool(case).address.parse().unwrap();
        let before = stand.pool_state(pool).await;
        let (sold, log, bucket_after_sells) = fill_window(&stand, case, false, 7).await;
        let after = stand.pool_state(pool).await;
        let surge_total: u64 = log.iter().map(|p| p.surge_fee).sum();
        let accrued = bucket_after_sells - before.tokens[1].dynamics.protocol_fees_owed;
        // The LP swap-fee protocol cut lands on the INPUT token; the surge fee
        // is the only thing that lands on the OUTPUT token's bucket.
        assert_eq!(
            accrued, surge_total,
            "{case}: surge accrual in USDC protocol bucket"
        );
        assert_eq!(
            after.tokens[0].dynamics.current_selloff, sold,
            "{case}: window accumulator"
        );
        println!(
            "  {case:<12} {} chunks, sold {sold} (cap {}), surge fee accrued {accrued} USDC atoms across {} surged chunks, window full -> unavailable + MaxSelloffExceeded, buy side ok",
            log.len(),
            after.tokens[0].dynamics.selloff_vb_snapshot
                * stand.pool(case).selloff[0].max_selloff_pct as u64
                / 10_000,
            log.iter().filter(|p| p.surge_fee > 0).count()
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dynamic_window_filling_routed() {
    let Some(stand) = stand_or_skip("dynamic_window_filling_routed").await else {
        return;
    };
    println!("\n== dynamic: window filling through the Titan router (config b, kink at 50%) ==");
    let (sold, log, _) = fill_window(&stand, "dyn_router", true, 10).await;
    for (k, p) in log.iter().enumerate() {
        println!(
            "  chunk {k}: in {} out {} surge {}",
            p.amount_executed, p.received, p.surge_fee
        );
    }
    println!("  sold {sold}; {} routed chunks all exact", log.len());
}

/// 100% surge above the threshold: the venue reports the exhaustion point as
/// the fillable amount; the program pays nothing above it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dynamic_full_fee_exhaustion_point() {
    let Some(stand) = stand_or_skip("dynamic_full_fee_exhaustion_point").await else {
        return;
    };
    let case = "dyn_d";
    let pool: Pubkey = stand.pool(case).address.parse().unwrap();
    let venue = stand.venue(pool).await;
    let cap = venue.pool().tokens[0].dynamics.virtual_balance / 10;
    let threshold_point = cap * 8 / 10;
    let q = venue.quote(stand.request(&venue, 0, 1, cap)).unwrap();
    assert!(q.not_enough_liquidity, "{q:?}");
    assert_eq!(
        q.amount, threshold_point,
        "exhaustion point is the 80% threshold"
    );
    assert!(q.price > 0.0);
    let (_, ub) = venue.bounds(0, 1).unwrap();
    assert!(
        ub <= threshold_point && threshold_point - ub <= 100,
        "ub {ub} vs {threshold_point}"
    );
    // Execute at the exhaustion point: exact parity.
    let p = stand
        .parity_swap(&venue, 0, 1, threshold_point, false)
        .await;
    assert!(p.ok(), "{p:?}");
    // Now sell MORE than the quote admits (10% of the cap past the threshold)
    // for real: the program accepts it and pays only the below-threshold output.
    let venue = stand.venue(pool).await;
    let more = cap / 10;
    let q = venue.quote(stand.request(&venue, 0, 1, more)).unwrap();
    assert!(
        q.not_enough_liquidity && q.amount == 0 && q.expected_output == 0,
        "{q:?}"
    );
    let (out_mint, out_tp) = (venue.get_token(1).unwrap().pubkey, spl_token::ID);
    let before = stand.token_balance(&out_mint, &out_tp).await;
    let ix = venue
        .generate_swap_instruction(stand.request(&venue, 0, 1, more), stand.wallet.pubkey())
        .unwrap();
    stand.send(&[ix]).await.unwrap();
    let received = stand.token_balance(&out_mint, &out_tp).await - before;
    println!(
        "\n== dynamic: 100% surge (dyn_d) == exhaustion point {threshold_point} executed exactly; selling {more} more on-chain paid {received} atoms (quote: 0 fillable, 0 output)"
    );
    assert_eq!(
        received, 0,
        "above the exhaustion point the program pays nothing"
    );
}

/// 20-second window: fill it, watch the real clock rotate it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dynamic_real_time_rotation() {
    let Some(stand) = stand_or_skip("dynamic_real_time_rotation").await else {
        return;
    };
    let case = "dyn_short";
    let pool: Pubkey = stand.pool(case).address.parse().unwrap();
    let period = stand.pool(case).selloff[0].period_length as i64;
    // Start from a fresh window: wait until two periods have passed since the
    // window opened (pool creation), so the state is clean.
    let mut venue = stand.venue(pool).await;
    let opened = venue.pool().tokens[0].dynamics.window_start_timestamp;
    let now = venue.now;
    if now < opened + 2 * period + 1 {
        tokio::time::sleep(Duration::from_secs((opened + 2 * period + 1 - now) as u64)).await;
    }
    let t0 = Instant::now();
    let (sold, log, _) = fill_window(&stand, case, false, 4).await;
    println!("\n== dynamic: real-time rotation (dyn_short, period {period} s) ==");
    println!(
        "  filled the window: sold {sold} in {} chunks ({:.1} s)",
        log.len(),
        t0.elapsed().as_secs_f64()
    );
    venue = stand.venue(pool).await;
    let d = venue.pool().tokens[0].dynamics;
    let window_start = d.window_start_timestamp;
    // Wait for one rotation.
    let wake = window_start + period + 2;
    let now = stand.venue(pool).await.now;
    if wake > now {
        tokio::time::sleep(Duration::from_secs((wake - now) as u64)).await;
    }
    venue = stand.venue(pool).await;
    assert!(
        venue.directions_num().contains(&(0, 1)),
        "direction must come back after one period"
    );
    let head = headroom_of(&venue, 0, 1).amount;
    assert!(
        head > 0 && head < sold,
        "after one rotation the carry-over leaves partial headroom: {head} vs sold {sold}"
    );
    let p = stand.parity_swap(&venue, 0, 1, head, false).await;
    assert!(p.ok(), "sell exactly the post-rotation headroom: {p:?}");
    venue = stand.venue(pool).await;
    let ix = venue
        .generate_swap_instruction(stand.request(&venue, 0, 1, 1), stand.wallet.pubkey())
        .unwrap();
    let r = stand.send(&[ix]).await;
    // The weighted carry-over decays every second, so one more atom may already
    // fit by the time the tx lands; both outcomes are consistent with the quote
    // at the moment it was made.
    println!(
        "  after one period: headroom {head} sold exactly (parity ok); +1 atom -> {:?} (clock moved {} s since the quote)",
        r,
        stand.venue(pool).await.now - venue.now
    );
    // The post-rotation sell rotated the window once more (it ran at
    // elapsed >= period): count the "two periods" from the NEW window start.
    let window_start = stand.venue(pool).await.pool().tokens[0]
        .dynamics
        .window_start_timestamp;
    let wake = window_start + 2 * period + 2;
    let now = stand.venue(pool).await.now;
    if wake > now {
        tokio::time::sleep(Duration::from_secs((wake - now) as u64)).await;
    }
    venue = stand.venue(pool).await;
    let head2 = headroom_of(&venue, 0, 1).amount;
    let cap_now = venue.pool().tokens[0].dynamics.virtual_balance / 10;
    println!("  after two periods: headroom {head2} == cap {cap_now} (fresh window)");
    assert_eq!(head2, cap_now);
    let p = stand.parity_swap(&venue, 0, 1, cap_now / 3, false).await;
    assert!(p.ok(), "{p:?}");
}

/// Deactivate / reactivate BONK, disable / enable swaps and the pool.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dynamic_toggles() {
    let Some(stand) = stand_or_skip("dynamic_toggles").await else {
        return;
    };
    let case = "dyn_toggle";
    let pool: Pubkey = stand.pool(case).address.parse().unwrap();
    let me = stand.wallet.pubkey();
    println!("\n== dynamic: toggles (dyn_toggle) ==");

    // deactivate BONK
    stand
        .send(&[set_token_active_ix(LOCAL_CONFIG, pool, me, 0, false)])
        .await
        .unwrap();
    let venue = stand.venue(pool).await;
    assert_eq!(venue.directions_num(), vec![(1, 0)]);
    assert!(matches!(
        venue.quote(stand.request(&venue, 0, 1, 1_000_000)),
        Err(TradingVenueError::AmmMethodError(_))
    ));
    let ix = venue
        .generate_swap_instruction(stand.request(&venue, 0, 1, 1_000_000), me)
        .unwrap();
    assert_eq!(
        stand.send(&[ix]).await,
        Err(ErrorCode::TokenInactive.code())
    );
    let (lb, ub) = venue.bounds(1, 0).unwrap();
    let p = stand
        .parity_swap(&venue, 1, 0, geometric(lb, ub, 3)[1], true)
        .await;
    assert!(p.ok(), "buying the deactivated token via the router: {p:?}");
    println!(
        "  BONK deactivated: direction absent, quote refuses (TokenInactive), program reverts 6054, buying BONK via router exact"
    );
    // reactivate
    stand
        .send(&[set_token_active_ix(LOCAL_CONFIG, pool, me, 0, true)])
        .await
        .unwrap();
    let venue = stand.venue(pool).await;
    assert!(venue.directions_num().contains(&(0, 1)));
    let (lb, ub) = venue.bounds(0, 1).unwrap();
    let p = stand
        .parity_swap(&venue, 0, 1, geometric(lb, ub, 3)[1], true)
        .await;
    assert!(p.ok(), "{p:?}");
    println!("  BONK reactivated: direction back, routed sell exact");

    // swaps disabled
    stand
        .send(&[set_swaps_enabled_ix(LOCAL_CONFIG, pool, me, false)])
        .await
        .unwrap();
    let venue = stand.venue(pool).await;
    assert!(matches!(
        venue.quote(stand.request(&venue, 0, 1, 1_000)),
        Err(TradingVenueError::InactivePoolError(..))
    ));
    let ix = venue
        .generate_swap_instruction(stand.request(&venue, 0, 1, 1_000), me)
        .unwrap();
    assert_eq!(
        stand.send(&[ix]).await,
        Err(ErrorCode::SwapsDisabled.code())
    );
    stand
        .send(&[set_swaps_enabled_ix(LOCAL_CONFIG, pool, me, true)])
        .await
        .unwrap();
    println!(
        "  swaps disabled: InactivePoolError, program reverts SwapsDisabled (6021); re-enabled"
    );

    // pool disabled (protocol admin)
    stand
        .send(&[set_pool_enabled_ix(LOCAL_CONFIG, pool, me, false)])
        .await
        .unwrap();
    let venue = stand.venue(pool).await;
    assert!(matches!(
        venue.quote(stand.request(&venue, 0, 1, 1_000)),
        Err(TradingVenueError::InactivePoolError(..))
    ));
    let ix = venue
        .generate_swap_instruction(stand.request(&venue, 0, 1, 1_000), me)
        .unwrap();
    assert_eq!(stand.send(&[ix]).await, Err(ErrorCode::PoolDisabled.code()));
    stand
        .send(&[set_pool_enabled_ix(LOCAL_CONFIG, pool, me, true)])
        .await
        .unwrap();
    let venue = stand.venue(pool).await;
    let p = stand.parity_swap(&venue, 0, 1, 1_000_000, true).await;
    assert!(p.ok(), "{p:?}");
    println!(
        "  pool disabled: InactivePoolError, program reverts PoolDisabled (6020); re-enabled, routed swap exact"
    );
}
