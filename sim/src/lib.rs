//! Twin simulator for the Terminal engine (this competition's season-5 compat ruleset).
//! Fidelity contract: sim/MECHANICS.md. Every ⚠ item there maps to a diff-harness case.

pub mod config;
pub mod geo;
pub mod state;
pub mod path;
pub mod engine;
pub mod replay;
pub mod nn;

#[cfg(feature = "python")]
pub mod py;
